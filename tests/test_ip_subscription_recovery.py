"""Subscription recovery policy, independent of network and wall-clock time."""

import asyncio
from unittest import mock

import pytest

from aiohomekit.controller.ip.pairing import IpPairing
from aiohomekit.exceptions import (
    AccessoryDisconnectedError,
    HttpErrorResponse,
    RequestNotSentError,
    RequestOutcomeUnknownError,
)
from aiohomekit.model import Accessories, AccessoriesState
from aiohomekit.protocol.statuscodes import HapStatusCode
from tests.subscription_clock import ControlledLoop


@pytest.fixture
async def subscription_pairing():
    controller = mock.Mock()
    controller._char_cache.get_map.return_value = None
    pairing = IpPairing(
        controller,
        {"AccessoryPairingID": "00:00:00:00:00:01", "AccessoryIP": "127.0.0.1", "AccessoryPort": 9},
    )
    pairing.connection.transport = mock.Mock()
    pairing.connection.protocol = mock.Mock()
    pairing.connection.is_secure = True
    pairing.connection.put_json = mock.AsyncMock(return_value={})
    pairing.connection.get_json = mock.AsyncMock(
        side_effect=lambda *args: {"characteristics": [{"aid": 1, "iid": 9, "value": False}]}
    )
    pairing._accessories_state = AccessoriesState(Accessories(), 1)
    pairing._subscription_loop = ControlledLoop()
    yield pairing
    await pairing.close()


async def test_disconnect_does_not_infer_unsupported(subscription_pairing):
    pairing = subscription_pairing
    pairing.connection.put_json.side_effect = AccessoryDisconnectedError("Connection closed")
    assert await pairing.subscribe([(1, 9)]) == {}
    assert pairing.subscriptions == {(1, 9)}
    assert pairing.supports_subscribe


async def finish_worker(pairing):
    task = pairing._subscription_task
    assert task is not None
    return await asyncio.wait_for(asyncio.shield(task), 1)


async def advance_recovery(pairing):
    pairing._subscription_loop.advance()
    return await finish_worker(pairing)


async def test_backoff_caps_at_one_hour_without_disabling_push(subscription_pairing):
    pairing = subscription_pairing
    pairing.connection.put_json.side_effect = RequestOutcomeUnknownError("Response lost")
    await pairing.subscribe([(1, 9)])
    expected_delays = [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 3600, 3600]
    for attempts, delay in enumerate(expected_delays, 1):
        assert pairing._subscription_retry_at - pairing._subscription_loop.time() == delay
        timer = pairing._subscription_timer
        await pairing.subscribe([(1, 9), (1, 10)])
        assert pairing._subscription_timer is timer
        assert pairing.connection.put_json.await_count == attempts
        assert await pairing.get_characteristics([(1, 9)]) == {(1, 9): {"value": False}}
        assert pairing._subscription_retry_at == timer.deadline
        assert pairing.supports_subscribe
        await advance_recovery(pairing)
    assert pairing.connection.put_json.await_count == len(expected_delays) + 1


async def test_reconnect_and_discovery_do_not_bypass_cooldown(subscription_pairing):
    pairing = subscription_pairing
    pairing.connection.put_json.side_effect = RequestOutcomeUnknownError("Response lost")
    await pairing.subscribe([(1, 9)])
    deadline = pairing._subscription_retry_at
    pairing.connection.protocol = mock.Mock()
    await pairing.connection_made(True)
    pairing._async_description_update(None)
    assert pairing.connection.put_json.await_count == 1
    assert pairing._subscription_retry_at == deadline
    assert pairing._subscription_retry_delay == 10
    pairing.connection.put_json.side_effect = None
    await advance_recovery(pairing)
    assert pairing._acknowledged_subscriptions == {(1, 9)}
    assert pairing._subscription_retry_at == 0
    assert pairing._subscription_retry_delay == 5


async def test_partial_batch_success_does_not_reset_backoff(subscription_pairing):
    pairing = subscription_pairing
    pairing.connection.put_json.side_effect = [
        {},
        RequestOutcomeUnknownError("Second accessory disconnected"),
        {},
        RequestOutcomeUnknownError("Second accessory disconnected"),
        {},
        {},
    ]
    await pairing.subscribe([(2, 9), (1, 9)])
    assert pairing._acknowledged_subscriptions == {(1, 9)}
    for delay in [5, 10]:
        assert pairing._subscription_retry_at - pairing._subscription_loop.time() == delay
        pairing.connection.protocol = mock.Mock()
        await pairing.connection_made(True)
        await advance_recovery(pairing)
    assert pairing.connection.put_json.await_count == 6
    assert pairing._acknowledged_subscriptions == {(1, 9), (2, 9)}
    assert pairing._subscription_retry_delay == 5
    assert pairing._subscription_timer is None


@pytest.mark.parametrize(
    "status",
    [
        HapStatusCode.UNABLE_TO_COMMUNICATE,
        HapStatusCode.RESOURCE_BUSY,
        HapStatusCode.OUT_OF_RESOURCES,
        HapStatusCode.TIMED_OUT,
        HapStatusCode.NOT_ALLOWED_IN_CURRENT_STATE,
    ],
)
async def test_retryable_hap_error_recovers_on_healthy_session(subscription_pairing, status):
    pairing = subscription_pairing
    session = pairing.connection.protocol
    pairing.connection.put_json.return_value = {"status": status.value}
    result = await pairing.subscribe([(1, 9)])
    assert result[(1, 9)]["status"] == status.value
    pairing.connection.put_json.return_value = {}
    await advance_recovery(pairing)
    assert pairing.connection.protocol is session
    pairing.connection.transport.close.assert_not_called()
    assert pairing._acknowledged_subscriptions == {(1, 9)}


async def test_unsupported_characteristic_is_cached_until_config_changes(subscription_pairing):
    pairing = subscription_pairing
    pairing.connection.put_json.return_value = {
        "characteristics": [
            {"aid": 1, "iid": 9, "status": 0},
            {"aid": 1, "iid": 10, "status": HapStatusCode.NOTIFICATION_NOT_SUPPORTED.value},
        ]
    }
    result = await pairing.subscribe([(1, 9), (1, 10)])
    assert result[(1, 10)]["status"] == HapStatusCode.NOTIFICATION_NOT_SUPPORTED.value
    pairing.connection.put_json.return_value = {}
    await pairing.subscribe([(1, 10)])
    assert pairing.connection.put_json.await_count == 1
    pairing.connection.protocol = mock.Mock()
    await pairing.connection_made(True)
    await finish_worker(pairing)
    assert pairing.connection.put_json.call_args.args[1] == {
        "characteristics": [{"aid": 1, "iid": 9, "ev": True}]
    }
    with mock.patch.object(pairing, "list_accessories_and_characteristics", mock.AsyncMock()):
        await pairing._process_config_changed(2)
    await finish_worker(pairing)
    assert pairing._unsupported_subscriptions == {}
    assert pairing._acknowledged_subscriptions == {(1, 9), (1, 10)}


@pytest.mark.parametrize("status", [HapStatusCode.INVALID_VALUE, HapStatusCode.INSUFFICIENT_AUTH])
async def test_terminal_hap_error_needs_explicit_call_or_new_session(subscription_pairing, status):
    pairing = subscription_pairing
    pairing.connection.put_json.return_value = {"status": status.value}
    await pairing.subscribe([(1, 9)])
    assert pairing._subscription_timer is None
    await pairing.connection_made(True)
    assert pairing.connection.put_json.await_count == 1
    await pairing.subscribe([(1, 9)])
    assert pairing.connection.put_json.await_count == 2
    pairing.connection.protocol = mock.Mock()
    pairing.connection.put_json.return_value = {}
    await pairing.connection_made(True)
    await finish_worker(pairing)
    assert pairing.connection.put_json.await_count == 3


async def test_http_rejection_does_not_schedule_transport_recovery(subscription_pairing, caplog):
    pairing = subscription_pairing
    session = pairing.connection.protocol
    pairing.connection.put_json.side_effect = HttpErrorResponse("Rejected", mock.Mock(code=400))
    await pairing.subscribe([(1, 9)])
    assert pairing._subscription_timer is None
    assert pairing._subscription_retry_delay == 5
    assert pairing.connection.protocol is session
    assert "HTTP 400" in caplog.text
    pairing.connection.put_json.side_effect = None
    await pairing.subscribe([(1, 9)])
    assert pairing._acknowledged_subscriptions == {(1, 9)}


async def test_not_sent_does_not_advance_subscription_backoff(subscription_pairing):
    pairing = subscription_pairing
    pairing.connection.put_json.side_effect = RequestNotSentError("Transport already closed")

    def connection_lost(error):
        pairing.connection.protocol = None
        pairing.connection.transport = None
        pairing.connection.is_secure = False

    with mock.patch.object(pairing.connection, "_connection_lost", side_effect=connection_lost) as lost:
        await pairing.subscribe([(1, 9)])
    lost.assert_called_once()
    assert pairing._subscription_retry_delay == 5
    assert pairing._subscription_retry_at == 0
    assert pairing._subscription_timer is None
    pairing.connection.protocol = mock.Mock()
    pairing.connection.transport = mock.Mock()
    pairing.connection.is_secure = True
    pairing.connection.put_json.side_effect = None
    await pairing.connection_made(True)
    await finish_worker(pairing)
    assert pairing._acknowledged_subscriptions == {(1, 9)}


async def test_overlapping_callers_coalesce_without_duplicate_requests(subscription_pairing):
    pairing = subscription_pairing
    started, release = asyncio.Event(), asyncio.Event()

    async def put_json(*args):
        started.set()
        await release.wait()
        return {}

    pairing.connection.put_json.side_effect = put_json
    first = asyncio.create_task(pairing.subscribe([(1, 9)]))
    await asyncio.wait_for(started.wait(), 1)
    worker = pairing._subscription_task
    second = asyncio.create_task(pairing.subscribe([(1, 9)]))
    release.set()
    await asyncio.wait_for(asyncio.gather(first, second), 1)
    assert worker.done()
    assert pairing.connection.put_json.await_count == 1


async def test_unsubscribe_during_restoration_does_not_resurrect_intent(subscription_pairing):
    pairing = subscription_pairing
    started, release = asyncio.Event(), asyncio.Event()

    async def put_json(*args):
        started.set()
        await release.wait()
        return {}

    pairing.connection.put_json.side_effect = put_json
    subscribe = asyncio.create_task(pairing.subscribe([(1, 9)]))
    await asyncio.wait_for(started.wait(), 1)
    unsubscribe = asyncio.create_task(pairing.unsubscribe([(1, 9)]))
    release.set()
    await asyncio.wait_for(asyncio.gather(subscribe, unsubscribe), 1)
    assert pairing.subscriptions == set()
    assert pairing._acknowledged_subscriptions == set()
    assert pairing._subscription_timer is None
    assert [
        call.args[1]["characteristics"][0]["ev"] for call in pairing.connection.put_json.call_args_list
    ] == [True, False]


async def test_caller_cancellation_does_not_cancel_worker(subscription_pairing):
    pairing = subscription_pairing
    started, release = asyncio.Event(), asyncio.Event()

    async def put_json(*args):
        started.set()
        await release.wait()
        return {}

    pairing.connection.put_json.side_effect = put_json
    caller = asyncio.create_task(pairing.subscribe([(1, 9)]))
    await asyncio.wait_for(started.wait(), 1)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert not pairing._subscription_task.cancelled()
    release.set()
    await finish_worker(pairing)
    assert pairing._acknowledged_subscriptions == {(1, 9)}


async def test_old_session_acknowledgement_is_ignored(subscription_pairing):
    pairing = subscription_pairing
    started, release = asyncio.Event(), asyncio.Event()

    async def put_json(*args):
        if pairing.connection.put_json.await_count == 1:
            started.set()
            await release.wait()
            return {"status": HapStatusCode.NOTIFICATION_NOT_SUPPORTED.value}
        return {}

    pairing.connection.put_json.side_effect = put_json
    caller = asyncio.create_task(pairing.subscribe([(1, 9)]))
    await asyncio.wait_for(started.wait(), 1)
    pairing.connection.protocol = mock.Mock()
    await pairing.connection_made(True)
    release.set()
    assert await asyncio.wait_for(caller, 1) == {}
    if pairing._subscription_task:
        await finish_worker(pairing)
    assert pairing.connection.put_json.await_count == 2
    assert pairing._unsupported_subscriptions == {}
    assert pairing._acknowledged_subscriptions == {(1, 9)}


async def test_close_cancels_timer_and_reopening_restores_intent(subscription_pairing):
    pairing = subscription_pairing
    pairing.connection.put_json.side_effect = RequestOutcomeUnknownError("Response lost")
    await pairing.subscribe([(1, 9)])
    timer = pairing._subscription_timer
    await pairing.close()
    assert timer.cancelled
    assert pairing._subscription_timer is None
    assert pairing._subscription_task is None
    timer.callback()
    assert pairing._subscription_task is None
    pairing.connection.transport = mock.Mock()
    pairing.connection.protocol = mock.Mock()
    pairing.connection.is_secure = True
    pairing.connection.put_json.side_effect = None
    await pairing.subscribe([(1, 9)])
    assert pairing._acknowledged_subscriptions == {(1, 9)}


async def test_close_cancels_and_awaits_inflight_worker(subscription_pairing):
    pairing = subscription_pairing
    started = asyncio.Event()
    response = asyncio.get_running_loop().create_future()

    async def put_json(*args):
        started.set()
        return await response

    pairing.connection.put_json.side_effect = put_json
    caller = asyncio.create_task(pairing.subscribe([(1, 9)]))
    await asyncio.wait_for(started.wait(), 1)
    worker = pairing._subscription_task
    await pairing.close()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert worker.cancelled()
    assert response.cancelled()
    assert pairing._subscription_task is None
    assert pairing._subscription_timer is None


async def test_shutdown_cannot_restart_recovery(subscription_pairing):
    pairing = subscription_pairing
    pairing.connection.put_json.side_effect = RequestOutcomeUnknownError("Response lost")
    await pairing.subscribe([(1, 9)])
    await pairing.shutdown()
    assert await pairing.subscribe([(1, 10)]) == {}
    await pairing.connection_made(True)
    assert pairing._subscription_task is None
    assert pairing._subscription_timer is None
    assert pairing.connection.put_json.await_count == 1


async def test_connected_callback_does_not_wait_for_subscription_io(subscription_pairing):
    pairing = subscription_pairing
    pairing.subscriptions.add((1, 9))
    started, release = asyncio.Event(), asyncio.Event()

    async def put_json(*args):
        started.set()
        await release.wait()
        return {}

    pairing.connection.put_json.side_effect = put_json
    await asyncio.wait_for(pairing.connection_made(True), 1)
    await asyncio.wait_for(started.wait(), 1)
    release.set()
    await finish_worker(pairing)


async def test_subscription_waits_for_authentication_with_tcp_transport_present(subscription_pairing):
    pairing = subscription_pairing
    pairing.connection.is_secure = False
    started, authenticated = asyncio.Event(), asyncio.Event()

    async def negotiate():
        started.set()
        await authenticated.wait()
        pairing.connection.is_secure = True
        await pairing.connection_made(True)

    with mock.patch.object(pairing.connection, "_connect_once", side_effect=negotiate):
        caller = asyncio.create_task(pairing.subscribe([(1, 9)]))
        await asyncio.wait_for(started.wait(), 1)
        assert pairing.connection.transport is not None
        assert not pairing.connection.is_connected
        pairing.connection.put_json.assert_not_awaited()
        authenticated.set()
        await asyncio.wait_for(caller, 1)
    pairing.connection.put_json.assert_awaited_once()


async def test_failed_connection_waits_for_existing_connector_to_recover(subscription_pairing):
    pairing = subscription_pairing
    pairing.connection.protocol = None
    pairing.connection.transport = None
    pairing.connection.is_secure = False
    with mock.patch.object(
        pairing.connection, "ensure_connection", side_effect=asyncio.TimeoutError
    ) as connect:
        assert await pairing.subscribe([(1, 9)]) == {}
    connect.assert_called_once()
    assert pairing._subscription_task is None
    assert pairing._subscription_timer is None
    assert pairing._subscription_retry_delay == 5
    pairing.connection.protocol = mock.Mock()
    pairing.connection.transport = mock.Mock()
    pairing.connection.is_secure = True
    await pairing.connection_made(True)
    await finish_worker(pairing)
    assert pairing._acknowledged_subscriptions == {(1, 9)}


async def test_unsubscribe_cancels_pending_recovery(subscription_pairing):
    pairing = subscription_pairing
    pairing.connection.put_json.side_effect = RequestOutcomeUnknownError("Response lost")
    await pairing.subscribe([(1, 9)])
    timer = pairing._subscription_timer
    pairing.connection.put_json.side_effect = None
    await pairing.unsubscribe([(1, 9)])
    assert pairing.subscriptions == set()
    assert timer.cancelled
    timer.callback()
    assert pairing._subscription_timer is None
    assert pairing._subscription_task is None
    assert pairing.connection.put_json.await_count == 2


async def test_success_and_failure_rows_preserve_unsubscribe_results(subscription_pairing):
    pairing = subscription_pairing
    await pairing.subscribe([(1, 9), (1, 10)])
    pairing.connection.put_json.return_value = {
        "characteristics": [
            {"aid": 1, "iid": 9, "status": 0},
            {"aid": 1, "iid": 10, "status": HapStatusCode.INSUFFICIENT_AUTH.value},
        ]
    }
    result = await pairing.unsubscribe([(1, 9), (1, 10)])
    assert result[(1, 9)]["status"] == 0
    assert result[(1, 10)]["status"] == HapStatusCode.INSUFFICIENT_AUTH.value
    assert pairing.subscriptions == {(1, 10)}
    assert pairing._acknowledged_subscriptions == {(1, 10)}


@pytest.mark.parametrize(
    "response",
    [
        [],
        {"status": "bad"},
        {"characteristics": None},
        {"characteristics": [True]},
        {"characteristics": [{"aid": 1, "iid": 9, "status": 0}]},
    ],
)
async def test_malformed_subscription_response_retains_intent(subscription_pairing, response):
    pairing = subscription_pairing
    pairing.connection.put_json.return_value = response
    assert await pairing.subscribe([(1, 9), (1, 10)]) == {}
    assert pairing.supports_subscribe
    assert pairing._acknowledged_subscriptions == set()
    assert pairing._subscription_retry_at - pairing._subscription_loop.time() == 5
    pairing.connection.put_json.return_value = {}
    await advance_recovery(pairing)
    assert pairing._acknowledged_subscriptions == {(1, 9), (1, 10)}


async def test_config_refresh_ignores_inflight_rejection(subscription_pairing):
    pairing = subscription_pairing
    started, release = asyncio.Event(), asyncio.Event()

    async def put_json(*args):
        if pairing.connection.put_json.await_count == 1:
            started.set()
            await release.wait()
            return {"status": HapStatusCode.NOTIFICATION_NOT_SUPPORTED.value}
        return {}

    pairing.connection.put_json.side_effect = put_json
    caller = asyncio.create_task(pairing.subscribe([(1, 9)]))
    await asyncio.wait_for(started.wait(), 1)
    with mock.patch.object(pairing, "list_accessories_and_characteristics", mock.AsyncMock()):
        await pairing._process_config_changed(2)
    release.set()
    assert await asyncio.wait_for(caller, 1) == {}
    if pairing._subscription_task:
        await finish_worker(pairing)
    assert pairing._unsupported_subscriptions == {}
    assert pairing._acknowledged_subscriptions == {(1, 9)}


async def test_http_rejection_does_not_infer_unsupported(subscription_pairing):
    pairing = subscription_pairing
    pairing.connection.put_json.side_effect = HttpErrorResponse("HTTP rejection", mock.Mock(code=400))
    assert await pairing.subscribe([(1, 9)]) == {}
    assert pairing.supports_subscribe


async def test_global_hap_error_is_returned_for_the_batch(subscription_pairing):
    pairing = subscription_pairing
    pairing.connection.put_json.return_value = {"status": HapStatusCode.NOTIFICATION_NOT_SUPPORTED.value}
    result = await pairing.subscribe([(1, 9), (1, 10)])
    assert result == {
        char: {
            "status": HapStatusCode.NOTIFICATION_NOT_SUPPORTED.value,
            "description": HapStatusCode.NOTIFICATION_NOT_SUPPORTED.description,
        }
        for char in [(1, 9), (1, 10)]
    }
    assert pairing.supports_subscribe
