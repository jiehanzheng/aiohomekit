import asyncio
from collections.abc import Generator
from datetime import timedelta
from typing import Any
from unittest import mock

import pytest

from aiohomekit.controller.ip.connection import HomeKitConnection, _normalize_host
from aiohomekit.controller.ip.pairing import IpPairing
from aiohomekit.exceptions import AccessoryDisconnectedError, IncorrectPairingIdError
from aiohomekit.model import Transport
from aiohomekit.model.categories import Categories
from aiohomekit.protocol import get_session_keys
from aiohomekit.protocol.statuscodes import HapStatusCode
from aiohomekit.zeroconf import HomeKitService
from tests.accessoryserver import AccessoryRequestHandler
from tests.subscription_clock import ControlledLoop


def wrong_accessory_get_session_keys(
    pairing_data: dict[str, str | int | list[Any]],
) -> Generator[Any, Any, None]:
    """Return a pair-verify state machine that reports the wrong accessory.

    The first exchange is delegated to the real state machine so a valid
    start request goes out on the wire, then the response is answered
    with the error a mismatched accessory would produce.
    """
    real_state_machine = get_session_keys(pairing_data)
    first_request = real_state_machine.send(None)

    def state_machine() -> Generator[Any, Any, None]:
        yield first_request
        raise IncorrectPairingIdError("step 3")

    return state_machine()


async def test_list_accessories(pairing: IpPairing):
    accessories = await pairing.list_accessories_and_characteristics()
    assert accessories[0]["aid"] == 1
    assert accessories[0]["services"][0]["iid"] == 1

    char = accessories[0]["services"][0]["characteristics"][0]

    assert char["description"] == "Identify"
    assert char["iid"] == 2
    assert char["format"] == "bool"
    assert char["perms"] == ["pw"]
    assert char["type"] == "00000014-0000-1000-8000-0026BB765291"


async def test_get_characteristics(pairing: IpPairing):
    characteristics = await pairing.get_characteristics([(1, 9)])

    assert characteristics[(1, 9)] == {"value": False}


async def test_duplicate_get_characteristics(pairing):
    characteristics = await pairing.get_characteristics([(1, 9), (1, 9)])
    assert characteristics[(1, 9)] == {"value": False}


async def test_get_characteristics_after_failure(pairing: IpPairing):
    characteristics = await pairing.get_characteristics([(1, 9)])

    assert characteristics[(1, 9)] == {"value": False}

    pairing.connection.transport.close()
    await asyncio.sleep(0)
    assert not pairing.connection.is_connected
    assert not pairing.is_available

    characteristics = await pairing.get_characteristics([(1, 9)])

    assert characteristics[(1, 9)] == {"value": False}


async def test_reconnect_soon_after_disconnected(pairing: IpPairing):
    characteristics = await pairing.get_characteristics([(1, 9)])

    assert characteristics[(1, 9)] == {"value": False}

    assert pairing.connection.is_connected
    assert pairing.is_available

    pairing.connection.transport.close()
    await asyncio.sleep(0)
    assert not pairing.connection.is_connected
    assert not pairing.is_available

    # Ensure we can safely call multiple times
    pairing._async_description_update(None)
    pairing._async_description_update(None)
    pairing._async_description_update(None)

    await asyncio.sleep(0)  # ensure the callback has a chance to run and create _connector
    await asyncio.wait_for(pairing.connection._connector, timeout=0.5)
    assert pairing.connection.is_connected

    characteristics = await pairing.get_characteristics([(1, 9)])

    assert characteristics[(1, 9)] == {"value": False}


async def test_reconnect_soon_after_device_is_offline_for_a_bit(pairing: IpPairing):
    characteristics = await pairing.get_characteristics([(1, 9)])

    assert characteristics[(1, 9)] == {"value": False}

    assert pairing.connection.is_connected
    assert pairing.is_available

    with mock.patch(
        "aiohomekit.controller.ip.connection.HomeKitConnection._connect_once",
        side_effect=asyncio.TimeoutError,
    ):
        pairing.connection.transport.close()
        await asyncio.sleep(0)
        assert not pairing.connection.is_connected
        assert not pairing.is_available

        for _ in range(3):
            pairing._async_description_update(None)
            # ensure the callback has a chance to run and create _connector
            await asyncio.sleep(0)
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(pairing.connection._connector), timeout=0.2)
            assert not pairing.connection.is_connected

    pairing._async_description_update(None)
    await asyncio.wait_for(pairing.connection._connector, timeout=0.5)
    assert pairing.connection.is_connected
    assert pairing.is_available

    characteristics = await pairing.get_characteristics([(1, 9)])

    assert characteristics[(1, 9)] == {"value": False}


async def test_reconnect_soon_on_device_reboot(pairing: IpPairing):
    characteristics = await pairing.get_characteristics([(1, 9)])

    assert characteristics[(1, 9)] == {"value": False}

    assert pairing.connection.is_connected
    assert pairing.is_available

    with mock.patch(
        "aiohomekit.controller.ip.connection.HomeKitConnection._connect_once",
        side_effect=asyncio.TimeoutError,
    ):
        pairing.connection.protocol.connection_lost(OSError("Connection reset by peer"))

    await asyncio.sleep(0)
    assert not pairing.connection.is_connected
    assert not pairing.is_available
    await asyncio.wait_for(pairing.connection._connector, timeout=0.5)
    assert pairing.connection.is_connected
    assert pairing.is_available

    characteristics = await pairing.get_characteristics([(1, 9)])

    assert characteristics[(1, 9)] == {"value": False}


async def test_get_connect_hosts_filters_failed_hosts() -> None:
    connection = HomeKitConnection(None, ["192.168.2.13", "192.168.2.10"], 5001)

    assert connection._get_connect_hosts() == ["192.168.2.13", "192.168.2.10"]

    connection._pair_verify_failed_hosts.add("192.168.2.13")
    assert connection._get_connect_hosts() == ["192.168.2.10"]

    # Once every host has failed pair-verify the exclusions are
    # reset so the accessory can never become permanently unreachable
    connection._pair_verify_failed_hosts.add("192.168.2.10")
    assert connection._get_connect_hosts() == ["192.168.2.13", "192.168.2.10"]
    assert not connection._pair_verify_failed_hosts


async def test_get_connect_hosts_with_hostname() -> None:
    """A host that is not an IP address is compared as given."""
    connection = HomeKitConnection(None, ["device.local", "192.168.2.10"], 5001)

    connection._pair_verify_failed_hosts.add(_normalize_host("device.local"))
    assert connection._get_connect_hosts() == ["192.168.2.10"]


async def test_get_connect_hosts_normalizes_addresses() -> None:
    connection = HomeKitConnection(None, ["2001:db8::1", "192.168.2.10"], 5001)

    # The peer address of the failed connection may format the same
    # IPv6 address differently than the advertised address
    connection._pair_verify_failed_hosts.add(_normalize_host("2001:0db8:0000:0000:0000:0000:0000:0001%eth0"))
    assert connection._get_connect_hosts() == ["192.168.2.10"]


async def test_pair_verify_wrong_accessory_marks_host_failed(pairing: IpPairing) -> None:
    connection = pairing.connection

    with mock.patch(
        "aiohomekit.controller.ip.connection.get_session_keys",
        side_effect=wrong_accessory_get_session_keys,
    ):
        with pytest.raises(IncorrectPairingIdError):
            await connection._connect_once()

    assert connection._pair_verify_failed_hosts == {"127.0.0.1"}
    assert connection.transport is None
    assert connection.protocol is None
    assert not connection.is_connected


async def test_pair_verify_wrong_accessory_recovers(pairing: IpPairing) -> None:
    connection = pairing.connection
    calls = []

    def flaky_get_session_keys(pairing_data):
        calls.append(pairing_data)
        if len(calls) == 1:
            return wrong_accessory_get_session_keys(pairing_data)
        return get_session_keys(pairing_data)

    with (
        mock.patch(
            "aiohomekit.controller.ip.connection.get_session_keys",
            side_effect=flaky_get_session_keys,
        ),
        # Skip the reconnect backoff so the test does not spend wall
        # clock time sleeping; a single host means no immediate retry
        mock.patch("asyncio.sleep", mock.AsyncMock()),
    ):
        await asyncio.wait_for(connection.ensure_connection(), timeout=5)

    assert len(calls) == 2
    assert connection.is_connected
    assert connection.is_secure
    assert connection.last_connector_error is None
    assert not connection._pair_verify_failed_hosts

    characteristics = await pairing.get_characteristics([(1, 9)])
    assert characteristics[(1, 9)] == {"value": False}


async def test_incorrect_pairing_id_retries_next_address_without_backoff() -> None:
    connection = HomeKitConnection(None, ["192.168.2.13", "192.168.2.10"], 5001)
    attempts = []

    async def fake_connect_once():
        attempts.append(connection._get_connect_hosts())
        if len(attempts) == 1:
            connection._pair_verify_failed_hosts.add("192.168.2.13")
            raise IncorrectPairingIdError("step 3")

    with mock.patch.object(connection, "_connect_once", fake_connect_once):
        # Without the immediate retry the backoff sleep would exceed the timeout
        await asyncio.wait_for(connection._reconnect(), timeout=0.5)

    assert attempts == [
        ["192.168.2.13", "192.168.2.10"],
        ["192.168.2.10"],
    ]


async def test_host_change_clears_failed_hosts(pairing: IpPairing) -> None:
    connection = pairing.connection
    port = pairing.pairing_data["AccessoryPort"]

    pairing.description = HomeKitService(
        name="unittestLight",
        id="12:34:56:00:01:0a",
        model="Demoserver",
        feature_flags=0,
        status_flags=0,
        config_num=1,
        state_num=0,
        category=Categories.LIGHTBULB,
        protocol_version="1.0",
        type="_hap._tcp.local",
        address="127.0.0.1",
        addresses=["127.0.0.1"],
        port=port,
    )
    connection.hosts = ["192.0.2.1"]
    connection._pair_verify_failed_hosts = {"192.0.2.1"}

    await connection._connect_once()

    assert connection.hosts == ["127.0.0.1"]
    assert not connection._pair_verify_failed_hosts
    assert connection.is_connected
    assert connection.is_secure


async def test_put_characteristics(pairing: IpPairing):
    characteristics = await pairing.put_characteristics([(1, 9, True)])

    assert characteristics == {}

    characteristics = await pairing.get_characteristics([(1, 9)])

    assert characteristics[(1, 9)] == {"value": True}


async def test_put_characteristics_cancelled(pairing: IpPairing):
    characteristics = await pairing.put_characteristics([(1, 9, True)])
    characteristics = await pairing.get_characteristics([(1, 9)])

    with mock.patch.object(pairing.connection.transport, "writelines"):
        task = asyncio.create_task(pairing.put_characteristics([(1, 9, False)]))
        await asyncio.sleep(0)
        for future in pairing.connection.protocol.result_cbs:
            future.cancel()
        await asyncio.sleep(0)
        with pytest.raises(asyncio.CancelledError):
            await task

    # We should wait a few seconds to see if the
    # connection can be re-established and the write can be
    # completed. But this is not currently possible because
    # we do not wait for the connection to be re-established
    # before we try to write the data. When we implement
    # reconnection we should remove this pytest.raises
    # and the sleep below.
    with pytest.raises(AccessoryDisconnectedError):
        await pairing.get_characteristics([(1, 9)])

    await asyncio.sleep(0)
    characteristics = await pairing.get_characteristics([(1, 9)])
    assert characteristics[(1, 9)] == {"value": True}


async def test_put_characteristics_callbacks(pairing: IpPairing):
    events = []

    def process_new_events(new_values_dict: dict[tuple[int, int], dict[str, Any]]) -> None:
        events.append(new_values_dict)

    pairing.dispatcher_connect(process_new_events)
    assert events == []

    characteristics = await pairing.put_characteristics([(1, 9, True)])
    assert events == [{}, {(1, 9): {"value": True}}]
    assert characteristics == {}

    # Identify is a write only characteristic, so we should not get a callback
    characteristics = await pairing.put_characteristics([(1, 2, True)])
    assert events == [{}, {(1, 9): {"value": True}}]

    characteristics = await pairing.get_characteristics([(1, 9)])

    assert characteristics[(1, 9)] == {"value": True}

    characteristics = await pairing.get_characteristics({(1, 9)})

    assert characteristics[(1, 9)] == {"value": True}


async def test_subscribe(pairing: IpPairing):
    assert pairing.subscriptions == set()

    await pairing.subscribe([(1, 9)])

    assert pairing.subscriptions == {(1, 9)}

    characteristics = await pairing.get_characteristics([(1, 9)])

    assert characteristics == {(1, 9): {"value": False}}


@pytest.mark.parametrize("concurrent", [False, True])
async def test_reconnect_restores_events_after_lost_subscription(pairings, concurrent):
    left, right = pairings
    await right.get_characteristics([(1, 9)])
    clock = right._subscription_loop = ControlledLoop()
    loop = asyncio.get_running_loop()
    connected, event = loop.create_future(), loop.create_future()

    def handler(data):
        if not data and not connected.done():
            connected.set_result(None)
        if (1, 9) in data and not event.done():
            event.set_result(data)

    right.dispatcher_connect(handler)
    do_put = AccessoryRequestHandler.do_PUT
    attempts = 0

    def disconnect_once(request):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            # Discard an actual encrypted request without replying. Reconnection
            # and event delivery must then succeed through the normal TCP path.
            request.close_connection = True
            return
        do_put(request)

    with mock.patch.object(AccessoryRequestHandler, "do_PUT", disconnect_once):
        if concurrent:
            await asyncio.gather(right.subscribe([(1, 9)]), right.subscribe([(1, 9)]))
        else:
            await right.subscribe([(1, 9)])
        await asyncio.wait_for(connected, 5)
        assert attempts == 1
        assert right._subscription_retry_at - clock.time() == 5
        clock.advance()
        await asyncio.wait_for(asyncio.shield(right._subscription_task), 5)

    assert attempts == 2
    assert right.supports_subscribe
    with mock.patch.object(right, "get_characteristics", side_effect=AssertionError("Unexpected polling")):
        await left.put_characteristics([(1, 9, True)])
        assert await asyncio.wait_for(event, 5) == {(1, 9): {"value": True}}


async def test_repeated_subscription_disconnects_keep_polling(pairing: IpPairing):
    await pairing.get_characteristics([(1, 9)])
    clock = pairing._subscription_loop = ControlledLoop()
    connections = asyncio.Queue()

    def handler(data):
        if not data:
            connections.put_nowait(pairing.connection.protocol)

    pairing.dispatcher_connect(handler)
    attempts = 0

    def disconnect(request):
        nonlocal attempts
        attempts += 1
        request.close_connection = True

    with mock.patch.object(AccessoryRequestHandler, "do_PUT", disconnect):
        await pairing.subscribe([(1, 9)])
        await asyncio.wait_for(connections.get(), 5)
        for delay in [5, 10]:
            assert pairing._subscription_retry_at - clock.time() == delay
            assert await pairing.get_characteristics([(1, 9)]) == {(1, 9): {"value": False}}
            clock.advance()
            await asyncio.wait_for(asyncio.shield(pairing._subscription_task), 5)
            await asyncio.wait_for(connections.get(), 5)
        assert await pairing.get_characteristics([(1, 9)]) == {(1, 9): {"value": False}}

    assert attempts == 3
    assert pairing.supports_subscribe
    assert pairing.subscriptions == {(1, 9)}
    assert pairing._subscription_retry_at - clock.time() == 20


async def test_unsubscribe(pairing: IpPairing):
    await pairing.subscribe([(1, 9)])

    assert pairing.subscriptions == {(1, 9)}

    characteristics = await pairing.get_characteristics([(1, 9)])

    assert characteristics == {(1, 9): {"value": False}}

    await pairing.unsubscribe([(1, 9)])

    assert pairing.subscriptions == set()

    characteristics = await pairing.get_characteristics([(1, 9)])

    assert characteristics == {(1, 9): {"value": False}}


async def test_dispatcher_connect(pairing: IpPairing):
    assert pairing.listeners == set()

    def callback(x):
        pass

    cancel = pairing.dispatcher_connect(callback)
    assert pairing.listeners == {callback}

    cancel()
    assert pairing.listeners == set()


async def test_receiving_events(pairings):
    """
    Test that can receive events when change happens in another session.

    We set up 2 controllers both with active secure sessions. One
    subscribes and then other does put() calls.

    This test is currently skipped because accessory server doesnt
    support events.
    """
    left: IpPairing = pairings[0]
    right: IpPairing = pairings[1]

    event_value = None
    ev = asyncio.Event()

    def handler(data):
        print(data)
        nonlocal event_value
        event_value = data
        ev.set()

    # Set where to send events
    right.dispatcher_connect(handler)

    # Set what events to get
    await right.subscribe([(1, 9)])

    # Trigger an event by writing a change on the other connection
    await left.put_characteristics([(1, 9, True)])

    # Wait for event to be received for up to 5s
    await asyncio.wait_for(ev.wait(), 5)

    assert event_value == {(1, 9): {"value": True}}


async def test_subscribe_invalid_iid(pairing: IpPairing):
    """
    Test that can get an error when subscribing to an invalid iid.
    """
    result = await pairing.subscribe([(1, 999999)])
    assert result == {
        (1, 999999): {
            "description": "Resource does not exist.",
            "status": HapStatusCode.RESOURCE_NOT_EXIST.value,
        }
    }


async def test_list_pairings(pairing: IpPairing):
    pairings = await pairing.list_pairings()
    assert pairings == [
        {
            "controllerType": "admin",
            "pairingId": "decc6fa3-de3e-41c9-adba-ef7409821bfc",
            "permissions": 1,
            "publicKey": "d708df2fbf4a8779669f0ccd43f4962d6d49e4274f88b1292f822edc3bcf8ed8",
        }
    ]


async def test_add_pairings(pairing: IpPairing):
    await pairing.add_pairing(
        "decc6fa3-de3e-41c9-adba-ef7409821bfe",
        "d708df2fbf4a8779669f0ccd43f4962d6d49e4274f88b1292f822edc3bcf8ed7",
        "User",
    )

    pairings = await pairing.list_pairings()
    assert pairings == [
        {
            "controllerType": "admin",
            "pairingId": "decc6fa3-de3e-41c9-adba-ef7409821bfc",
            "permissions": 1,
            "publicKey": "d708df2fbf4a8779669f0ccd43f4962d6d49e4274f88b1292f822edc3bcf8ed8",
        },
        {
            "controllerType": "regular",
            "pairingId": "decc6fa3-de3e-41c9-adba-ef7409821bfe",
            "permissions": 0,
            "publicKey": "d708df2fbf4a8779669f0ccd43f4962d6d49e4274f88b1292f822edc3bcf8ed7",
        },
    ]


async def test_add_and_remove_pairings(pairing: IpPairing):
    await pairing.add_pairing(
        "decc6fa3-de3e-41c9-adba-ef7409821bfe",
        "d708df2fbf4a8779669f0ccd43f4962d6d49e4274f88b1292f822edc3bcf8ed7",
        "User",
    )

    pairings = await pairing.list_pairings()
    assert pairings == [
        {
            "controllerType": "admin",
            "pairingId": "decc6fa3-de3e-41c9-adba-ef7409821bfc",
            "permissions": 1,
            "publicKey": "d708df2fbf4a8779669f0ccd43f4962d6d49e4274f88b1292f822edc3bcf8ed8",
        },
        {
            "controllerType": "regular",
            "pairingId": "decc6fa3-de3e-41c9-adba-ef7409821bfe",
            "permissions": 0,
            "publicKey": "d708df2fbf4a8779669f0ccd43f4962d6d49e4274f88b1292f822edc3bcf8ed7",
        },
    ]

    await pairing.remove_pairing("decc6fa3-de3e-41c9-adba-ef7409821bfe")

    pairings = await pairing.list_pairings()
    assert pairings == [
        {
            "controllerType": "admin",
            "pairingId": "decc6fa3-de3e-41c9-adba-ef7409821bfc",
            "permissions": 1,
            "publicKey": "d708df2fbf4a8779669f0ccd43f4962d6d49e4274f88b1292f822edc3bcf8ed8",
        }
    ]


async def test_identify(pairing):
    identified = await pairing.identify()
    assert identified is True


async def test_transport_property(pairing: IpPairing):
    assert pairing.transport == Transport.IP


async def test_polling_property(pairing: IpPairing):
    assert pairing.poll_interval == timedelta(seconds=60)


async def test_put_characteristics_invalid_value(pairing: IpPairing):
    aid, iid = (1, 2)
    characteristics = [(aid, iid, 100)]
    status_code = await pairing.put_characteristics(characteristics)
    assert status_code is not None
    assert status_code[(aid, iid)] is not None
    assert status_code[(aid, iid)]["status"] == HapStatusCode.INVALID_VALUE.value


async def test_put_characteristics_boolean_response(pairing: IpPairing):
    """Test handling of malformed response with boolean values (issue #465)."""
    # Mock the connection's put_json to return the malformed response
    # Use the existing accessory ID from the test fixture
    malformed_response = {
        "characteristics": [
            True,  # Boolean instead of dict
            {"iid": 9, "aid": 1, "status": -70402},
            True,  # Another boolean
        ]
    }

    with mock.patch.object(pairing.connection, "put_json", return_value=malformed_response):
        # This should not raise TypeError anymore
        result = await pairing.put_characteristics([(1, 9, True)])

        # Should only process the valid error response
        assert (1, 9) in result
        assert result[(1, 9)]["status"] == -70402
        assert "Unable to communicate" in result[(1, 9)]["description"]
