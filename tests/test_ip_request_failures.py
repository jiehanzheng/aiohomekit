"""Dispatch boundaries determine what a failed request can establish."""

import asyncio
from unittest import mock

import pytest

from aiohomekit.controller.ip.connection import HomeKitConnection, InsecureHomeKitProtocol
from aiohomekit.exceptions import HttpErrorResponse, RequestNotSentError, RequestOutcomeUnknownError


@pytest.fixture
async def connection():
    connection = HomeKitConnection(None, ["127.0.0.1"], 9)
    connection.host_header = "Host: 127.0.0.1"
    connection.transport = mock.Mock()
    connection.transport.is_closing.return_value = False
    connection.protocol = InsecureHomeKitProtocol(connection)
    connection.protocol.connection_made(connection.transport)
    return connection


async def test_missing_protocol_is_definitely_not_sent(connection):
    connection.protocol = None
    with pytest.raises(RequestNotSentError):
        await connection.request("PUT", "/characteristics")
    connection.transport.writelines.assert_not_called()


async def test_protocol_lost_while_acquiring_semaphore_is_not_sent(connection):
    semaphore = mock.MagicMock()

    async def acquire():
        connection.protocol = None

    semaphore.__aenter__.side_effect = acquire
    connection._concurrency_limit = semaphore
    with pytest.raises(RequestNotSentError):
        await connection.request("PUT", "/characteristics")
    connection.transport.writelines.assert_not_called()


async def test_closing_transport_is_definitely_not_sent(connection):
    connection.transport.is_closing.return_value = True
    with pytest.raises(RequestNotSentError):
        await connection.protocol.send_bytes(b"request")
    connection.transport.writelines.assert_not_called()


async def test_lost_response_has_unknown_outcome_and_keeps_cause(connection):
    protocol = connection.protocol
    error = OSError("Connection reset")
    connection.transport.writelines.side_effect = lambda payload: protocol._cancel_pending_requests(error)
    with pytest.raises(RequestOutcomeUnknownError) as caught:
        await protocol.send_bytes(b"request")
    assert caught.value.__cause__ is error
    connection.transport.close.assert_called_once()


async def test_timeout_has_unknown_outcome_and_keeps_cause(connection):
    protocol = connection.protocol
    connection.transport.writelines.side_effect = lambda payload: protocol._handle_timeout(
        protocol.result_cbs[0]
    )
    with pytest.raises(RequestOutcomeUnknownError) as caught:
        await protocol.send_bytes(b"request")
    assert isinstance(caught.value.__cause__, asyncio.TimeoutError)
    connection.transport.close.assert_called_once()


async def test_failed_write_has_unknown_outcome(connection):
    error = OSError("Write failed")
    connection.transport.writelines.side_effect = error
    with pytest.raises(RequestOutcomeUnknownError) as caught:
        await connection.protocol.send_bytes(b"request")
    assert caught.value.__cause__ is error
    assert connection.protocol.result_cbs[0].cancelled()


async def test_cancellation_remains_cancellation(connection):
    protocol = connection.protocol
    connection.transport.writelines.side_effect = lambda payload: protocol.result_cbs[0].cancel()
    with pytest.raises(asyncio.CancelledError):
        await protocol.send_bytes(b"request")
    connection.transport.close.assert_called_once()


@pytest.mark.parametrize("body", [b"\xff", b"not-json"])
async def test_malformed_response_has_unknown_outcome(connection, body):
    connection.put = mock.AsyncMock(return_value=mock.Mock(code=200, body=body))
    with pytest.raises(RequestOutcomeUnknownError) as caught:
        await connection.put_json("/characteristics", {})
    assert caught.value.__cause__ is not None
    connection.transport.close.assert_called_once()


@pytest.mark.parametrize("code", [400, 500])
async def test_http_rejection_is_a_received_response(connection, code):
    response = mock.Mock(code=code)
    connection.protocol.send_bytes = mock.AsyncMock(return_value=response)
    with pytest.raises(HttpErrorResponse) as caught:
        await connection.request("PUT", "/characteristics")
    assert caught.value.response is response
    connection.transport.close.assert_not_called()
