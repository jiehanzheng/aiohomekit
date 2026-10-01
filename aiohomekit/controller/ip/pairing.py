#
# Copyright 2019 aiohomekit team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from contextlib import suppress
from datetime import timedelta
from itertools import groupby
from operator import itemgetter
from typing import Any

from aiohomekit import hkjson
from aiohomekit.controller.abstract import AbstractController, AbstractPairingData
from aiohomekit.exceptions import (
    AccessoryDisconnectedError,
    AuthenticationError,
    HttpErrorResponse,
    HttpException,
    InvalidError,
    RequestNotSentError,
    RequestOutcomeUnknownError,
    UnknownError,
    UnpairedError,
)
from aiohomekit.http import HttpContentTypes
from aiohomekit.model import Accessories, AccessoriesState, Transport
from aiohomekit.model.characteristics import (
    CharacteristicPermissions,
    CharacteristicsTypes,
)
from aiohomekit.protocol import error_handler
from aiohomekit.protocol.statuscodes import HapStatusCode, to_status_code
from aiohomekit.protocol.tlv import TLV
from aiohomekit.utils import async_create_task, asyncio_timeout
from aiohomekit.uuid import normalize_uuid
from aiohomekit.zeroconf import HomeKitService, ZeroconfPairing

from .connection import SecureHomeKitConnection

logger = logging.getLogger(__name__)


EMPTY_EVENT = {}
SUBSCRIBE_RETRY_INITIAL = 5
SUBSCRIBE_RETRY_MAX = 60 * 60
RETRYABLE_SUBSCRIBE_STATUSES = {
    HapStatusCode.UNABLE_TO_COMMUNICATE,
    HapStatusCode.RESOURCE_BUSY,
    HapStatusCode.OUT_OF_RESOURCES,
    HapStatusCode.TIMED_OUT,
    HapStatusCode.NOT_ALLOWED_IN_CURRENT_STATE,
}


def format_characteristic_list(
    data: dict[str, Any], requested_characteristics: set[tuple[int, int]] | None = None
) -> dict[tuple[int, int], dict[str, Any]]:
    tmp: dict[tuple[int, int], dict[str, Any]] = {}

    # Handle global error status first - set defaults for all requested characteristics
    if "status" in data and data["status"] != 0:
        # Device returned a global error status
        status_code = to_status_code(data["status"])
        # For unknown codes, include the actual code in the description
        if status_code == HapStatusCode.UNKNOWN:
            description = f"Unknown error code: {data['status']}"
        else:
            description = status_code.description

        logger.debug(
            "Device returned error status %s (%s) for characteristics request",
            data["status"],
            description,
        )
        # If we know what was requested, mark them all as failed initially
        if requested_characteristics:
            for aid, iid in requested_characteristics:
                tmp[(aid, iid)] = {"status": data["status"], "description": description}

    # Process any characteristics that are present - these override the defaults
    for c in data.get("characteristics", []):
        # Skip malformed characteristics (e.g., boolean values) or missing aid/iid
        if not isinstance(c, dict) or "aid" not in c or "iid" not in c:
            logger.debug("Skipping malformed characteristic: %s", c)
            continue

        key = (c["aid"], c["iid"])
        del c["aid"]
        del c["iid"]

        if "status" in c and c["status"] == 0:
            del c["status"]
        if "status" in c and c["status"] != 0:
            status_code = to_status_code(c["status"])
            if status_code == HapStatusCode.UNKNOWN:
                c["description"] = f"Unknown error code: {c['status']}"
            else:
                c["description"] = status_code.description
        tmp[key] = c

    return tmp


class IpPairing(ZeroconfPairing):
    """
    This represents a paired HomeKit IP accessory.
    """

    def __init__(self, controller: AbstractController, pairing_data: AbstractPairingData) -> None:
        """
        Initialize a Pairing by using the data either loaded from file or obtained after calling
        Controller.perform_pairing().

        :param pairing_data:
        """
        self.pairing_data = pairing_data
        self.connection = SecureHomeKitConnection(self, self.pairing_data)
        self.supports_subscribe = True
        self._subscription_loop = asyncio.get_running_loop()
        self._subscription_lock = asyncio.Lock()
        self._subscription_task: asyncio.Task[dict] | None = None
        self._subscription_timer: asyncio.TimerHandle | None = None
        self._subscription_closed = False
        self._subscription_session = None
        self._subscription_generation = 0
        self._acknowledged_subscriptions: set[tuple[int, int]] = set()
        self._unsupported_subscriptions: dict[tuple[int, int], dict] = {}
        self._rejected_subscriptions: dict[tuple[int, int], dict] = {}
        self._subscription_retry_delay = SUBSCRIBE_RETRY_INITIAL
        self._subscription_retry_at = 0.0

        super().__init__(controller, pairing_data)

    @property
    def is_connected(self) -> bool:
        return self.connection.is_connected

    @property
    def is_available(self) -> bool:
        """Returns true if the device is currently available."""
        return self.connection.is_connected

    @property
    def transport(self) -> Transport:
        """The transport used for the connection."""
        return Transport.IP

    @property
    def poll_interval(self) -> timedelta:
        """Returns how often the device should be polled."""
        return timedelta(minutes=1)

    @property
    def name(self) -> str:
        """Return the name of the pairing with the address."""
        connection = self.connection
        host = connection.connected_host or connection.hosts
        if self.description:
            return f"{self.description.name} [{host}:{connection.port}] (id={self.id})"
        return f"[{host}:{connection.port}] (id={self.id})"

    def event_received(self, event):
        self._callback_listeners(format_characteristic_list(event))

    async def connection_made(self, secure):
        if not secure or self._shutdown or self._subscription_closed:
            return

        self._set_subscription_session(self.connection.protocol)
        # Let our listeners know the connection is available again
        self._callback_listeners(EMPTY_EVENT)

        # Restoration must not run inside the connector: a failed subscription
        # can disconnect again while that connector is still marked as running.
        self._schedule_subscriptions()

    async def _ensure_connected(self):
        """Ensure we are connected to the device."""
        connection = self.connection
        if self._shutdown:
            raise RequestNotSentError("Pairing has been shut down")
        self._subscription_closed = False
        if connection.is_connected:
            return

        try:
            async with asyncio_timeout(10):
                await connection.ensure_connection()
        except asyncio.TimeoutError:
            last_connector_error = connection.last_connector_error
            if not last_connector_error or isinstance(last_connector_error, asyncio.TimeoutError):
                raise AccessoryDisconnectedError(
                    f"Timeout while waiting for connection to device {connection.hosts}:{connection.port}"
                )
            # The exception name is included since otherwise the error message
            # is not very helpful as it could be something like `step 3`
            raise AccessoryDisconnectedError(
                f"Error while connecting to device {connection.hosts}:{connection.port}: "
                f"{last_connector_error} ({type(last_connector_error).__name__})"
            )

        if not connection.is_connected:
            raise AccessoryDisconnectedError(
                f"Ensure connection returned but still not connected: {connection.hosts}:{connection.port}"
            )

        self._callback_availability_changed(True)

    async def close(self) -> None:
        """
        Close the pairing's communications. This closes the session.
        """
        self._subscription_closed = True
        self._cancel_subscription_timer()
        task, self._subscription_task = self._subscription_task, None
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        self._subscription_session = None
        self._subscription_generation += 1
        self._acknowledged_subscriptions.clear()
        self._rejected_subscriptions.clear()
        self._subscription_retry_at = 0.0
        self._subscription_retry_delay = SUBSCRIBE_RETRY_INITIAL
        await self.connection.close()
        await asyncio.sleep(0)

    async def list_accessories_and_characteristics(self) -> list[dict[str, Any]]:
        """
        This retrieves a current set of accessories and characteristics behind this pairing.

        :return: the accessory data as described in the spec on page 73 and following
        :raises AccessoryNotFoundError: if the device can not be found via zeroconf
        """
        await self._ensure_connected()

        response = await self.connection.get_json("/accessories")

        accessories = response["accessories"]

        for accessory in accessories:
            for service in accessory["services"]:
                service["type"] = normalize_uuid(service["type"])

                for characteristic in service["characteristics"]:
                    characteristic["type"] = normalize_uuid(characteristic["type"])

        self._accessories_state = AccessoriesState(Accessories.from_list(accessories), self.config_num or 0)
        self._update_accessories_state_cache()
        return accessories

    async def list_pairings(self):
        """
        This method returns all pairings of a HomeKit accessory. This always includes the local controller and can only
        be done by an admin controller.

        The keys in the resulting dicts are:
         * pairingId: the pairing id of the controller
         * publicKey: the ED25519 long-term public key of the controller
         * permissions: bit value for the permissions
         * controllerType: either admin or regular

        :return: a list of dicts
        :raises: UnknownError: if it receives unexpected data
        :raises: UnpairedError: if the polled accessory is not paired
        """
        await self._ensure_connected()

        data = await self.connection.post_tlv(
            "/pairings",
            [(TLV.kTLVType_State, TLV.M1), (TLV.kTLVType_Method, TLV.ListPairings)],
        )

        if not (data[0][0] == TLV.kTLVType_State and data[0][1] == TLV.M2):
            raise UnknownError("unexpected data received: " + str(data))

        if data[1][0] == TLV.kTLVType_Error and data[1][1] == TLV.kTLVError_Authentication:
            raise UnpairedError("Must be paired")

        tmp = []
        r = {}
        for d in data[1:]:
            if d[0] == TLV.kTLVType_Identifier:
                r = {}
                tmp.append(r)
                r["pairingId"] = d[1].decode()
            if d[0] == TLV.kTLVType_PublicKey:
                r["publicKey"] = d[1].hex()
            if d[0] == TLV.kTLVType_Permissions:
                controller_type = "regular"
                if d[1] == b"\x01":
                    controller_type = "admin"
                r["permissions"] = int.from_bytes(d[1], byteorder="little")
                r["controllerType"] = controller_type
        return tmp

    async def get_characteristics(
        self,
        characteristics: Iterable[tuple[int, int]],
    ) -> dict[tuple[int, int], dict[str, Any]]:
        """
        This method is used to get the current readouts of any characteristic of the accessory.

        :param characteristics: a list of 2-tupels of accessory id and instance id
        :param include_meta: if True, include meta information about the characteristics. This contains the format and
                             the various constraints like maxLen and so on.
        :param include_perms: if True, include the permissions for the requested characteristics.
        :param include_type: if True, include the type of the characteristics in the result. See CharacteristicsTypes
                             for translations.
        :param include_events: if True on a characteristics that supports events, the result will contain information if
                               the controller currently is receiving events for that characteristic. Key is 'ev'.
        :return: a dict mapping 2-tupels of aid and iid to dicts with value or status and description, e.g.
                 {(1, 8): {'value': 23.42}
                  (1, 37): {'description': 'Resource does not exist.', 'status': -70409}
                 }
        """
        await self._ensure_connected()

        if not self.accessories:
            await self.list_accessories_and_characteristics()

        if isinstance(characteristics, set):
            characteristics_set = characteristics
        else:
            characteristics_set = set(characteristics)

        url = "/characteristics?id=" + ",".join(f"{aid}.{iid}" for aid, iid in characteristics_set)

        response = await self.connection.get_json(url)

        return format_characteristic_list(response, characteristics_set)

    async def put_characteristics(
        self, characteristics: Iterable[tuple[int, int, Any]]
    ) -> dict[tuple[int, int], dict[str, Any]]:
        """
        Update the values of writable characteristics. The characteristics have to be identified by accessory id (aid),
        instance id (iid). If do_conversion is False (the default), the value must be of proper format for the
        characteristic since no conversion is done. If do_conversion is True, the value is converted.

        :param characteristics: a list of 3-tupels of accessory id, instance id and the value
        :param do_conversion: select if conversion is done (False is default)
        :return: a dict from (aid, iid) onto {status, description}
        :raises FormatError: if the input value could not be converted to the target type and conversion was
                             requested
        """
        await self._ensure_connected()

        if not self.accessories:
            await self.list_accessories_and_characteristics()

        char_payload: list[dict[str, Any]] = []
        listener_update: dict[tuple[int, int], dict[str, Any]] = {}
        for characteristic in characteristics:
            aid, iid, value = characteristic
            char_payload.append({"aid": aid, "iid": iid, "value": value})
            accessory_chars = self.accessories.aid(aid).characteristics
            char = accessory_chars.iid(iid)
            if CharacteristicPermissions.paired_read in char.perms:
                listener_update[(aid, iid)] = {"value": value}

        response = await self.connection.put_json("/characteristics", {"characteristics": char_payload})
        response_status: dict[tuple[int, int], dict[str, Any]] = {}
        if response:
            # If there is a response it means something failed so
            # we need to remove the listener update for the failed
            # characteristics.
            for characteristic in response["characteristics"]:
                # Skip malformed characteristics (e.g., boolean values)
                if (
                    not isinstance(characteristic, dict)
                    or "aid" not in characteristic
                    or "iid" not in characteristic
                ):
                    logger.debug("Skipping malformed characteristic in response: %s", characteristic)
                    continue
                aid, iid = characteristic["aid"], characteristic["iid"]
                key = (aid, iid)
                status = characteristic["status"]
                status_code = to_status_code(status).description
                if status_code != HapStatusCode.SUCCESS:
                    listener_update.pop(key, None)
                response_status[key] = {"status": status, "description": status_code}

        if listener_update:
            self._callback_listeners(listener_update)

        return response_status

    async def thread_provision(
        self,
        dataset: str,
    ) -> None:
        """Provision a device with Thread network credentials."""

    async def subscribe(self, characteristics):
        """Retain subscription intent and attempt it when recovery permits.

        An empty result contains no characteristic errors; it does not prove
        subscription acceptance or subsequent event delivery.
        """
        characteristics = set(characteristics)
        await super().subscribe(characteristics)

        if not self.supports_subscribe:
            logger.info(
                "This device does not support push, so only polling operations will be supported during this session"
            )
            return None

        if self._shutdown:
            return {}
        self._subscription_closed = False
        for char in characteristics:
            self._rejected_subscriptions.pop(char, None)

        task = self._schedule_subscriptions()
        result = await asyncio.shield(task) if task else {}
        statuses = self._unsupported_subscriptions | self._rejected_subscriptions | result
        return {char: status for char, status in statuses.items() if char in characteristics and status}

    def _set_subscription_session(self, protocol) -> None:
        if protocol is not self._subscription_session:
            self._subscription_session = protocol
            self._subscription_generation += 1
            self._acknowledged_subscriptions.clear()
            self._rejected_subscriptions.clear()

    def _pending_subscriptions(self) -> set[tuple[int, int]]:
        return (
            self.subscriptions
            - self._acknowledged_subscriptions
            - self._unsupported_subscriptions.keys()
            - self._rejected_subscriptions.keys()
        )

    def _cancel_subscription_timer(self) -> None:
        if self._subscription_timer:
            self._subscription_timer.cancel()
            self._subscription_timer = None

    def _schedule_subscriptions(self) -> asyncio.Task[dict] | None:
        if (
            self._shutdown
            or self._subscription_closed
            or not self.supports_subscribe
            or not self._pending_subscriptions()
        ):
            self._cancel_subscription_timer()
            return None
        if self._subscription_loop.time() < self._subscription_retry_at:
            if not self._subscription_timer:
                self._subscription_timer = self._subscription_loop.call_at(
                    self._subscription_retry_at, self._subscription_retry_ready
                )
            return None
        self._cancel_subscription_timer()
        if not self._subscription_task or self._subscription_task.done():
            self._subscription_task = async_create_task(
                self._restore_subscriptions(), name=f"HomeKit subscriptions {self.id}"
            )
            self._subscription_task.add_done_callback(self._subscription_finished)
        return self._subscription_task

    def _subscription_retry_ready(self) -> None:
        self._subscription_timer = None
        self._schedule_subscriptions()

    def _subscription_finished(self, task: asyncio.Task[dict]) -> None:
        if self._subscription_task is not task:
            return
        self._subscription_task = None
        if task.cancelled() or task.exception():
            return
        # If connection establishment failed, the existing connector will wake
        # us on authentication. Do not create another worker in a tight loop.
        if self.connection.is_connected or self._subscription_loop.time() < self._subscription_retry_at:
            self._schedule_subscriptions()

    def _defer_subscriptions(self) -> None:
        if self._shutdown or self._subscription_closed:
            return
        delay = self._subscription_retry_delay
        self._subscription_retry_at = self._subscription_loop.time() + delay
        self._subscription_retry_delay = min(delay * 2, SUBSCRIBE_RETRY_MAX)
        logger.debug("%s: Subscription recovery deferred for %s seconds", self.name, delay)
        self._schedule_subscriptions()

    async def _restore_subscriptions(self) -> dict:
        statuses = {}
        if self._shutdown or self._subscription_closed:
            return statuses
        # No subscription lock may be held while waiting for authentication.
        try:
            await self._ensure_connected()
        except AccessoryDisconnectedError:
            logger.debug("%s: Could not connect to restore subscriptions", self.name)
            return statuses

        async with self._subscription_lock:
            if self._shutdown or self._subscription_closed or not self.connection.is_connected:
                return statuses
            self._set_subscription_session(self.connection.protocol)
            while pending := self._pending_subscriptions():
                if self._subscription_loop.time() < self._subscription_retry_at:
                    return statuses
                retry = False
                for _, group in groupby(sorted(pending), key=itemgetter(0)):
                    batch = set(group) & self._pending_subscriptions()
                    if not batch:
                        continue
                    session = self.connection.protocol
                    generation = self._subscription_generation
                    try:
                        result = await self._update_subscriptions(batch, True)
                    except HttpErrorResponse as ex:
                        if (
                            session is not self.connection.protocol
                            or generation != self._subscription_generation
                        ):
                            return statuses
                        self._rejected_subscriptions.update({char: {} for char in batch})
                        logger.warning("%s: Subscription rejected with HTTP %s", self.name, ex.response.code)
                        continue
                    except RequestNotSentError as ex:
                        if session is self.connection.protocol:
                            self.connection._connection_lost(ex)
                        return statuses
                    except AccessoryDisconnectedError:
                        # A lost response cannot establish unsupported capability.
                        # Retain intent; the timer bounds attempts, not their count.
                        self._defer_subscriptions()
                        return statuses

                    if session is not self.connection.protocol or generation != self._subscription_generation:
                        return statuses
                    statuses.update(result)
                    for char in batch:
                        status = result.get(char, {})
                        code = to_status_code(status.get("status", 0))
                        if code == HapStatusCode.SUCCESS:
                            self._acknowledged_subscriptions.add(char)
                        elif code == HapStatusCode.NOTIFICATION_NOT_SUPPORTED:
                            self._unsupported_subscriptions[char] = status
                        elif code in RETRYABLE_SUBSCRIBE_STATUSES:
                            retry = True
                        else:
                            self._rejected_subscriptions[char] = status
                if retry:
                    self._defer_subscriptions()
                    return statuses

            # Partial batch success must not continually reset the backoff for
            # a bridge whose other subscriptions still fail on every attempt.
            if self.subscriptions & self._acknowledged_subscriptions:
                self._subscription_retry_at = 0.0
                self._subscription_retry_delay = SUBSCRIBE_RETRY_INITIAL
            return statuses

    async def unsubscribe(self, characteristics):
        char_set = set(characteristics)
        if self.connection.is_connected:
            await self._ensure_connected()
        async with self._subscription_lock:
            status = await self._update_subscriptions(char_set, False) if self.connection.is_connected else {}
            removed = {
                char
                for char in char_set
                if to_status_code(status.get(char, {}).get("status", 0)) == HapStatusCode.SUCCESS
            }
            await super().unsubscribe(removed)
            self._acknowledged_subscriptions.difference_update(removed)
            for char in removed:
                self._rejected_subscriptions.pop(char, None)
        self._schedule_subscriptions()
        return status

    async def _update_subscriptions(self, characteristics, ev):
        """Subscribe or unsubscribe to characteristics."""
        status = {}
        # We do one aid at a time to match what iOS does
        # even though its inefficient
        # https://github.com/home-assistant/core/issues/37996
        #
        # Prebuild the payloads to avoid the set size changing
        # between await calls
        char_payloads = [
            [{"aid": aid, "iid": iid, "ev": ev} for aid, iid in aid_iids]
            for _, aid_iids in groupby(sorted(characteristics), key=itemgetter(0))
        ]
        for char_payload in char_payloads:
            response = await self.connection.put_json(
                "/characteristics",
                {"characteristics": char_payload},
            )
            if not isinstance(response, dict):
                raise RequestOutcomeUnknownError("Malformed subscription response")
            if response:
                batch = {(row["aid"], row["iid"]) for row in char_payload}
                if "status" in response and (
                    not isinstance(response["status"], int) or isinstance(response["status"], bool)
                ):
                    raise RequestOutcomeUnknownError("Malformed subscription status")
                if "status" in response and response["status"] != 0:
                    status.update(
                        {
                            char: {
                                "status": response["status"],
                                "description": to_status_code(response["status"]).description,
                            }
                            for char in batch
                        }
                    )
                # An empty body is a success response
                rows = response.get("characteristics", [])
                if not isinstance(rows, list):
                    raise RequestOutcomeUnknownError("Malformed subscription characteristics")
                for row in rows:
                    if isinstance(row, dict) and all(
                        isinstance(row.get(key), int) and not isinstance(row[key], bool)
                        for key in ("aid", "iid", "status")
                    ):
                        status[(row["aid"], row["iid"])] = {
                            "status": row["status"],
                            "description": to_status_code(row["status"]).description,
                        }
                    else:
                        raise RequestOutcomeUnknownError("Malformed subscription characteristic status")
                if "status" not in response and not batch <= status.keys():
                    raise RequestOutcomeUnknownError("Incomplete subscription response")

        return status

    async def async_populate_accessories_state(
        self, force_update: bool = False, attempts: int | None = None
    ) -> bool:
        """Populate the state of all accessories.

        This method should try not to fetch all the accessories unless
        we know the config num is out of date or force_update is True
        """
        if not self.accessories or force_update:
            await self.list_accessories_and_characteristics()

    async def _process_config_changed(self, config_num: int) -> None:
        """Process a config change.

        This method is called when the config num changes.
        """
        await self.list_accessories_and_characteristics()
        self._accessories_state = AccessoriesState(self._accessories_state.accessories, config_num)
        # An in-flight reply from the previous configuration must not restore
        # a rejection that this refresh just invalidated.
        self._subscription_generation += 1
        self._unsupported_subscriptions.clear()
        self._rejected_subscriptions.clear()
        self._acknowledged_subscriptions.clear()
        self._callback_and_save_config_changed(self.config_num)
        self._schedule_subscriptions()

    def _process_disconnected_events(self):
        """Process any events that happened while we were disconnected.

        We don't disconnect in IP so there is no need to do anything here.
        """

    async def identify(self):
        """
        This call can be used to trigger the identification of a paired accessory. A successful call should
        cause the accessory to perform some specific action by which it can be distinguished from the others (blink a
        LED for example).

        It uses the identify characteristic as described on page 152 of the spec.

        :return True, if the identification was run, False otherwise
        """
        await self._ensure_connected()

        if not self.accessories:
            await self.list_accessories_and_characteristics()

        # we are looking for a characteristic of the identify type
        identify_type = CharacteristicsTypes.IDENTIFY

        # search all accessories, all services and all characteristics
        logger.debug("Searching for identify characteristic in %s", self.accessories)
        for accessory in self.accessories:
            aid = accessory.aid
            for service in accessory.services:
                for characteristic in service.characteristics:
                    iid = characteristic.iid
                    c_type = normalize_uuid(characteristic.type)
                    if identify_type == c_type:
                        # found the identify characteristic, so let's put a value there
                        if not await self.put_characteristics([(aid, iid, True)]):
                            return True
        return False

    async def add_pairing(self, additional_controller_pairing_identifier, ios_device_ltpk, permissions):
        await self._ensure_connected()

        if permissions == "User":
            permissions = TLV.kTLVType_Permission_RegularUser
        elif permissions == "Admin":
            permissions = TLV.kTLVType_Permission_AdminUser
        else:
            raise RuntimeError(f"Unknown permission: {permissions}")

        request_tlv = [
            (TLV.kTLVType_State, TLV.M1),
            (TLV.kTLVType_Method, TLV.AddPairing),
            (
                TLV.kTLVType_Identifier,
                additional_controller_pairing_identifier.encode(),
            ),
            (TLV.kTLVType_PublicKey, bytes.fromhex(ios_device_ltpk)),
            (TLV.kTLVType_Permissions, permissions),
        ]

        data = dict(await self.connection.post_tlv("/pairings", request_tlv))

        if data.get(TLV.kTLVType_State, TLV.M2) != TLV.M2:
            raise InvalidError("Unexpected state after add pairing request")

        if TLV.kTLVType_Error in data:
            error_handler(data[TLV.kTLVType_Error], "M2")

        return True

    async def remove_pairing(self, pairingId: str) -> bool:
        """
        Remove a pairing between the controller and the accessory. The pairing data is delete on both ends, on the
        accessory and the controller.

        Important: no automatic saving of the pairing data is performed. If you don't do this, the accessory seems still
            to be paired on the next start of the application.

        :param alias: the controller's alias for the accessory
        :param pairingId: the pairing id to be removed
        :raises AuthenticationError: if the controller isn't authenticated to the accessory.
        :raises AccessoryNotFoundError: if the device can not be found via zeroconf
        :raises UnknownError: on unknown errors
        """
        await self._ensure_connected()

        request_tlv = [
            (TLV.kTLVType_State, TLV.M1),
            (TLV.kTLVType_Method, TLV.RemovePairing),
            (TLV.kTLVType_Identifier, pairingId.encode("utf-8")),
        ]

        data = dict(await self.connection.post_tlv("/pairings", request_tlv))

        if data.get(TLV.kTLVType_State, TLV.M2) != TLV.M2:
            raise InvalidError("Unexpected state after removing pairing request")

        if TLV.kTLVType_Error in data:
            if data[TLV.kTLVType_Error] == TLV.kTLVError_Authentication:
                raise AuthenticationError("Remove pairing failed: insufficient access")
            raise UnknownError("Remove pairing failed: unknown error")

        await self._shutdown_if_primary_pairing_removed(pairingId)
        return True

    async def image(self, accessory: int, width: int, height: int) -> bytes:
        await self._ensure_connected()

        try:
            resp = await self.connection.post(
                "/resource",
                content_type=HttpContentTypes.JSON,
                body=hkjson.dump_bytes(
                    {
                        "aid": accessory,
                        "resource-type": "image",
                        "image-width": width,
                        "image-height": height,
                    }
                ),
            )

        except HttpException:
            return None

        except HttpErrorResponse:
            return None

        except AccessoryDisconnectedError:
            return None

        return resp.body

    def _async_description_update(self, description: HomeKitService | None) -> None:
        """We have new zeroconf metadata for this device."""
        super()._async_description_update(description)

        # If we are not connected, or are in the process of reconnecting, hasten the process
        if not self._shutdown:
            self.connection.reconnect_soon()
