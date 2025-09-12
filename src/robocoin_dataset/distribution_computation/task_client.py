import asyncio
import json
import logging
import socket
import traceback
from abc import abstractmethod

from websockets.exceptions import ConnectionClosed
from websockets.legacy.client import connect

from .constant import (
    CLIENT_ID,
    CLIENT_IP,
    ERR_MSG,
    ERROR,
    ERROR_MSG,
    MSG_CONTENT,
    MSG_TYPE,
    NO_TASK,
    PING,
    PONG,
    REGISTER,
    REGISTERED,
    REQUEST_TASK,
    TASK,
    TASK_CATEGORY,
    TASK_CONTENT,
    TASK_FAILED,
    TASK_ID,
    TASK_RESULT,
    TASK_RESULT_CONTENT,
    TASK_RESULT_STATUS,
    TASK_SUCCESS,
)


def get_client_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception as e:
        raise e


class TaskClient:
    def __init__(
        self,
        server_uri: str = "ws://localhost:8765",
        heartbeat_interval: float = 10.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.server_uri = server_uri
        self.heartbeat_interval = heartbeat_interval
        self.websocket = None
        self.connected = False
        self.client_id: str | None = None
        self.local_ip: str = get_client_ip()

        self._heartbeat_task: asyncio.Task | None = None
        self._receiver_task: asyncio.Task | None = None
        self._response_future: asyncio.Future | None = None  # May be None
        self.logger = logger

    async def connect_to_server(self, max_retries: int = 5, delay: float = 3.0) -> None:
        for attempt in range(max_retries):
            try:
                self.websocket = await connect(self.server_uri)
                self.connected = True
                if self.logger:
                    self.logger.info(f"✅ Successfully connected to the server: {self.server_uri}")
                return
            except Exception as e:  # noqa: PERF203
                if self.logger:
                    self.logger.warning(f"Connection failed ({attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                else:
                    raise ConnectionError(
                        f"❌ All retries failed, unable to connect to server {self.server_uri}"
                    )

    async def _start_heartbeat(self) -> None:
        async def send_ping() -> None:
            try:
                while True:
                    if self.connected and self.websocket and not self.websocket.closed:
                        try:
                            await self.websocket.send(json.dumps({MSG_TYPE: PING}))
                        except Exception as e:
                            if self.logger:
                                self.logger.warning(f"Heartbeat sending failed: {e}")
                            break
                    await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Heartbeat task exception: {e}")

        self._heartbeat_task = asyncio.create_task(send_ping())

    async def _stop_heartbeat(self) -> None:
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

    async def _message_receiver(self) -> None:
        try:
            async for message in self.websocket:
                try:
                    msg = json.loads(message)
                    msg_type = msg.get(MSG_TYPE)
                    if self.logger:
                        self.logger.debug(f"📩 Message received: {msg}")
                    msg_content = msg.get(MSG_CONTENT, {})

                    if msg_type == REGISTERED:
                        self.client_id = str(msg_content[CLIENT_ID])
                        if self.logger:
                            self.logger.info(
                                f"🏷️ Client registered, server assigned ID: {self.client_id} | IP: {self.local_ip}"
                            )
                        # Wake up the waiting registration future
                        if self._response_future is not None and not self._response_future.done():
                            self._response_future.set_result(msg)

                    elif msg_type == PING:
                        await self.websocket.send(json.dumps({MSG_TYPE: PONG}))
                        if self.logger:
                            self.logger.debug("🔁 Client reply PONG")

                    elif msg_type in (TASK, NO_TASK):
                        if self._response_future is not None and not self._response_future.done():
                            self._response_future.set_result(msg)
                        else:
                            if self.logger:
                                self.logger.warning(
                                    f"⚠️ Received task response but no pending request: {msg_type}"
                                )

                    # elif msg_type == ACK:
                    #     self._logger.info(f"✅ Server acknowledged: {msg}")

                    elif msg_type == ERROR:
                        if self.logger:
                            self.logger.error(f"❌ Server error: {msg.get(ERROR_MSG)}")

                    else:
                        if self.logger:
                            self.logger.debug(f"📩 Unknown message type: {msg_type}")

                except Exception as e:
                    if self.logger:
                        self.logger.error(f"Failed to process message: {e}")

        except ConnectionClosed as e:
            if self.logger:
                self.logger.warning(f"⚠️ The connection to the server has been closed: {e}")
            self.connected = False
            if self._response_future is not None and not self._response_future.done():
                self._response_future.set_exception(e)

        except Exception as e:
            if self.logger:
                self.logger.error(f"Message receiving exception: {e}")
            self.connected = False
            if self._response_future is not None and not self._response_future.done():
                self._response_future.set_exception(e)

    async def register(self) -> bool:
        if not self.connected:
            await self.connect_to_server()

        if self.client_id:
            if self.logger:
                self.logger.info("ℹ️ Client already registered, skipping")
            return True

        # Use Future to wait for registration response
        old_future = self._response_future
        if old_future is not None and not old_future.done():
            old_future.cancel()  # Cancel the old one

        self._response_future = asyncio.Future()
        try:
            register_msg = {
                MSG_TYPE: REGISTER,
                MSG_CONTENT: {
                    CLIENT_IP: get_client_ip(),
                    TASK_CATEGORY: self.get_task_category(),
                },
            }
            await self.websocket.send(json.dumps(register_msg))
            if self.logger:
                self.logger.info(f"📤 Registration request sent | IP: {self.local_ip}")

            try:
                result = await asyncio.wait_for(self._response_future, timeout=10.0)
                return result is not None
            except asyncio.TimeoutError:
                if self.logger:
                    self.logger.error(
                        "❌ Registration timeout, no response received from the server"
                    )
                if self._response_future and not self._response_future.done():
                    self._response_future.cancel()
                return False
        except Exception as e:
            if self.logger:
                self.logger.error(f"Registration failed: {e}")
            self.connected = False
            return False
        # Do not set to None here, handled by cleanup uniformly

    @abstractmethod
    def get_task_category(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generate_task_request_desc(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def _sync_process_task(self, task: dict) -> dict:
        raise NotImplementedError

    async def request_task(self) -> dict | None:
        if not self.client_id:
            if self.logger:
                self.logger.warning("❌ Registration failed")
            return None

        # ✅ Check and cancel old future (if exists)
        if self._response_future is not None and not self._response_future.done():
            if self.logger:
                self.logger.warning("⚠️ There are unfinished task requests, canceling old request")
            self._response_future.set_exception(
                RuntimeError("Old request overridden by new request")
            )

        self._response_future = asyncio.Future()

        try:
            task_request_desc = self.generate_task_request_desc()
            task_request_dict = {
                MSG_TYPE: REQUEST_TASK,
                MSG_CONTENT: {
                    TASK_CATEGORY: self.get_task_category(),
                    TASK_CONTENT: task_request_desc,
                },
            }
            await self.websocket.send(json.dumps(task_request_dict))
            if self.logger:
                self.logger.info("📤 Task request sent")

            try:
                task_msg = await asyncio.wait_for(self._response_future, timeout=15.0)
            except asyncio.TimeoutError:
                if self.logger:
                    self.logger.warning("⚠️ Request task timeout (no response within 15 seconds)")
                if self._response_future is not None and not self._response_future.done():
                    self._response_future.cancel()
                return None

            # Parse response
            msg_type = task_msg.get(MSG_TYPE)
            if msg_type == TASK:
                if self.logger:
                    self.logger.info(f"📥 Task received: {task_msg}")
                return task_msg[MSG_CONTENT]

            if msg_type == NO_TASK:
                if self.logger:
                    self.logger.info("📭 No task available from server")
                return None

            if self.logger:
                self.logger.warning(f"⚠️ Unexpected response type: {msg_type}")
            return None

        except Exception as e:
            if self.logger:
                self.logger.error(f"Request task failed: {e}")
            return None

        # ✅ Do not set to None here

    async def submit_result(self, result: dict) -> None:
        if not self.client_id:
            if self.logger:
                self.logger.error("❌ Not registered, cannot submit result")
            return

        try:
            await self.websocket.send(json.dumps(result))
            if self.logger:
                self.logger.info(f"📤 Submitting task result, task_id is: {result.get(TASK_ID)}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Submit result failed: {e}")

    async def process_task(self, task_data: dict) -> dict:
        loop = asyncio.get_event_loop()
        task_id = task_data.get(TASK_ID)
        try:
            task_result_content = await loop.run_in_executor(
                None, self._sync_process_task, task_data
            )
            return {TASK_RESULT_STATUS: TASK_SUCCESS, TASK_RESULT_CONTENT: task_result_content}
        except Exception:
            if self.logger:
                self.logger.error(f"Task {task_id} failed. {traceback.format_exc()}")
            return {
                TASK_RESULT_STATUS: TASK_FAILED,
                ERR_MSG: f"Task {task_id} failed. {traceback.format_exc()}",
                TASK_RESULT_CONTENT: {},
            }

    async def run_until_no_task(self) -> None:
        try:
            if not self.connected:
                await self.connect_to_server()

                # Start receiver and heartbeat
                self._receiver_task = asyncio.create_task(self._message_receiver())
                await self.register()
                if not self.client_id:
                    if self.logger:
                        self.logger.error("❌ Registration failed, exiting")
                    return

                await self._start_heartbeat()
                if self.logger:
                    self.logger.info(f"✅ Client {self.client_id} is ready, starting task loop")

            while True:
                task = await self.request_task()
                if task is None:
                    if self.logger:
                        self.logger.info("📭 No task from server, client exiting")
                    break

                if self.logger:
                    self.logger.info(f"🚀 Starting to process task: {task.get(TASK_ID)}")
                result_content = await self.process_task(task)
                result = {
                    MSG_TYPE: TASK_RESULT,
                    MSG_CONTENT: result_content,
                }
                result[TASK_ID] = task.get(TASK_ID)
                result[CLIENT_ID] = self.client_id
                await self.submit_result(result)
                if self.logger:
                    self.logger.info("📤 Task result submitted, preparing to request next task...")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Client runtime exception: {e}")
        finally:
            await self._cleanup()

    async def _cleanup(self) -> None:
        """Unified resource cleanup"""
        self.connected = False

        # Clean up response_future
        if self._response_future is not None and not self._response_future.done():
            self._response_future.set_exception(asyncio.CancelledError())
        self._response_future = None  # ✅ Only set to None here

        # Cancel receiver task
        if self._receiver_task:
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass
            self._receiver_task = None

        # Stop heartbeat
        await self._stop_heartbeat()

        # Close websocket
        if self.websocket:
            try:
                await self.websocket.close()
                if self.logger:
                    self.logger.info("🔌 Client connection closed")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to close connection: {e}")

    async def run(self) -> None:
        try:
            await self.run_until_no_task()
        except KeyboardInterrupt:
            if self.logger:
                self.logger.info("🛑 Client interrupted by user")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Client terminated abnormally: {e}")
        finally:
            await self._cleanup()
