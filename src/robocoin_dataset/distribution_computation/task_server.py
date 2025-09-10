import asyncio
import json
import logging
from abc import abstractmethod

from websockets.exceptions import ConnectionClosed
from websockets.legacy.server import WebSocketServerProtocol, serve

from robocoin_dataset.utils.logger import setup_logger

from .constant import (
    CLIENT_ID,
    CLIENT_IP,
    ERROR,
    ERROR_MSG,
    LAST_PING,
    LAST_PONG,
    MSG_CONTENT,
    MSG_TYPE,
    PING,
    PONG,
    REGISTER,
    REGISTERED,
    REQUEST_TASK,
    TASK,
    TASK_CATEGORY,
    TASK_ID,
    TASK_RESULT,
)


class TaskServer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        heartbeat_interval: float = 30.0,
        timeout: float = 15.0,
        logger: logging.Logger | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.heartbeat_interval = heartbeat_interval
        self.timeout = timeout
        self.clients: set[WebSocketServerProtocol] = set()
        self.client_info: dict[WebSocketServerProtocol, dict[str, any]] = {}

        if logger is not None:
            self.logger = logger
        else:
            self.logger = setup_logger(
                name=f"task_server_{self.get_task_category()}", log_path="./log"
            )
        self._client_num = 0
        self._current_task_idx = 0
        self._lock = asyncio.Lock()
        self._task_content_dict: dict[str, dict] = {}

    @abstractmethod
    def get_task_category(self) -> str:
        raise NotImplementedError

    @property
    def current_task_id(self) -> str:
        return self.get_task_category() + "_" + str(self._current_task_idx)

    def get_task_content(self, task_id: str) -> dict:
        return self._task_content_dict.get(task_id, {})

    async def register_client(self, websocket: WebSocketServerProtocol, msg_content: dict) -> None:
        client_ip = msg_content.get(CLIENT_IP)

        # Prevent duplicate registration
        if websocket in self.client_info:
            self.logger.warning("Client attempted duplicate registration")
            return

        async with self._lock:
            client_id = f"{self.get_task_category()}_{self._client_num}"
            self._client_num += 1

        self.clients.add(websocket)
        self.client_info[websocket] = {
            CLIENT_ID: client_id,
            CLIENT_IP: client_ip,
            LAST_PING: None,
            LAST_PONG: None,
            TASK_CATEGORY: msg_content.get(TASK_CATEGORY),
        }

        msg = json.dumps(
            {
                MSG_TYPE: REGISTERED,
                MSG_CONTENT: {
                    CLIENT_ID: client_id,
                },
            }
        )
        await websocket.send(msg)

        self.logger.info(
            f"Client registered successfully | ID: {client_id} | IP: {client_ip} | Current connections: {len(self.clients)}"
        )

    async def unregister_client(self, websocket: WebSocketServerProtocol) -> None:
        if websocket in self.clients:
            self.clients.discard(websocket)
            info = self.client_info.pop(websocket, None)
            if info:
                self.logger.info(f"Client disconnected | ID: {info[CLIENT_ID]}")
        return

    @abstractmethod
    def generate_task_content(self) -> dict | None:
        raise NotImplementedError

    async def send_task_to_client(self, websocket: WebSocketServerProtocol) -> None:
        info = self.client_info.get(websocket)
        if not info:
            return

        task_content = await asyncio.to_thread(self.generate_task_content)
        if not task_content:
            # Send NO_TASK
            await websocket.send(json.dumps({MSG_TYPE: "NO_TASK"}))
            self.logger.info(f"No task available for client {info[CLIENT_ID]}")
            return

        async with self._lock:
            task_id = self.current_task_id
            self._current_task_idx += 1

        task_content[TASK_ID] = task_id

        msg = json.dumps(
            {
                MSG_TYPE: TASK,
                MSG_CONTENT: task_content,
                TASK_ID: task_id,
                CLIENT_ID: info[CLIENT_ID],
            }
        )

        self._task_content_dict[task_id] = task_content
        await websocket.send(msg)
        self.logger.info(f"Assigned task {TASK_ID} to client {info[CLIENT_ID]}")

    @abstractmethod
    def handle_task_result(self, task_content: dict, task_result_content: dict) -> None:
        raise NotImplementedError

    async def handle_message(self, websocket: WebSocketServerProtocol, message: str) -> None:
        try:
            msg = json.loads(message)
            msg_type = msg.get(MSG_TYPE)
            msg_content = msg.get(MSG_CONTENT)

            if msg_type == REGISTER:
                await self.register_client(websocket, msg_content)

            elif msg_type == REQUEST_TASK:
                await self.send_task_to_client(websocket)

            elif msg_type == TASK_RESULT:
                client_id = msg.get(CLIENT_ID)
                task_id = msg.get(TASK_ID)

                self.logger.info(
                    f"Received task result | Client ID: {client_id} | Task ID: {task_id}"
                )
                try:
                    task_result_content = msg.get(MSG_CONTENT)
                    task_content = self.get_task_content(task_id)

                    await asyncio.to_thread(
                        self.handle_task_result,
                        task_content=task_content,
                        task_result_content=task_result_content,
                    )
                except Exception as e:
                    self.logger.error(f"Error handling task result: {e}")

            elif msg_type == PONG:
                ctx = self.client_info.get(websocket)
                if ctx:
                    ctx[LAST_PONG] = asyncio.get_event_loop().time()
                    self.logger.debug(f"Received PONG from client {ctx[CLIENT_ID]}")

            elif msg_type == PING:
                # Client sent PING, server replies with PONG
                await websocket.send(json.dumps({MSG_TYPE: PONG}))
                self.logger.debug("Server replied with PONG")

            else:
                await websocket.send(
                    json.dumps(
                        {
                            MSG_TYPE: ERROR,
                            ERROR_MSG: f"Unknown message type: {msg_type}",
                        }
                    )
                )

        except json.JSONDecodeError:
            error_msg = {
                MSG_TYPE: ERROR,
                MSG_CONTENT: {
                    ERROR_MSG: "Invalid JSON format",
                },
            }
            await websocket.send(json.dumps(error_msg))
        except Exception as e:
            self.logger.error(f"Failed to process message: {e}")
            error_msg = {
                MSG_TYPE: ERROR,
                MSG_CONTENT: {
                    ERROR_MSG: str(e),  # Ensure it's a string
                },
            }
            try:
                await websocket.send(json.dumps(error_msg))
            except Exception:
                pass  # Ignore errors when sending error response

    async def _send_heartbeat(self, websocket: WebSocketServerProtocol) -> None:
        try:
            while True:
                await asyncio.sleep(self.heartbeat_interval)
                if websocket.closed:
                    break
                try:
                    await websocket.send(json.dumps({MSG_TYPE: PING}))
                    loop = asyncio.get_event_loop()
                    self.client_info[websocket][LAST_PING] = loop.time()
                    self.logger.debug("Server sent PING to client")
                except Exception as e:
                    self.logger.warning(f"Failed to send PING: {e}")
                    break
        except asyncio.CancelledError:
            pass

    async def _check_heartbeat_alive(self, websocket: WebSocketServerProtocol) -> None:
        try:
            while True:
                await asyncio.sleep(self.heartbeat_interval)
                if websocket.closed:
                    break
                ctx = self.client_info.get(websocket)
                if not ctx:
                    continue

                last_ping = ctx.get(LAST_PING)
                last_pong = ctx.get(LAST_PONG)

                # If a ping was sent but no pong received, and timeout has passed
                if last_ping is not None and (last_pong is None or last_pong < last_ping):
                    time_since_ping = asyncio.get_event_loop().time() - last_ping
                    if time_since_ping > self.timeout:
                        self.logger.warning(f"Client {ctx[CLIENT_ID]} heartbeat timeout")
                        await self._handle_timeout(websocket)
                        break
        except asyncio.CancelledError:
            pass

    async def _handle_timeout(self, websocket: WebSocketServerProtocol) -> None:
        self.logger.error("Client lost connection, disconnecting")
        await websocket.close(code=4000, reason="Heartbeat timeout")
        await self.unregister_client(websocket)

    async def _message_receiver(self, websocket: WebSocketServerProtocol) -> None:
        try:
            async for message in websocket:
                await self.handle_message(websocket, message)
        except ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)

    async def handler(self, websocket: WebSocketServerProtocol) -> None:
        try:
            hb_send_task = asyncio.create_task(self._send_heartbeat(websocket))
            hb_check_task = asyncio.create_task(self._check_heartbeat_alive(websocket))
            receiver_task = asyncio.create_task(self._message_receiver(websocket))

            done, pending = await asyncio.wait(
                [hb_send_task, hb_check_task, receiver_task],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            for task in pending:
                try:
                    await task
                except asyncio.CancelledError:  # noqa: PERF203
                    pass  # Expected when tasks are cancelled

        except Exception as e:
            self.logger.error(f"Exception in connection handler: {e}")
        finally:
            await self.unregister_client(websocket)

    async def start(self) -> None:
        async with serve(self.handler, self.host, self.port):
            self.logger.info(f"Task server started successfully: ws://{self.host}:{self.port}")
            await asyncio.Future()  # Run forever

    def get_client_count(self, category: str = None) -> int:
        if not category:
            return len(self.clients)
        return len(
            [c for c in self.client_info.values() if c.get(TASK_CATEGORY) == category]
        )  # Use TASK_CATEGORY constant
