"""OpenPI-compatible framing using StarVLA's existing NumPy/msgpack codec."""
import asyncio
import http
import logging
import time
import traceback

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from deployment.model_server.tools import msgpack_numpy


class YAMWebsocketServer:
    def __init__(self, policy, host="127.0.0.1", port=8002):
        self.policy, self.host, self.port = policy, host, port

    async def handler(self, socket):
        await socket.send(msgpack_numpy.packb(self.policy.metadata))
        try:
            async for message in socket:
                started = time.perf_counter()
                # Synchronous inference serializes model use across connections.
                result = self.policy.infer(msgpack_numpy.unpackb(message))
                result["server_timing"] = {"infer_ms": (time.perf_counter() - started) * 1000}
                await socket.send(msgpack_numpy.packb(result))
        except ConnectionClosed:
            pass
        except Exception:
            logging.exception("YAM policy request failed")
            # Existing OpenPI clients recognize text frames as errors.
            try:
                await socket.send(traceback.format_exc())
                await socket.close(code=1011, reason="Policy request failed")
            except ConnectionClosed:
                pass

    @staticmethod
    def health_check(connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    async def run(self):
        async with serve(self.handler, self.host, self.port, compression=None,
                         max_size=None, process_request=self.health_check):
            print(f"READY ws://{self.host}:{self.port} task={self.policy.metadata['task']} "
                  f"horizon={self.policy.metadata['horizon']}", flush=True)
            await asyncio.Future()

    def serve_forever(self):
        asyncio.run(self.run())
