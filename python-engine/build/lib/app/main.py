import asyncio
import logging

from app.websocket import EngineServer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)


async def main() -> None:
    server = EngineServer(host="127.0.0.1", port=8765)
    await server.run_forever()


if __name__ == "__main__":
    asyncio.run(main())
