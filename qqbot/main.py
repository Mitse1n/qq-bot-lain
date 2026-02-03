import asyncio

from qqbot.bot import ChatBot
from qqbot.config_loader import settings


async def main():
    bot = ChatBot(settings=settings)
    try:
        await bot.run()
    finally:
        await bot.close()


if __name__ == "__main__":
    print("Lain Bot is staring...")
    asyncio.run(main())

