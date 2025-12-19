import asyncio
import time
async def main():
    print("start")
    await asyncio.to_thread(time.sleep, 10)  # ✅
    print("done")

asyncio.run(main())