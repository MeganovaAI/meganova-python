import asyncio
import os
from dotenv import load_dotenv

load_dotenv()


async def main():
    from meganova import AsyncMegaNova

    api_key = os.getenv("MEGANOVA_API_KEY")
    if not api_key:
        print("Error: MEGANOVA_API_KEY not found.")
        return

    async with AsyncMegaNova(api_key=api_key) as client:
        # Non-streaming
        response = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Say hello in three languages."}],
            model="meganova-ai/manta-flash-1.0",
        )
        print(response.choices[0].message.content)

        # Streaming
        print("\n--- Streaming ---")
        stream = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Count to 5."}],
            model="meganova-ai/manta-flash-1.0",
            stream=True,
        )
        async for chunk in stream:
            content = chunk.choices[0].delta.get("content", "")
            print(content, end="", flush=True)
        print()


asyncio.run(main())
