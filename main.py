from __future__ import annotations
from agents import Agent,Runner, set_tracing_disabled,function_tool
from agents.extensions.models.litellm_model import LitellmModel
import os
import asyncio
# Disable tracing for the agent
set_tracing_disabled(disabled=True)
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")
MODEL =  os.getenv("MODEL") 
if not MODEL:
    raise ValueError("MODEL environment variable is not set.")

@function_tool
def get_weather(city: str):
    print(f"[debug] getting weather for {city}")
    return f"The weather in {city} is sunny."


async def main():
    agent = Agent(
        name="Assistant",
        instructions="You only respond in haikus.",
        model=LitellmModel(model=model, api_key=GOOGLE_API_KEY),
        tools=[get_weather],
    )

    result = await Runner.run(agent, "What's the weather in Tokyo?")
    print(result.final_output)


if __name__ == "__main__":
    # First try to get model/api key from args
    asyncio.run(main())
