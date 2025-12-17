import json

from agent import Agent
from llm import LLM

with open("flow.json") as f:
    flow = json.load(f)

agent = Agent(flow=flow, llm=LLM())
result = agent.run()

print(result)
