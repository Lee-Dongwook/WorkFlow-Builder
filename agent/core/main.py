import json

from agent import Agent
from llm import LLM
from tools import UppercaseTool

with open("flow.json") as f:
    flow = json.load(f)

tools = {
    "uppercase": UppercaseTool(),
}

agent = Agent(flow=flow, llm=LLM(), tools=tools)
result = agent.run()

print(result)
