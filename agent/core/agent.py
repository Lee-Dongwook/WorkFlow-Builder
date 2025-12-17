import re

class Agent:
    def __init__(self, flow: dict, llm, tools: dict):
        self.flow = flow
        self.llm = llm
        self.tools = tools
        self.context = {}
    
    def run(self):
        self.context.update(self.flow.get("inputs", {}))

        result = None

        for step in self.flow["steps"]:
            result = self.execute_step(step)
            self.context["previous_output"] = result

        return result

    def execute_step(self, step: dict):
        if "if" in step:
            return self.execute_if(step)

        if step["type"] == "llm":
            prompt = self.render(step["prompt"])
            return self.llm.run(prompt)

        if step["type"] == "tool":
            tool = self.tools[step["tool_name"]]
            tool_input = self.render(step["input"])
            return tool.run(tool_input)

    def execute_if(self, step):
        condition = self.render(step["if"])

        if eval(condition):
            return self.run_steps(step["then"])
        else:
            return self.run_steps(step["else"])

    def run_steps(self, steps: list):
        for step in steps:
            result = self.execute_step(step)
            self.context["previous_output"] = result

        return result

    def render(self, template: str) -> str:
        def replace(match):
            key = match.group(1)
            if key not in self.context:
                raise RuntimeError(f"❌ render 실패: {key} 없음")
            return str(self.context[key])

        return re.sub(r"\{\{(\w+)\}\}", replace, template)
