import re

class Agent:
    def __init__(self, flow: dict, llm, tools: dict):
        self.flow = flow
        self.llm = llm
        self.tools = tools
        self.context = {}
        self.logs = []
    
    def run(self):
        self.context.update(self.flow.get("inputs", {}))
        self.run_steps(self.flow["steps"])
        return self.logs

    def run_steps(self, steps):
        result = None
        for step in steps:
            result = self.execute_step(step)
            self.context["previous_output"] = result
        return result


    def execute_step(self, step):
        if "if" in step:
            return self.execute_if(step)

        if "for_each" in step:
            return self.execute_loop(step)

        if step["type"] == "llm":
            prompt = self.render(step["prompt"])
            result = self.llm.run(prompt)
            self.logs.append(result)
            return result

        if step["type"] == "tool":
            tool = self.tools[step["tool_name"]]
            tool_input = self.render(step["input"])
            result = tool.run(tool_input)
            self.logs.append(result)
            return result

        raise RuntimeError(f"알 수 없는 step: {step}")

   
    def execute_if(self, step):
        condition = self.render(step["if"])
        if eval(condition):
            return self.run_steps(step.get("then", []))
        else:
            return self.run_steps(step.get("else", []))

    def execute_loop(self, step):
        items = self.context.get(step["for_each"], [])
        var_name = step.get("as", "item")

        results = []
        for item in items:
            self.context[var_name] = item
            result = self.run_steps(step["steps"])
            results.append(result)
            
        return results

    def render(self, template: str) -> str:
        def replace(match):
            key = match.group(1)
            if key not in self.context:
                raise RuntimeError(f"❌ render 실패: {key} 없음")
            return str(self.context[key])

        return re.sub(r"\{\{(\w+)\}\}", replace, template)
