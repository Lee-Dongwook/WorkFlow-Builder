class Agent:
    def __init__(self, flow: dict, llm):
        self.flow = flow
        self.llm = llm
        self.context = {}
    
    def run(self):
        result = None

        for step in self.flow["steps"]:
            if step["type"] == "llm":
                result = self.llm.run(step["prompt"])
                self.context["previous_output"] = result

        return result
