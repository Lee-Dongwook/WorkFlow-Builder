class BaseTool:
    name: str

    def run(self, input: str) -> str:
        raise NotImplementedError


class UppercaseTool(BaseTool):
    name = "uppercase"

    def run(self, input: str) -> str:
        return input.upper()
