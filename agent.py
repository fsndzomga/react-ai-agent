from anonLLM.llm import OpenaiLanguageModel as Brain
from typing import Callable
from dotenv import load_dotenv
from pydantic import BaseModel
import inspect
import datetime
import wikipedia
import os

load_dotenv()


class ReactEnd(BaseModel):
    stop: bool
    final_answer: str


class ToolChoice(BaseModel):
    tool_name: str
    reason_of_choice: str


class Tool:
    def __init__(self, name: str, func: Callable) -> None:
        self.name = name
        self.func = func

    def act(self, **kwargs) -> str:
        return self.func(**kwargs)

class Agent:
    def __init__(self, api_key: str = os.environ.get('OPENAI_API_KEY'), model: str = 'gpt-4') -> None:
        self.api_key = api_key
        self.model = model
        self.brain = Brain(api_key=api_key,
                           model=model,
                           anonymize=False)  # I know it violates the dependency injection rule,
                                            # but it is not a big issue here.
        self.tools = []

        self.messages = []

        self.request = ""

        self.token_count = 0

        self.token_limit = 5000

    def add_tool(self, tool: Tool) -> None:
        self.tools.append(tool)

    def append_message(self, message):
        self.messages.append(message)
        self.token_count += len(message)

        # Check if token_count exceeds the limit
        while self.token_count > self.token_limit and len(self.messages) > 1:
            # Remove messages from the end until token_count is within the limit
            removed_message = self.messages.pop(1)  # Keep the first message, remove the second one
            self.token_count -= len(removed_message)

    @staticmethod
    def extract_first_nested_dict(data_dict):
        for key, value in data_dict.items():
            if isinstance(value, dict):
                return value
        return {}

    def background_info(self) -> str:
        return f"Here are your previous think steps: {self.messages[1:]}" if len(self.messages) > 1 else ""

    def think(self) -> None:

        prompt = f"""Answer the following request as best you can: {self.request}.
                    {self.background_info()}
                    First think about what to do. What action to take first if any.
                    Here are the tools at your disposal: {[tool.name for tool in self.tools]}"""

        self.append_message(prompt)

        response = self.brain.generate(prompt=prompt, max_tokens=100)

        print(f"Thought: {response}")

        self.append_message(response)

        self.choose_action()

    def choose_action(self) -> None:
        prompt = f"""To Answer the following request as best you can: {self.request}.
                    {self.background_info()}
                    Choose the tool to use if need be. The tool should be among:
                    {[tool.name for tool in self.tools]}.
                    """
        self.append_message(prompt)

        response = self.brain.generate(prompt=prompt, output_format=ToolChoice)

        print(f"""Action: I should use this tool: {response["tool_name"]}.
              {response["reason_of_choice"]}""")

        self.append_message(response)

        tool = [tool for tool in self.tools if tool.name == response["tool_name"]].pop()

        self.action(tool)

    def action(self, tool: Tool) -> None:
        prompt = f"""To Answer the following request as best you can: {self.request}.
                    {self.background_info()}
                    Determine the inputs to send to the tool: {tool.name}
                    Given that the source code of the tool function is: {inspect.getsource(tool.func)}.
                    """
        self.append_message(prompt)

        parameters = inspect.signature(tool.func).parameters

        class DynamicClass(BaseModel):
            pass

        for name, param in parameters.items():
            # Setting default value if it exists, else None
            default_value = param.default if param.default is not inspect.Parameter.empty else None
            setattr(DynamicClass, name, (param.annotation, default_value))

        response = self.brain.generate(prompt=prompt, output_format=DynamicClass)

        self.append_message(response)

        input_parameters = self.extract_first_nested_dict(response)

        try:
            action_result = tool.func(**input_parameters)

            self.append_message(f"Results of action: {action_result}")

            self.observation()

        except:
            self.action(tool)

    def observation(self) -> None:
        prompt = f"Observation:{self.messages[-1]}."
        self.append_message(prompt)

        check_final = self.brain.generate(prompt=f"Is {self.background_info()} enough to finally answer to this request: {self.messages[0]}",
                 output_format=ReactEnd)

        if check_final["stop"]:
            print("Thought: I now know the final answer. \n")
            prompt = f"""Give the final answer the following request: {self.request}.
                    given {self.background_info()}
                    """
            print(f"Final Answer: {self.brain.generate(prompt=prompt)}")
        else:
            self.think()


    def react(self, input: str) -> str:
        self.append_message(input)
        self.request = input
        self.think()


# Equivalent of the perform_calculation function
def perform_calculation(operation, a, b):
    # Validating the operation
    if operation not in ['add', 'subtract', 'multiply', 'divide']:
        return f"Invalid operation: {operation}, should be among ['add', 'subtract', 'multiply', 'divide']"

    if operation == 'add':
        return a + b
    elif operation == 'subtract':
        return a - b
    elif operation == 'multiply':
        return a * b
    elif operation == 'divide':
        if b == 0:
            return "Division by zero"
        return a / b


# Equivalent of the search_wikipedia function
def search_wikipedia(search_query):
    # Fetch the page content

    page = wikipedia.page(search_query)

    # Extract the text
    text = page.content

    # Print and return the first 100 characters
    return text[:300]


# Equivalent of the date_req function
def date_of_today():
    return datetime.date.today()


# Creating instances of the Tool class
wikipedia_search_tool = Tool("WikipediaSearch", search_wikipedia)
calculator_tool = Tool("Calculator", perform_calculation)
date_request_tool = Tool("Date_of_request", date_of_today)

# Creating an Agent

agent = Agent()

agent.add_tool(wikipedia_search_tool)
agent.add_tool(calculator_tool)
agent.add_tool(date_request_tool)

agent.react("What is the double of barack obama's age?")
