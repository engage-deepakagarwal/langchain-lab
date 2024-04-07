from typing import Union, List
import re
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_openai import AzureChatOpenAI
from langchain.agents import Tool

from callbacks import AgentCallbackHandler
from tools.text_length import get_text_length

load_dotenv()


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


if __name__ == '__main__':
    print("Hello ReAct Langchain!")

    # Step 1: Creating Tools
    tools = [get_text_length]

    # Step 2: The ReAct template
    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad}
    """

    # Step 3: Creating PromptTemplate
    prompt_template = PromptTemplate.from_template(template=template).partial(
        tools="\n".join([f"{tool.name}: {tool.description}" for tool in tools]),
        tool_names=", ".join([tool.name for tool in tools]))

    # Step 4: LLM
    # So this will tell the LLM to stop generating words and to finish working once it's outputted the "Observation"
    # token. And why do we need it? Because if we won't put the stop token, then the LLM would continue to
    # generate text and it's going to guess one word after another observation. And the observation is the result
    # of the tool. And this is something that will come from running our tool.
    llm = AzureChatOpenAI(temperature=0, azure_deployment='turbo', stop=["Observation"], callbacks=[AgentCallbackHandler()])

    # This will hold the history of the agent invocations a.k.a agent_scratchpad
    intermediate_steps = []

    # Step 5: ReAct agent The lambda function will receive a dictionary and is accessing the input key of that
    # dictionary. The actual 'input' to the prompt_template will be provided when we invoke the chain later.
    # The ReAct style output of LLM will be provided to ReActSingleInputOutputParser to parse
    agent = {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])
                # formatting to str is required as the intermediate_steps tuple contains AgentAction which is a
                # Langchain object format_log_to_str in-built Langchain function provides string representation in
                # ReAct format
            } | prompt_template | llm | ReActSingleInputOutputParser()

    agent_step = None

    while not isinstance(agent_step, AgentFinish):
        # Step 6: Invoking from agent The prompt written here needs to be careful as the agents will look up the tool
        # names and descriptions to closely match the prompt input provided
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length in characters of the text: DOG",
                "agent_scratchpad": intermediate_steps

            }
        )

        # Step 7: Tool execution
        # If the next agent step from ReAct style LLM output is AgentAction, execute the tool
        if isinstance(agent_step, AgentAction):
            # Get the tool name parsed from ReActSingleInputOutputParser
            tool_name = agent_step.tool

            # Get the type 'Tool' from the tool name to execute it
            tool_to_use = find_tool_by_name(tools, tool_name)

            tool_input = agent_step.tool_input
            # This is just nice to have in case the input parsed has special characters. We know the input is
            # alpha-numeric. i.e. DOG in this case
            tool_input = re.sub('[^A-Za-z0-9]+', '', tool_input)

            # Execute the tool (a.k.a. function) and return the observation
            observation = tool_to_use.func(str(tool_input))

            # Every time we run an iteration, we want to update this list and append to it the history
            # and what we have performed.
            intermediate_steps.append((agent_step, observation))
            print(f"Input: {str(tool_input)}: Observation: {observation}")

        # Step 8: Since previous agent_step was AgentAction we will run another iteration until we get AgentFinish
        # However, we will run the next iteration with all the context of previous iteration details too so LLM
        # can reason it well. For this we added {agent_scratchpad} which will hold intermediate results

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)
