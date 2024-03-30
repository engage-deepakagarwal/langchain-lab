from re import template
from langchain import PromptTemplate, hub
from langchain_openai import AzureChatOpenAI
from langchain.agents import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)

from tools.tools import get_profile_url

def lookup(search_string: str) -> str:

    # Pick a model which supports ChatCompletion API
    llm = AzureChatOpenAI(temperature=0, azure_deployment='turbo')
    template = """Given the search context of the person {search_context_string} I want you to get me a link to their LinkedIn profile page.
        Your answer should only contain a URL.
    """

    prompt_template = PromptTemplate(
        template = template,
        input_variables = ["search_context_string"]
    )

    tools_for_agent = [
        Tool(name="Crawl Google for LinkedIn profile page", func=get_profile_url, description='Useful tool when you need to get the LinkedIn page URL of a person')
    ]

    # This prompt is going to help the LLM decide which tool to use.
    # The prompt can be viewed here which Langchain downloads and use it - https://smith.langchain.com/hub/hwchase17/react
    react_prompt = hub.pull("hwchase17/react")

    # Why dynamically plugin the prompt? - More flexibility, freedom to tweak the algorithm of choosing the correct tools.
    agent = create_react_agent(llm=llm, tools=tools_for_agent, prompt=react_prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools_for_agent, verbose=True)
    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(search_context_string=search_string)}
    )

    linked_profile_url = result["output"]
    return linked_profile_url