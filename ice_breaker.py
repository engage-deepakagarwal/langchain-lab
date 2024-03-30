from dotenv import load_dotenv
from langchain import LLMChain, PromptTemplate
# Use the AzureChatOpenAI instead of AzureOpenAI which is deprecated
from langchain_openai import AzureChatOpenAI

# The module name must use an underscore in the name
from agents.linkedin_lookup_agent import lookup
from third_parties.linkedin import scrape_linkedin_profile

# Passing 'override' flag to override existing environment variable with the same name.
load_dotenv(override=True)

if __name__ == '__main__':

    summary_template = """
    Given the information {information} about a person, summarize it and provide the text output for the following points:
    1. A short summary of the information
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template)

    # Pick a model which supports ChatCompletion API
    llm = AzureChatOpenAI(azure_deployment='turbo', temperature=0.1)
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)

    linkedin_profile_url = lookup(search_string = "deepak agarwal azure certified solutions architect")
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)
    result = chain.invoke(input={"information": linkedin_data})

    print(result)
