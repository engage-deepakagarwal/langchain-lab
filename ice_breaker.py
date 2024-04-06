from dotenv import load_dotenv
from langchain import LLMChain, PromptTemplate
# Use the AzureChatOpenAI instead of AzureOpenAI which is deprecated
from langchain_openai import AzureChatOpenAI

# The module name must use an underscore in the name
from agents.linkedin_lookup_agent import lookup
from third_parties.linkedin import scrape_linkedin_profile
from output_parsers.pydantic_output_parser import PersonIntel, person_intel_parser

# Passing 'override' flag to override existing environment variable with the same name.
load_dotenv(override=True)

def ice_break_with(search_string: str) -> PersonIntel:
    linkedin_profile_url = lookup(search_string = search_string)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)
    
    summary_template = """
        Given a LinkedIn information about a person, create:
    1. A short summary 
    2. Two interesting facts about them
    3. A topic that may interest them
    4. Two creative ice breakers to open a conversation with them
    Here is the LinkedIn information : {linkedin_information}
    \n{format_instructions}     
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["linkedin_information"],
        template=summary_template,
        partial_variables={"format_instructions": person_intel_parser.get_format_instructions()}
    )

    # Pick a model which supports ChatCompletion API
    llm = AzureChatOpenAI(azure_deployment='turbo', temperature=0.1)
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    
    result = chain.invoke(input={"linkedin_information": linkedin_data})
    print(result)
    return person_intel_parser.parse(result)

if __name__ == '__main__':
    ice_break_with("deepak agarwal azure certified solutions architect")
    
