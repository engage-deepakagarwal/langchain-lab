from langchain.agents import tool


@tool
def get_text_length(text: str) -> int:
    """ Returns the length of a text by characters """
    return len(text)