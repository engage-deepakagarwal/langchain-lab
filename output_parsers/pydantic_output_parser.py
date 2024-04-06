from typing import List
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

class PersonIntel(BaseModel):
    summary: str = Field(description="summary")
    facts: List[str] = Field(description="interesting facts about them")
    ice_breakers: List[str] = Field(description="ice breaker list")
    topics_of_interest: List[str] = Field(description="topic that might interest the person")

    def to_dict(self):
        return {
            "summary": self.summary, 
            "facts": self.facts, 
            "ice_breakers": self.ice_breakers, 
            "topics_of_interest": self.topics_of_interest
        }

# pydantic_object is the class of the object we want to parse later 
person_intel_parser = PydanticOutputParser(pydantic_object=PersonIntel)
