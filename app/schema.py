from pydantic import BaseModel

from .prompts import SummaryLength, Summary


# model_name is string to avoid circular imports
# summary parameters will not be accessed outside of endpoint
# while AIModel will be used outside
class SummaryParameters(BaseModel):
    url: str 
    model_name: str = "gpt-3.5-turbo"
    summary_length: SummaryLength = SummaryLength.SHORT


class SummaryResponse(BaseModel):
    summary_parameters: SummaryParameters
    summary: Summary
    input_cost: float
    output_cost: float
    total_cost: float
    currency: str
    num_input_tokens: int
    num_output_tokens: int
