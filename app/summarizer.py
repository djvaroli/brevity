import os
from enum import Enum
from dataclasses import dataclass
from logging import getLogger, INFO

import tiktoken
import openai

from .prompts import SummarizationPrompt, Summary
from .display import print_to_console

logger = getLogger(__name__)
logger.setLevel(INFO)


class AIModel(Enum):
    gpt_3_5_turbo = "gpt-3.5-turbo"
    gpt_3_5_turbo_16k = "gpt-3.5-turbo-16k"
    gpt_4 = "gpt-4"
    gpt_4_32k = "gpt-4-32k"
    
    @staticmethod
    def get_context_length(model: "AIModel", buffer_fraction: float = 1.00) -> int:
        """Get the context length of the given model.

        Args:
            model (AIModel): model to get the context length of.
            buffer_fraction (float, optional): fraction of the context length to use as a buffer. Defaults to 1.0.
                This is included to avoid exceeding context length accidentally.
        
        Returns:
            int: context length of the given model.
        """

        model_to_context_length = {
            AIModel.gpt_3_5_turbo: buffer_fraction * 4000, # 4097
            AIModel.gpt_3_5_turbo_16k: buffer_fraction * 16000,
            AIModel.gpt_4: buffer_fraction * 1024,
            AIModel.gpt_4_32k: buffer_fraction * 1024
        }
        
        return int(model_to_context_length[model])



class AITextSummarizer:
    def __init__(
        self,
        model: AIModel,
        prompt: SummarizationPrompt
    ) -> None:
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model(model_name=self.model.value)
        self.prompt = prompt

    def _warn_if_not_natural_stop(
        self, 
        # openai_response can be a lot of options so hard to type for now
        openai_response
    ) -> None:
        """Raises a warning if the model did not come to a natural stop. 

        Args:
            openai_response: response from OpenAI API.
        """
        global_debug = os.environ.get("GLOBAL_DEBUG", False)

        finish_reason = openai_response.choices[0].finish_reason
        if finish_reason != "stop":
            if global_debug:
                print_to_console(
                    "WARNING: Model did not come to a natural stop.", 
                    color="yellow"
                )
            logger.warning(f"WARNING: Model did not come to a natural stop. Reason: {finish_reason}")

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in the given text.

        Args:
            text (str): text to count tokens of.

        Returns:
            int: number of tokens in the given text.
        """
        return len(self.tokenizer.encode(text))
    
    def estimate_cost(
        self, 
        text: str, 
        type: str = "input",
        precision: int = 2
    ) -> float:
        """Estimate the cost of generating a summary of the given text.

        Args:
            text (str): text to summarize.
            type (str, optional): type of text to summarize. Defaults to "input".
                Can be one of "input" or "output".
            precision (int, optional): number of decimal places to round to. Defaults to 2.

        Returns:
            float: cost of generating a summary of the given text.
        """
        cost_per_input_token = 0.001
        cost_per_output_token = 0.002
        cost_per_token = cost_per_input_token if type == "input" else cost_per_output_token

        n_tokens = self.count_tokens(text)
        return round(cost_per_token * n_tokens, precision)
    
    def summarize(
        self, 
        text: str,
        **kwargs
    ) -> Summary:
        """Create a summary of the given text.

        Args:
            text (str): text to summarize.

        Returns:
            Summary: summary of the text.
        """
        prompt = self.prompt.make(text)
        openai_response = openai.ChatCompletion.create(
            model=self.model.value,
            messages=[
                {"role": "user", "content": prompt}
            ],
            **kwargs
        )

        # check reason for stopping and warn if not natural stop
        # TODO: may be worth raising an exception here instead to communicate the error to the user
        self._warn_if_not_natural_stop(openai_response)

        content = openai_response.choices[0].message.content
        return self.prompt.extract_summary(content)