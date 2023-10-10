import json
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .display import print_to_console

_DEFAULT_COD_TEMPLATE = """
Article: {text}

You will generate increasingly concise, entity-dense summaries of the above article.

Repeat the following 2 steps 5 times.

Step 1. Identify 1-3 informative entities (";" delimited) from the article which are missing from the previously generated summary.

Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the missing entities.

A missing entity is:

- relevant to the main story,

- specific yet concise (5 words or fewer),

- novel (not in the previous summary),

- faithful (present in the article),

- anywhere (can be located anywhere in the article).

Guidelines:

- The first summary should be long ({n_sentences} sentences, ~{n_words} words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach ~{n_words} words.

- Make every word count: rewrite the previous summary to improve flow and make space for additional entities.

- Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".

- The summaries should become highly dense and concise yet self-contained, i.e., easily understood without the article.

- Missing entities can appear anywhere in the new summary.

- Never drop entities from the previous summary. If space cannot be made, add fewer new entities.

Remember, use the exact same number of words for each summary. Answer in complete and valid JSON. The JSON should be a correctly formatted list (length 5) of dictionaries whose keys are "Missing_Entities" and "Denser_Summary".
"""


_DEFUAULT_JOIN_SUMMARIES_TEMPLATE = """
Article chunk summaries: {text}

You will generate an increasingly coherent complete summary of an article by joining the above article chunk summaries.

Repeat the following 3 steps 5 times.

Step 1. Identify 1-3 informative entities (";" delimited) from the chunk summaries which are missing from the previously generated summary.

Step 2. Write a new summary of identical length which covers every entity and detail from the previous summary plus the missing entities.

Step 3. Generate a title for the summary you have just written.

A missing entity is:

- relevant to the provided chunk summaries of the article,

- specific yet concise (5 words or fewer),

- novel (not in the previous summary),

- faithful (present in the chunk summaries),

- anywhere (can be located anywhere in the chunk summaries).

Guidelines:

- The first summary should be long ({n_sentences} sentences, ~{n_words} words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach ~{n_words} words.

- Make every word count: rewrite the previous summary to improve flow and make space for additional entities.

- The summaries should become highly dense and concise yet self-contained, i.e., easily understood without the originak article.

- Missing entities can appear anywhere in the new summary.

- Never drop entities from the previous summary. If space cannot be made, add fewer new entities.

Remember, use the exact same number of words for each summary. Answer in complete and valid JSON. The JSON should be a correctly formatted list (length 5) of dictionaries whose keys are "Missing_Entities", "Denser_Summary" and "Title".
"""


@dataclass
class Summary:
    title: str
    content: str


class SummaryLength(str, Enum):
    SHORT = "3 - 5 sentences"
    MEDIUM = "10 - 12 sentences"
    LONG = "25 - 30 sentences"

    @staticmethod
    def estimate_length_in_words(summary_length: "SummaryLength") -> str:
        lengt_to_words = {
            SummaryLength.SHORT: "80",
            SummaryLength.MEDIUM: "160",
            SummaryLength.LONG: "320",
        }

        n_words = lengt_to_words.get(summary_length, None)

        if n_words is None:
            raise ValueError("Invalid summary length.")

        return n_words


class SummarizationPrompt:
    def __call__(self, text: str) -> Any:
        return text

    def make(self, text: str) -> Any:
        return self(text)

    def extract_summary(self, text: str) -> Summary:
        return Summary(title="Title", content=text)


class ChainOfDensityPrompt(SummarizationPrompt):
    def __init__(
        self,
        summary_length: SummaryLength = SummaryLength.SHORT,
        template: str = _DEFAULT_COD_TEMPLATE,
    ) -> None:
        self.template = template
        self.summary_length = summary_length

    def __call__(
        self,
        text: str,
    ) -> str:
        return self.template.format(
            text=text,
            n_sentences=self.summary_length.value,
            n_words=SummaryLength.estimate_length_in_words(self.summary_length),
        )

    def extract_summary(self, model_response: str) -> Summary:
        """ """
        # expected output format is a dictionary with keys "Missing_Entities" and "Denser_Summary"
        # we want the last output
        print_to_console(model_response)

        try:
            model_response = json.loads(model_response)
            summary = model_response[-1]["Denser_Summary"]
        except json.JSONDecodeError as e:
            raise RuntimeError("Model output could not be decoded.") from e
        except KeyError as e:
            raise KeyError(
                "Model output does is missing ``Denser_Summary`` key."
            ) from e

        return Summary(title="Title", content=summary)


class JoinSummariesPrompt(SummarizationPrompt):
    def __init__(
        self,
        summary_length: SummaryLength = SummaryLength.MEDIUM,
        template: str = _DEFUAULT_JOIN_SUMMARIES_TEMPLATE,
    ) -> None:
        self.template = template
        self.summary_length = summary_length

    def __call__(
        self,
        text: str,
    ) -> str:
        return self.template.format(
            text=text,
            n_sentences=self.summary_length.value,
            n_words=SummaryLength.estimate_length_in_words(self.summary_length),
        )

    def extract_summary(self, model_response: str) -> Summary:
        """ """
        # expected output format is a dictionary with keys "Missing_Entities" and "Denser_Summary"
        # we want the last output
        try:
            model_response = json.loads(model_response)
            summary = model_response[-1]["Denser_Summary"]
            title = model_response[-1]["Title"]
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError("Model output could not be decoded.") from e
        except KeyError as e:
            raise KeyError(
                "Model output does is missing ``Denser_Summary`` or ``Title`` key."
            ) from e

        return Summary(title=title, content=summary)
