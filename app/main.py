import requests
import os
from typing import List

from fastapi import FastAPI
from fastapi.exceptions import HTTPException
import openai
from dotenv import load_dotenv
import tiktoken

from .prompts import (
    ChainOfDensityPrompt, 
    SummaryLength, 
    Summary,
    JoinSummariesPrompt
)
from .file.remote import RemoteFile
from .schema import SummaryParameters, SummaryResponse
from .summarizer import AITextSummarizer, AIModel
from .display import print_to_console


load_dotenv(".env")
openai.api_key = os.environ["OPENAI_API_KEY"]
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hey, I'm Arxiver!"}



def split_into_chunks(
    text: str, 
    max_tokens_per_chunk: int,
    model_name: str
) -> List[str]:
    chunks = []
    tokenizer = tiktoken.encoding_for_model(model_name=model_name)
    tokens = tokenizer.encode(text)
    for i in range(0, len(tokens), max_tokens_per_chunk):
        chunks.append(tokenizer.decode(tokens[i:i+max_tokens_per_chunk]))
    return chunks


def chunk_and_summarize(
    text: str,
    summarizer: AITextSummarizer,
    max_tokens_per_chunk: int
) -> Summary:
    """Summarize text over the context length of the model.

    Args:
        text (str): _description_
        summarizer (AITextSummarizer): _description_
        max_chunk_length (int): _description_

    Raises:
        HTTPException: _description_
        HTTPException: _description_

    Returns:
        Summary: _description_
    """
    # split the text into chunks
    chunks = split_into_chunks(
        text, 
        max_tokens_per_chunk, 
        summarizer.model.value
    )

    # summarize each chunk
    summaries = []
    for chunk_index, chunk in enumerate(chunks):
        if os.environ.get("GLOBAL_DEBUG", False):
            n_chunk_tokens = summarizer.count_tokens(chunk)
            with open(f"chunk-{chunk_index}-{n_chunk_tokens}.txt", "w") as f:
                f.write(chunk)
        
        summary = summarizer.summarize(chunk, temperature=0.5)
        if os.environ.get("GLOBAL_DEBUG", False):
            print_to_console(
                content=summary.content,
                heading=f"Summary of chunk {chunk_index + 1}"
            )
        
        summaries.append(summary)

    # write summaries to file
    if os.environ.get("GLOBAL_DEBUG", False):
        with open("summaries.txt", "w") as f:
            f.write("\n***\n".join([summary.content for summary in summaries]))
    
    # join individual summaries into a single summary
    join_summarizer = AITextSummarizer(
        model=summarizer.model,
        prompt=JoinSummariesPrompt()
    )
    joined_summary = join_summarizer.summarize(
        "\n***\n".join([summary.content for summary in summaries])
    )

    if os.environ.get("GLOBAL_DEBUG", False):
        print_to_console(joined_summary.content, heading="Joined summary", color="blue")
    
    return joined_summary


@app.post("/api/v1/summarize/file")
async def summarize_file_at_url(
    url: str = "https://arxiv.org/pdf/2309.10668.pdf",
    model: AIModel = AIModel.gpt_3_5_turbo,
    summary_length: SummaryLength = SummaryLength.MEDIUM
) -> SummaryResponse:
    """Given a URL pointing to a file, fetch the file and return a summary of the content.

    Args:
        summary_parameters (SummaryParameters): parameters for summarization.
    """
    summary_parameters = SummaryParameters(
        url=url,
        model_name=model.value,
        summary_length=summary_length
    )

    remote_file = RemoteFile(summary_parameters.url)

    try:
        remote_file.download()
    except requests.exceptions.HTTPError as e:
        raise HTTPException(
            status_code=400, 
            detail=f"Unable to download file at URL: {summary_parameters.url}. An HTTP error occurred."
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Unable to download file at URL: {summary_parameters.url}. A server error occurred."
        )
    
    file_text_content = remote_file.extract_text()

    prompt = ChainOfDensityPrompt(
        summary_length=summary_parameters.summary_length
    )

    summarizer = AITextSummarizer(model=model, prompt=prompt)
    
    # input.len + output.len must be <= context.len, lower buffer means more room for output
    # gpt-3.5-turbo can fail even at 0.75, so we use 0.65
    # Also, https://arxiv.org/abs/2307.03172 shows performance drops when key info in middle
    # reducing size of input text chunks may yield better summaries
    model_context_length = AIModel.get_context_length(model, buffer_fraction=0.65)
    prompt_n_tokens = summarizer.count_tokens(prompt.make(file_text_content))

    print_to_console(f"Number of tokens in prompt: {prompt_n_tokens}")
    print_to_console(f"Model context length: {model_context_length}")

    if prompt_n_tokens >= model_context_length:
        summary = chunk_and_summarize(
            text=file_text_content,
            summarizer=summarizer,
            max_tokens_per_chunk=model_context_length
        )
    else:
        summary = summarizer.summarize(file_text_content)
    
    # get an idea of how much the summary cost
    input_cost = summarizer.estimate_cost(file_text_content)
    output_cost = summarizer.estimate_cost(summary.content, type="output")

    # remove the downloaded file
    remote_file.delete()
    
    # other potentially useful metadata
    n_input_tokens = summarizer.count_tokens(prompt.make(file_text_content))
    n_output_tokens = summarizer.count_tokens(summary.content)

    print_to_console(summary.content, heading=summary.title, color="blue")

    return SummaryResponse(
        summary_parameters=summary_parameters,
        summary=summary,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
        currency="USD",
        num_input_tokens=n_input_tokens,
        num_output_tokens=n_output_tokens
    )
