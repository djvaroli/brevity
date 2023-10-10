# Brevity
An application for summarizing large (and small) texts from URLs.

##  Installation

1. Create a new python virtual environment, e.g. with Conda
```shell
conda create -n brevity python=3.11 -y && activate brevity
```

2. Install dependencies
```shell
pip install -r requirements.txt
```

3. Paste your OpenAI API key into the `example.env` and then rename the file to `.env`. You can do that by running
```shell
mv example.env .env
``` 

4. Run FastAPI application
```shell
uvicorn app.main:app --reload
```

5. Navigate to docs page at `http://127.0.0.1:8000/docs`

## Example request

### Curl
```shell
curl -X 'GET' \
  'http://127.0.0.1:8000/api/v1/summarize/file?url=https%3A%2F%2Farxiv.org%2Fpdf%2F2309.10668.pdf&model=gpt-3.5-turbo-16k&summary_length=10%20-%2012%20sentences' \
  -H 'accept: application/json'
```

### Python Requests
```python
# install requests with ``pip install requests`` if not installed
import requests

# file to summarize
endpoint = "http://127.0.0.1:800/api/v1/summarize/file"

params = {
    "url": "https://arxiv.org/pdf/2309.10668.pdf",
    "model": "gpt-3.5-turbo-16k",  # pre-selected options, see `http://127.0.0.1:8000/docs`
    "summary_length": "10 - 12 sentences"    # pre-selected options see `http://127.0.0.1:8000/docs`
}
headers = {
    "accept": "application/json"
}

response = requests.get(url, params=params, headers=headers)

# To get the response as a JSON object:
data = response.json()

print(data)
```

## Limitations
* Currently the application is in development, so there might be bugs or errors that won't be handled properly. The quality will improve over time, but for the time being quality of generated summaries can vary depending on multiple factors.  
* The application currently uses a recursive approach to summarization, where by the larger text is broken down intom smaller "chunks", each of which is summarized using Chain of Density (CoD) prompting (https://arxiv.org/abs/2309.04269). The final summary is generated using a customized CoD prompt. This can cause issues with loss of key information, especially if a chunk is sliced at a informationally critical location. One idea to improve this is identify topics (e.g. use Louvain algorithm), group chunks by topics and summarize topics. Then generate the final summary from those. This is currently in progress.
* Only GPT models are supported, and it is recommended to use at least GPT-3.5-turbo-16k, as the GPT-3.5-turbo model can result in out-of-context errors, when the summarized content combined with the chunk content exceeds a model's given context length. There are several ways this can be addressed, but at the moment it is recommended to select the GPT-3.5-turbo-16k model. 
* There is evidence to suggest that LLMs may be better at identifying relevant info that is located towards the end or beginning of their context (https://arxiv.org/abs/2307.03172). This means poor chunk truncation and larger chunk sizes could lead to decreased summarization performance. I am currently invistgating ways that this issue can be mitigated.
 