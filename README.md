# Agentic AI Programming Test: Two-Agent RAG

This project implements the challenge with the OpenAI Agents SDK and an
OpenAI-compatible API Management gateway for GPT-5-mini. It uses two agents in
a sequential workflow:

```text
User question
    |
    v
Data Retriever Agent
    |
    | calls search_local_knowledge
    v
knowledge_base.txt -> BM25 + query expansion + tag boost
    |
    v
Raw snippets with chunk IDs
    |
    v
Report Generator Agent
    |
    v
Grounded, cited final answer
```

The Data Retriever is configured with `tool_choice="required"` and
`tool_use_behavior="stop_on_first_tool"`. Therefore, it must search and its raw
tool result becomes its final output; it cannot replace the evidence with its
own answer. Python orchestration then passes that exact result to the Report
Generator.

This direct sequence uses two model calls per question. It is more suitable for
the free 1,000-token-per-minute key than a manager/agent-as-tool loop, which
requires three model calls.

## Project structure

```text
.
|-- src/
|   |-- agents.py       # Agent definitions and orchestration
|   |-- config.py       # Safe environment configuration
|   |-- main.py         # CLI and demo queries
|   `-- retrieval.py    # Local deterministic RAG retrieval
|-- tests/
|-- screenshots/
|-- knowledge_base.txt
|-- .env.example
`-- requirements.txt
```

## Setup

Python 3.10 or newer is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
```

Edit `.env` and replace only this value:

```dotenv
AZURE_OPENAI_API_KEY=replace-with-your-api-key
RATE_LIMIT_MAX_RETRIES=2
RATE_LIMIT_RETRY_SECONDS=30
DEMO_DELAY_SECONDS=60
```

Configure the gateway base URL and model in `.env`. The application uses this
request contract:

```text
POST <AZURE_OPENAI_ENDPOINT>/responses
api-key: <AZURE_OPENAI_API_KEY>
Content-Type: application/json
```

The `.env` endpoint ends at `/llm/`; the OpenAI Responses client appends
`responses`. The client deliberately suppresses its default Bearer header
because this gateway authenticates with `api-key`.

> Never commit `.env`. It is ignored by Git. Commit `.env.example`, which
> contains no secret.

### Wait and retry configuration

| Setting | Purpose | Disable |
| --- | --- | --- |
| `RATE_LIMIT_MAX_RETRIES` | Number of HTTP 429 retries | Set to `0` |
| `RATE_LIMIT_RETRY_SECONDS` | Fallback delay when the gateway sends no reset header | Used only when retries are enabled |
| `DEMO_DELAY_SECONDS` | Proactive delay between `--demo` questions | Set to `0` |

When retries are enabled, a gateway-provided `retry-after` or token-reset value
takes priority over the fallback delay. To guarantee that the application never
waits after a 429, set `RATE_LIMIT_MAX_RETRIES=0`.

Show the effective configuration without displaying the API key:

```powershell
python -m src.main --show-config
```

## Run

Ask one question:

```powershell
python -m src.main "What is the policy on international travel?"
```

Use the interactive prompt:

```powershell
python -m src.main
```

Run four sample queries for evaluation screenshots:

```powershell
python -m src.main --demo
```

The demo uses `DEMO_DELAY_SECONDS` between questions (60 seconds by default).
This avoids starting a new two-agent workflow inside the same token-per-minute
window. A command-line value overrides the environment for one run:

```powershell
python -m src.main --demo --demo-delay 10
```

To run the demo with no proactive delay:

```powershell
python -m src.main --demo --demo-delay 0
```

The demo covers direct retrieval, a paraphrased request, a multi-chunk answer,
and an unknown topic that must not be answered from outside knowledge.

If either agent receives HTTP 429, only that failed agent stage is retried. The
program honors `retry-after`, `retry-after-ms`, or
`x-ratelimit-reset-tokens` response headers and otherwise waits
`RATE_LIMIT_RETRY_SECONDS`. A successful Retriever call is not repeated when
only the Report Generator was rate-limited.

## Test

The default retrieval, configuration, orchestration, API-contract, and retry
tests do not call the LLM and do not
consume API quota:

```powershell
python -m pytest -q
```

Run the opt-in live quality test only when you intend to consume API quota:

```powershell
$env:RUN_LIVE_TESTS = "1"
python -m pytest tests/test_live_workflow.py -q -s
Remove-Item Env:RUN_LIVE_TESTS
```

The live case checks that the answer includes the 30-day rule, manager and
department-head approvals, the `[TRAVEL-001]` citation, and no observed
`byyour` spacing defect. It uses the same automatic 429 retry behavior as the
CLI.

## Retrieval approach

The custom search tool:

1. Parses independently citable chunks from `knowledge_base.txt`.
2. Tokenizes the query and removes domain-generic stop words.
3. Adds a small, inspectable synonym expansion for phrases such as
   `overseas -> international` and `paperwork -> documents`.
4. Ranks chunks with BM25 and a soft tag boost.
5. Returns the top raw snippets with stable chunk IDs.

This is RAG without a vector database: retrieval finds external evidence,
augmentation passes that evidence to the Report Generator, and generation
creates a grounded answer. The design works with the supplied GPT-5-mini
deployment and does not require a separate embedding model.

## Screenshot checklist

After adding the real key, run `python -m src.main --demo` and capture:

1. International travel approval.
2. Expense-submission deadline.
3. Overseas approval plus flight-class rules.
4. Missing pet policy (`NOT_FOUND` behavior).

Place selected images in `screenshots/`. Before committing them, check that the
images contain no key, terminal history, email address, or other private
information.

## References

- [OpenAI Agents SDK: tool-use behavior](https://openai.github.io/openai-agents-python/agents/#tool-use-behavior)
- [OpenAI Agents SDK: running agents](https://openai.github.io/openai-agents-python/running_agents/)
- [OpenAI Agents SDK: model configuration](https://openai.github.io/openai-agents-python/models/)
