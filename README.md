# SHL Conversational Assessment Recommender

A FastAPI service that takes a hiring manager from a vague intent
("I'm hiring a Java developer") to a grounded shortlist of SHL
assessments through conversation. Built for the SHL Labs AI Intern
take-home.

## Endpoints

- `GET /health` -> `{"status": "ok"}`
- `POST /chat` -> stateless multi-turn. See `app/schemas.py` for the
  exact request/response shape.

## Architecture

```
+----------------+      +-----------------+      +-----------------+
|  FastAPI       | ---> |  Agent          | ---> |  Catalog        |
|  /chat handler |      |  (state machine)|      |  BM25 + dense   |
+----------------+      +--------+--------+      +--------+--------+
                                 |                        |
                                 v                        |
                          +-------------+                 |
                          |  Groq LLM   |                 |
                          |  (classify  |                 |
                          |   + reply)  |                 |
                          +-------------+                 |
                                                          v
                                                  +---------------+
                                                  | catalog.json  |
                                                  | (scraped)     |
                                                  +---------------+
```

**Per request flow:**

1. Validate `messages` payload against the schema.
2. Agent classifies the latest user turn via a Groq JSON-mode call:
   `off_topic | vague | recommend | refine | compare`.
3. Hard rules override the classifier:
   - never recommend on turn 1 unless the user pasted a long JD (>=200 chars);
   - force commit to a shortlist if we are about to hit the 8-turn cap.
4. Dispatch:
   - **off_topic** -> short refusal, empty recs.
   - **vague** -> one clarifying question, empty recs.
   - **recommend / refine** -> hybrid retrieval (RRF over BM25 + MiniLM
     embeddings), optionally filtered by required test types / time
     budget, top-10 hydrated from the catalog.
   - **compare** -> name lookup over catalog, build a fact sheet, ask
     the LLM to compare using only those facts.

Recommendations always come from the catalog object. The LLM never
writes URLs or assessment names into the structured response.

## Quickstart (local)

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Set up env
cp .env.example .env
# edit .env and add GROQ_API_KEY (free key at https://console.groq.com/keys)

# 3. Build the catalog
python scripts/scrape_catalog.py
# (Add --limit 20 for a smoke test.)

# 4. Run
uvicorn app.main:app --reload --port 8000

# 5. Try it
curl -s localhost:8000/health
curl -s -X POST localhost:8000/chat \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hiring a mid-level Java dev who works with stakeholders"}]}' | jq .
```

## Tests

```bash
# Unit tests (no LLM needed)
pytest tests/test_units.py -v

# Trace replay (needs LLM + catalog + traces in data/traces/*.json)
python tests/replay.py data/traces/
```

The replay harness simulates an LLM user from a persona + fact set,
runs a full multi-turn conversation, then reports Recall@10 and
behavior-probe pass rate -- matching what SHL's evaluator does.

## Deploy to Render

1. Push this repo to GitHub.
2. In Render: New -> Blueprint -> point at this repo. `render.yaml`
   is detected.
3. Set `GROQ_API_KEY` in the Render dashboard (it's marked `sync:
   false` so the YAML doesn't leak it).
4. Render builds the Docker image. First build runs the embedding
   model download in the image build, so cold start is fast.
5. **Important:** before pushing, build the catalog locally and commit
   `data/catalog.json`, OR add a `preDeployCommand` in `render.yaml`
   that runs the scraper. The current setup expects the file to be
   committed.

## Notes on the design

See `APPROACH.md` for the 2-page write-up covering decisions,
trade-offs, what didn't work, and evaluation methodology.
