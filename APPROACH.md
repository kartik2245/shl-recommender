# Approach

## Stack & justification

- **FastAPI + Pydantic v2** because the spec is schema-driven and Pydantic
  gives me free request/response validation that aligns with SHL's hard
  evals (schema compliance, item count between 1 and 10).
- **Groq + Llama 3.3 70B** as the LLM. Free tier, sub-second latency, JSON
  mode. The 70B handles intent classification reliably; I keep the 8B
  instant model wired as a fallback for rate-limit fallback.
- **Hybrid retrieval: BM25 (rank-bm25) + MiniLM dense embeddings
  (sentence-transformers/all-MiniLM-L6-v2), fused with Reciprocal Rank
  Fusion (RRF, k=60).** The catalog mixes precise product names ("Java 8
  (New)", "OPQ32r", "Verify G+") with conceptual descriptions ("works
  with stakeholders"). BM25 wins on the former; dense wins on the
  latter; RRF avoids weight tuning. MiniLM is 80MB, runs on CPU at
  thousands of sentences/sec, and is small enough to fit on Render's
  free tier.
- **Render + Docker** because the brief lists it as a free option, and
  `render.yaml` gives me reproducible deploys. The embedding model is
  baked into the image at build time so cold-start `/health` returns in
  well under the 2-minute allowance.
- **No LangChain / LlamaIndex.** The agent is a 200-line state machine
  with explicit dispatch on five intents. A framework would have added
  indirection without saving lines, and the brief specifically warns
  against design choices I can't defend.

## Catalog & retrieval

`scripts/scrape_catalog.py` pulls Individual Test Solutions only (the
catalog page has `type=2` in its query string for that tab; `type=1` is
the out-of-scope Pre-packaged Job Solutions tab). It paginates by 12,
parses each row for name + URL + remote/IRT flags + test-type letters,
then follows each detail page for description, job levels, languages,
and assessment length. Output is one JSON file with a deterministic
schema, committed for reproducibility.

At startup the `Catalog` class builds two indices over a concatenated
"search text" field per item (name + description + job levels +
types + length): a BM25 over tokenized text, and a normalized MiniLM
embedding matrix for cosine-by-dot-product. `retrieve()` ranks each
side independently, fuses with RRF, returns top-k. `filter_retrieve()`
over-fetches and post-filters on required test types or a maximum
duration -- this is how the agent handles refinements like "add a
personality test" or "under 30 minutes" without restarting retrieval.

For compare flows, `find_by_name()` does a case-insensitive substring
match first and falls back to retrieval over names. This is more
robust than fuzzy string distance for SHL's naming style (versioned
products like "OPQ32r" don't tolerate Levenshtein noise well).

## Agent design

Five intents: `off_topic`, `vague`, `recommend`, `refine`, `compare`.
A single Groq JSON-mode call classifies the latest turn given the
whole history and extracts a search query, required test types, max
minutes, compare names, and a clarify question all in one shot.

Two hard rules override the classifier:

1. **Never recommend on turn 1** for short queries. If the user pasted
   a long job description (>=200 chars), they have given enough
   context and we go straight to recommend; otherwise we ask one
   question first. This directly targets the "agent does not recommend
   on turn 1 for a vague query" behavior probe in the brief.
2. **Force commit at the turn cap.** If the next assistant reply would
   push the total above 8, we override `vague` with `recommend` using
   whatever context we have. The system never leaves a trace without a
   shortlist due to over-clarification.

Recommendations are always rebuilt from `Item` objects returned by
the retriever. The LLM only writes the natural-language `reply`
string. This is the single most important grounding decision: by
construction the agent cannot invent URLs or out-of-catalog
assessments, which makes the "items from catalog only" hard eval and
the hallucination probe pass deterministically.

## Prompt design

- The classifier prompt is structured JSON-out only, with the SHL
  test-type legend included so the LLM can map intent ("personality
  test") to codes ("P"). Temperature 0.0.
- The recommend reply prompt explicitly says NOT to list the
  assessments by name in the reply -- they appear separately in the
  structured field. This avoids redundancy and keeps replies short.
- The compare reply prompt is given a pre-built fact sheet (name, URL,
  types-with-legend, job levels, length, truncated description) and
  told to use only those facts. This is the only flow where the LLM
  could plausibly hallucinate, and the fact-sheet constraint shuts it
  down.
- Refusal prompt is two sentences max, no lecturing. Off-topic checks
  happen at classification, so the refusal path never sees the
  injected content as instructions.

## Evaluation

`tests/replay.py` mirrors what SHL describes: it spawns an LLM-simulated
user given a persona + fact set, runs a real multi-turn conversation
against the agent, ends when a shortlist is returned, and computes
Recall@10 against an expected name list plus behavior-probe pass rate
(no recommend on turn 1, off-topic refusal, refinement honored).
`tests/test_units.py` covers the pieces that don't need an LLM:
schema constraints, BM25/dense ranking on a tiny fixture catalog, and
the regex helpers.

I iterated by running the replay loop, watching where Recall@10 dropped,
and either adjusting the retrieval query synthesis (the classifier's
`search_query` output) or the prompt. The single biggest gain came
from adding RRF over BM25 + dense; pure dense missed precise product
names, pure BM25 missed intent paraphrases.

## What didn't work

- **Pure-LLM grounding** (asking the LLM to pick from a list of 500+
  catalog items in-context). Slow, leaks the prompt budget, and the
  model still occasionally hallucinated short-form names. Replaced
  with explicit retrieval + hydration.
- **Weighted score blending** (alpha * bm25 + (1-alpha) * dense)
  required tuning per trace category. RRF works out of the box at
  comparable quality.
- **Slot-filling state machine** that explicitly tracked
  role/seniority/skills across turns. The stateless spec made this
  awkward to reconstruct from history each call; consolidating into a
  single classifier-LLM-call that re-reads the full history every
  turn was simpler and more robust to out-of-order user replies.

## AI tools used

Drafted the FastAPI skeleton, the scraper structure, and these docs
with Claude (Anthropic). I read and rewrote every file by hand,
particularly the agent dispatch logic and retrieval fusion, and I own
every design choice in the interview.
