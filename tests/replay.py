"""
Local replay harness for conversation traces.

Each trace is a JSON file:
{
  "name": "java_mid_level",
  "persona": "Hiring manager for a mid-level Java developer.",
  "facts": {
    "role": "Java developer",
    "seniority": "mid-level, ~4 years",
    "soft_skills": "works with stakeholders",
    "time_budget": null
  },
  "expected_assessment_names": ["Java 8 (New)", "OPQ32r", ...],
  "behavior_probes": [
    {"name": "no_recommend_turn_1", "assertion": "first_assistant_no_recommendations"},
    {"name": "stays_on_topic", "user_says": "What's the weather?", "assertion": "refuses_off_topic"}
  ]
}

Usage:
    python tests/replay.py data/traces/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

# Make sibling app/ importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import Agent, MAX_TURNS
from app.catalog import Catalog
from app.llm import LLMClient
from app.schemas import Message


SIM_USER_SYSTEM = """You are simulating a hiring manager talking to an SHL assessment
recommender. You will be given a persona and a set of facts. Rules:
- Answer the assistant's questions truthfully using your facts.
- If asked about something not in your facts, say "no preference" or "doesn't matter".
- Keep replies short, one or two sentences.
- When the assistant gives you a shortlist of assessments, reply with a brief thanks
  and end the conversation (your output: END).
Output ONLY your next user reply, no role prefix."""


@dataclass
class TraceResult:
    name: str
    turns_used: int
    final_recommendations: list[str]  # names
    recall_at_10: float
    probe_results: dict[str, bool]
    transcript: list[dict]


def simulate_user_reply(llm: LLMClient, persona: str, facts: dict, transcript: list[Message]) -> str:
    convo = "\n".join(f"{m.role}: {m.content}" for m in transcript)
    user_msg = (
        f"Persona: {persona}\n"
        f"Facts: {json.dumps(facts, ensure_ascii=False)}\n\n"
        f"Conversation so far:\n{convo}\n\n"
        "Your next user reply:"
    )
    return llm.chat(
        [
            {"role": "system", "content": SIM_USER_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.4,
        max_tokens=120,
    ).strip()


def run_trace(agent: Agent, sim_llm: LLMClient, trace: dict) -> TraceResult:
    transcript: list[Message] = []
    final_recs: list[str] = []

    # The first user message is either the trace's opening or a synthesized one.
    opener = trace.get("opening_message") or _synth_opener(trace)
    transcript.append(Message(role="user", content=opener))

    for _ in range(MAX_TURNS):
        if len(transcript) >= MAX_TURNS:
            break
        # Assistant turn
        resp = agent.handle(transcript)
        transcript.append(Message(role="assistant", content=resp.reply))
        if resp.recommendations:
            final_recs = [r.name for r in resp.recommendations]
        if resp.end_of_conversation or final_recs:
            break

        if len(transcript) >= MAX_TURNS:
            break
        # User turn
        next_user = simulate_user_reply(sim_llm, trace["persona"], trace["facts"], transcript)
        if next_user.upper().strip() == "END":
            break
        transcript.append(Message(role="user", content=next_user))

    expected = {n.lower() for n in trace.get("expected_assessment_names", [])}
    recommended_lower = {n.lower() for n in final_recs}
    if expected:
        hits = len(expected & recommended_lower)
        recall = hits / len(expected)
    else:
        recall = 0.0

    probe_results = _run_probes(agent, trace.get("behavior_probes", []))

    return TraceResult(
        name=trace.get("name", "anon"),
        turns_used=len(transcript),
        final_recommendations=final_recs,
        recall_at_10=recall,
        probe_results=probe_results,
        transcript=[{"role": m.role, "content": m.content} for m in transcript],
    )


def _synth_opener(trace: dict) -> str:
    f = trace["facts"]
    bits = []
    if f.get("role"):
        bits.append(f"I'm hiring a {f['role']}.")
    if f.get("seniority"):
        bits.append(f"Seniority: {f['seniority']}.")
    return " ".join(bits) or "I need to hire someone."


def _run_probes(agent: Agent, probes: list[dict]) -> dict[str, bool]:
    results: dict[str, bool] = {}
    for probe in probes:
        name = probe["name"]
        assertion = probe["assertion"]
        if assertion == "first_assistant_no_recommendations":
            r = agent.handle([Message(role="user", content="I need an assessment.")])
            results[name] = len(r.recommendations) == 0
        elif assertion == "refuses_off_topic":
            text = probe.get("user_says", "What's the weather today?")
            r = agent.handle([Message(role="user", content=text)])
            # Off-topic refusal = empty recommendations + reply mentions SHL/assessment scope.
            results[name] = (
                len(r.recommendations) == 0
                and any(k in r.reply.lower() for k in ["shl", "assessment", "only help"])
            )
        elif assertion == "honors_refinement":
            msgs = [
                Message(role="user", content="Hiring a mid-level Java developer."),
                Message(role="assistant", content="Here are some options."),
                Message(role="user", content="Add a personality test to the mix."),
            ]
            r = agent.handle(msgs)
            results[name] = any("P" in rec.test_type for rec in r.recommendations)
        else:
            results[name] = False  # unknown probe -> conservative fail
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces_dir", help="Directory of trace JSON files.")
    ap.add_argument("--catalog", default="data/catalog.json")
    args = ap.parse_args()

    catalog = Catalog.from_json(args.catalog)
    llm = LLMClient()
    agent = Agent(catalog=catalog, llm=llm)
    sim_llm = LLMClient()  # same client; could be a different model

    traces_dir = Path(args.traces_dir)
    trace_files = sorted(traces_dir.glob("*.json"))
    if not trace_files:
        print(f"No trace files in {traces_dir}", file=sys.stderr)
        sys.exit(2)

    results = []
    for tf in trace_files:
        trace = json.loads(tf.read_text())
        print(f"--- {trace.get('name', tf.stem)} ---")
        res = run_trace(agent, sim_llm, trace)
        results.append(res)
        print(f"  turns: {res.turns_used}")
        print(f"  recall@10: {res.recall_at_10:.2f}")
        print(f"  probes: {res.probe_results}")
        print(f"  recommended: {res.final_recommendations[:5]}{'...' if len(res.final_recommendations) > 5 else ''}")

    mean_recall = sum(r.recall_at_10 for r in results) / len(results)
    all_probes = {k: v for r in results for k, v in r.probe_results.items()}
    probe_pass = sum(1 for v in all_probes.values() if v) / max(len(all_probes), 1)

    print("\n=== SUMMARY ===")
    print(f"Traces: {len(results)}")
    print(f"Mean Recall@10: {mean_recall:.3f}")
    print(f"Probe pass rate: {probe_pass:.3f}")


if __name__ == "__main__":
    main()
