"""Unit tests that don't require the LLM."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.catalog import Catalog, Item, _parse_minutes
from app.agent import _extract_compare_names
from app.schemas import ChatResponse, Recommendation


# ---------- schema ----------

def test_recommendation_schema_shape():
    r = Recommendation(name="OPQ32r", url="https://www.shl.com/x", test_type="P")
    assert r.model_dump() == {"name": "OPQ32r", "url": "https://www.shl.com/x", "test_type": "P"}


def test_chat_response_empty_recs_ok():
    r = ChatResponse(reply="hi", recommendations=[], end_of_conversation=False)
    assert r.model_dump()["recommendations"] == []


def test_chat_response_max_10():
    recs = [Recommendation(name=f"x{i}", url="https://x", test_type="K") for i in range(11)]
    with pytest.raises(Exception):
        ChatResponse(reply="...", recommendations=recs, end_of_conversation=False)


# ---------- helpers ----------

def test_parse_minutes_variants():
    assert _parse_minutes("Approximate Completion Time in minutes = 35") == 35
    assert _parse_minutes("about 25 mins") == 25
    assert _parse_minutes("") is None
    assert _parse_minutes("no time info") is None


def test_extract_compare_names_diff():
    assert _extract_compare_names("What is the difference between OPQ and GSA?") == ["OPQ", "GSA"]


def test_extract_compare_names_vs():
    assert _extract_compare_names("OPQ32r vs Verify G+") == ["OPQ32r", "Verify G+"]


def test_extract_compare_names_compare_to():
    out = _extract_compare_names("Compare OPQ to GSA please.")
    assert out == ["OPQ", "GSA"]


# ---------- catalog ----------

@pytest.fixture
def tiny_catalog(tmp_path: Path) -> Catalog:
    data = {
        "source": "test",
        "scope": "Individual Test Solutions",
        "test_type_legend": {"K": "Knowledge", "P": "Personality"},
        "count": 3,
        "items": [
            {
                "name": "Java 8 (New)",
                "url": "https://www.shl.com/java8",
                "test_types": ["K"],
                "description": "Multiple-choice Java programming knowledge test.",
                "job_levels": ["Mid", "Senior"],
                "languages": ["English"],
                "assessment_length": "Approximate Completion Time in minutes = 30",
                "remote_testing": True,
                "adaptive_irt": False,
            },
            {
                "name": "OPQ32r",
                "url": "https://www.shl.com/opq32r",
                "test_types": ["P"],
                "description": "Occupational Personality Questionnaire for workplace behavior.",
                "job_levels": ["All"],
                "languages": ["English"],
                "assessment_length": "25 minutes",
                "remote_testing": True,
                "adaptive_irt": False,
            },
            {
                "name": "Python (New)",
                "url": "https://www.shl.com/python",
                "test_types": ["K"],
                "description": "Multiple-choice Python programming knowledge test.",
                "job_levels": ["Mid"],
                "languages": ["English"],
                "assessment_length": "30 minutes",
                "remote_testing": True,
                "adaptive_irt": False,
            },
        ],
    }
    p = tmp_path / "catalog.json"
    p.write_text(json.dumps(data))
    return Catalog.from_json(p)


def test_catalog_retrieve_finds_relevant(tiny_catalog):
    hits = tiny_catalog.retrieve("hiring a java developer", k=3)
    names = [it.name for it, _ in hits]
    assert "Java 8 (New)" in names
    # Java should rank above Python for a Java query.
    assert names.index("Java 8 (New)") < names.index("Python (New)")


def test_catalog_retrieve_personality(tiny_catalog):
    hits = tiny_catalog.retrieve("personality test workplace behavior", k=3)
    names = [it.name for it, _ in hits]
    assert names[0] == "OPQ32r"


def test_catalog_filter_by_type(tiny_catalog):
    hits = tiny_catalog.filter_retrieve(
        "java developer with personality assessment",
        k=5,
        required_types=["P"],
    )
    names = [it.name for it, _ in hits]
    assert names == ["OPQ32r"]


def test_find_by_name_exact(tiny_catalog):
    hits = tiny_catalog.find_by_name("OPQ32r")
    assert hits[0].name == "OPQ32r"


def test_find_by_name_substring(tiny_catalog):
    hits = tiny_catalog.find_by_name("opq")
    assert hits[0].name == "OPQ32r"
