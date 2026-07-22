import json

from perception.facts import load_rule_base
from prompts import build_system_prompt, build_user_message


def test_system_prompt_contains_full_rule_base():
    rule_base = load_rule_base()
    prompt_a = build_system_prompt("A")
    for rule in rule_base:
        assert rule["id"] in prompt_a
        assert rule["criterion"] in prompt_a


def test_system_prompt_scenario_specific_schema_hint():
    prompt_a = build_system_prompt("A")
    prompt_b = build_system_prompt("B")
    assert "placements" in prompt_a
    assert "issues" in prompt_b
    assert "dos" in prompt_b and "donts" in prompt_b


def test_user_message_embeds_facts_json():
    facts = {"scenario": "A", "rule_verdicts": {"solid_back_empty_front": {"triggered": True}}}
    message = build_user_message(facts)
    embedded = json.loads(message.split("Facts JSON:\n", 1)[1])
    assert embedded == facts
