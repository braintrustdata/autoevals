import json

import pytest

from autoevals.llm import AgentBehavior, Behavior, discover_agent_behaviors


def write_behavior(project_root, name="verify-work"):
    directory = project_root / ".agents" / "behaviors" / name
    directory.mkdir(parents=True)
    behavior_file = directory / "BEHAVIOR.md"
    behavior_file.write_text(
        f"---\nname: {name}\ndescription: Verify work before answering.\n---\n"
        "# Verify work\n\nThe agent MUST show its calculation.\n"
    )
    return behavior_file


def test_discovers_behavior_from_project_root(tmp_path):
    behavior_file = write_behavior(tmp_path)

    behaviors = discover_agent_behaviors(tmp_path)

    assert len(behaviors) == 1
    assert behaviors[0].name == "verify-work"
    assert behaviors[0].location == str(behavior_file.resolve())


def test_behavior_builds_judge_prompt_and_processes_na():
    scorer = Behavior(
        behavior=AgentBehavior(
            name="verify-work",
            description="Verify work before answering.",
            body="# Verify work\n\nThe agent MUST show its calculation.",
        ),
        model="gpt-4o-mini",
    )

    request = scorer._request_args(
        output={"events": [{"type": "answer", "content": "2 + 2 = 4"}]},
        expected=None,
        input={"question": "What is 2 + 2?"},
    )
    prompt = request["messages"][0]["content"]
    assert "The agent MUST show its calculation." in prompt
    assert "What is 2 + 2?" in prompt

    score = scorer._process_response(
        {
            "tool_calls": [
                {
                    "function": {
                        "name": "select_choice",
                        "arguments": json.dumps({"choice": "na", "reasons": "There is not enough evidence."}),
                    }
                }
            ]
        }
    )
    assert score.score is None
    assert score.metadata["choice"] == "na"
    assert score.metadata["behavior"]["name"] == "verify-work"


def test_behavior_auto_discovers_one_spec(tmp_path):
    write_behavior(tmp_path)

    scorer = Behavior(behavior_root=tmp_path, model="gpt-4o-mini")

    assert scorer.behavior.name == "verify-work"


def test_behavior_name_ignores_unrelated_project_path(tmp_path):
    write_behavior(tmp_path)
    (tmp_path / "verify-work").write_text("unrelated project file")

    scorer = Behavior(behavior="verify-work", behavior_root=tmp_path)

    assert scorer.behavior.name == "verify-work"


def test_behavior_discovery_reports_invalid_spec(tmp_path):
    behavior_file = write_behavior(tmp_path)
    behavior_file.write_text("---\nname: INVALID\ndescription: Invalid behavior.\n---\n# Invalid\n")

    with pytest.raises(ValueError, match="Diagnostics: Agent Behavior name"):
        Behavior(behavior_root=tmp_path)


def test_behavior_requires_explicit_selection_for_multiple_specs(tmp_path):
    write_behavior(tmp_path, "first-behavior")
    write_behavior(tmp_path, "second-behavior")

    with pytest.raises(ValueError, match="Multiple Agent Behavior specs were discovered"):
        Behavior(behavior_root=tmp_path)
