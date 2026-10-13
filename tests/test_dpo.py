import pytest

from distillkit.dpo import mine_pairs, prose_rows, rejection_reason

VALID = {"time", "nodes", "ntasks", "output", "error", "mem"}
SYS = "sys"


def answer(*flags):
    return "```bash\n#!/bin/bash\n" + "".join(f"#SBATCH --{f}=1\n" for f in flags) + "srun ./a\n```"


def row(rid, chosen, q="q?"):
    return {"id": rid, "messages": [{"role": "user", "content": q}, {"role": "assistant", "content": chosen}]}


def test_invented_flag_is_rejected():
    assert rejection_reason(answer("time"), answer("time", "gpu-magic"), VALID) == "bad_flags"


def test_missing_half_of_teacher_flags_is_rejected():
    assert rejection_reason(answer("time", "nodes"), answer("time"), VALID) == "missed_flags"


def test_good_sample_is_not_rejected():
    assert rejection_reason(answer("time", "nodes", "mem"), answer("time", "nodes", "mem"), VALID) is None


def test_missing_rule_needs_two_teacher_flags():
    assert rejection_reason(answer("time"), "no script", VALID) is None


def test_flag_names_have_no_dashes():
    assert "time" in VALID and "--time" not in VALID  # slurm_flags.txt stores names without "--"
    assert rejection_reason(answer("time", "nodes"), answer("time", "nodes"), VALID) is None


def test_mine_pairs_skips_bad_chosen_and_takes_first_rejected():
    rows = [row("a", answer("time", "nodes")), row("b", answer("bogus")), row("c", answer("mem"))]
    samples = [[answer("time", "nodes"), answer("time")], [answer("time")], [answer("mem")]]
    pairs, why = mine_pairs(rows, samples, SYS, VALID)
    assert [p["id"] for p in pairs] == ["a"]
    assert pairs[0]["rejected"][0]["content"] == answer("time")
    assert pairs[0]["prompt"] == [{"role": "system", "content": SYS}, {"role": "user", "content": "q?"}]
    assert why == {"bad_flags": 0, "missed_flags": 1, "skipped_bad_chosen": 1}


def test_prose_rows_drops_tool_and_multiturn():
    tool = {"id": "t", "messages": [{"role": "user", "content": "q"},
                                    {"role": "assistant", "content": "", "tool_calls": [{"name": "x"}]}]}
    multi = {"id": "m", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"},
                                     {"role": "user", "content": "q2"}]}
    assert [r["id"] for r in prose_rows([row("p", "a"), tool, multi])] == ["p"]


def test_scale_state_interpolates_only_changed_tensors():
    torch = pytest.importorskip("torch")
    from distillkit.merge import scale_state

    w0 = {"a": torch.zeros(2), "b": torch.ones(2)}
    wm = {"a": torch.full((2,), 10.0), "b": torch.ones(2)}
    out = scale_state(w0, wm, 0.2)
    assert set(out) == {"a"} and torch.allclose(out["a"], torch.full((2,), 2.0))
