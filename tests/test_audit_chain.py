"""The audit log must be tamper-evident. Pure-Python tests (no database needed)."""
from common.audit import GENESIS, build_record, canonical, compute_hash, verify_chain


def make_chain(n=5):
    rows, prev = [], GENESIS
    for i in range(n):
        rec = build_record(actor="dr_demo", action="decision_approve", study_uid=f"1.2.840.{i}",
                           model_version="densenet121-abc", ai_output={"p": 0.5 + i / 100, "label": "Edema"},
                           reason="looks right", ts=f"2026-11-0{i + 1}T00:00:00.000000+00:00")
        h = compute_hash(prev, rec)
        rows.append({**rec, "id": i + 1, "prev_hash": prev, "hash": h})
        prev = h
    return rows


def test_intact_chain_verifies():
    res = verify_chain(make_chain())
    assert res["ok"] and res["checked"] == 5 and res["first_bad_id"] is None


def test_empty_chain_is_ok():
    assert verify_chain([])["ok"]


def test_editing_a_field_is_detected():
    rows = make_chain()
    rows[2]["actor"] = "mallory"
    res = verify_chain(rows)
    assert not res["ok"] and res["first_bad_id"] == 3


def test_editing_ai_output_is_detected():
    rows = make_chain()
    rows[1]["ai_output"] = canonical({"p": 0.99, "label": "Edema"})
    assert verify_chain(rows)["first_bad_id"] == 2


def test_deleting_a_middle_row_is_detected():
    rows = make_chain()
    del rows[2]
    assert verify_chain(rows)["first_bad_id"] == 4


def test_swapping_rows_is_detected():
    rows = make_chain()
    rows[1], rows[2] = rows[2], rows[1]
    assert not verify_chain(rows)["ok"]


def test_tail_truncation_needs_an_external_head_hash():
    """Documented limitation: dropping the LAST rows still verifies, but the head hash changes."""
    rows = make_chain()
    full_head = verify_chain(rows)["head_hash"]
    res = verify_chain(rows[:-1])
    assert res["ok"] and res["head_hash"] != full_head


def test_hash_does_not_depend_on_key_order():
    a = build_record(actor="x", action="y", ai_output={"a": 1, "b": 2}, ts="t")
    b = build_record(actor="x", action="y", ai_output={"b": 2, "a": 1}, ts="t")
    assert compute_hash(GENESIS, a) == compute_hash(GENESIS, b)
