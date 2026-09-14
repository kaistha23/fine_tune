from credit_risk.workbench import memory


def test_question_normalisation_and_keys():
    assert memory.normalise_question("  What is  obl 65 stage?? ") == "what is OBL-0065 stage"
    assert memory.normalise_question("WHAT IS OBL-0065 STAGE") == "what is OBL-0065 stage"
    context = memory.context_key({"plan": {"obligor_id": "OBL-0065"}, "snapshot": {"load_id": 1}})
    same = memory.answer_key(context, "what is obl-65 stage?")
    assert same == memory.answer_key(context, "What is OBL 0065 stage")
    assert same != memory.answer_key(context, "What is OBL-0066 stage")
    later = memory.context_key({"plan": {"obligor_id": "OBL-0065"}, "snapshot": {"load_id": 2}})
    assert memory.answer_key(later, "what is obl-65 stage") != same
    plan = {"obligor_id": "OBL-0065"}
    assert memory.question_key("Stage of OBL-65?", plan) == memory.question_key("stage of obl 0065", plan)


def test_consistency_fields_ignore_prose_and_detect_decisions():
    base = {
        "answer_status": "ANSWERED",
        "executive_summary": "The obligor is in Stage 2.",
        "facts": [{"statement": "current_position.stage = 2", "evidence_ids": ["CASE-1"]}],
        "risk_drivers": ["Utilisation  rising"],
    }
    reworded = {**base, "executive_summary": "Stage two applies to this borrower."}
    assert memory.consistency_fields(base) == memory.consistency_fields(reworded)
    fields = memory.consistency_fields(base)
    assert fields["stage"] == 2 and fields["risk_drivers"] == ["utilisation rising"]
    moved = {**base, "facts": [{"statement": "Stage 3 recorded", "evidence_ids": ["CASE-1"]}]}
    assert memory.field_diff(fields, memory.consistency_fields(moved)) == {"stage": {"before": 2, "after": 3}}
    assert memory.consistency_fields("not json") is None
    real = {
        "answer_status": "PARTIAL",
        "facts": [
            {"statement": "A stage migration from 1 to 2 was observed in the window."},
            {"statement": "The current credit stage for OBL-0002 is 2."},
        ],
    }
    assert memory.consistency_fields(real)["stage"] == 2
    assert memory.context_changes({"snapshot": 1, "model": "a"}, {"snapshot": 2, "model": "a"}) == [
        "data snapshot (source loads)"
    ]


def test_lookup_prefers_verified_and_never_reuses_unstable(tmp_path):
    from credit_risk.workbench.store import Store

    store = Store(tmp_path)
    keys = {"answer_key": "a", "context_key": "c", "question_key": "q"}
    common = {"keys": keys, "fields": {}, "context_summary": {}, "question": "q"}
    memory.remember(store, answer_id="unstable", status="unstable", **common)
    assert memory.lookup(store, "a") is None
    memory.remember(store, answer_id="model", status="model", **common)
    assert memory.lookup(store, "a")["answer_id"] == "model"
    memory.remember(store, answer_id="human", status="verified", **common)
    assert memory.lookup(store, "a")["answer_id"] == "human"
    memory.remember(store, answer_id="human", status="superseded", **common)
    assert memory.lookup(store, "a")["answer_id"] == "model"
