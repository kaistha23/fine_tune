"""Allowlisted native training/evaluation entrypoint, never a shell-command runner."""

from __future__ import annotations

import fcntl
import hashlib
import json
import sys
from pathlib import Path

import yaml

from credit_risk.review_store import digest
from credit_risk.tokenization import CHAT_TEMPLATE_MODE, configure_non_thinking
from credit_risk.workbench.contracts import Case, inspect_dataset, messages
from credit_risk.workbench.evaluation import assess_training_target, evaluate
from credit_risk.workbench.store import Store

PROJECT = Path(__file__).resolve().parents[3]

ATTENTION_KEYS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_a",
    "linear_attn.out_proj",
]
MLP_KEYS = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def model_catalog():
    base = Path.home() / ".cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-4bit/snapshots"
    return [
        {"id": p.name, "path": str(p), "label": "Qwen3.5-9B 4-bit / " + p.name[:12]}
        for p in sorted(base.glob("*"))
        if (p / "config.json").is_file()
    ]


def load_cases(spec):
    if spec.get("dataset"):
        fresh = inspect_dataset(spec["dataset"]["path"])
        if fresh["hash"] != spec["dataset"]["hash"]:
            raise ValueError("Dataset manifest changed after registration")
        return [Case.model_validate(c) for c in fresh["cases"]]
    return [Case.model_validate(c) for c in spec.get("cases", [])]


def prepare_training(spec, tokenizer=None, write=False):
    from credit_risk.training import iterations_for

    cases = load_cases(spec)
    version = spec["version"]
    if spec["dataset"]["manifest"].get("format") != "credit-workbench-v2":
        raise ValueError("Phase-2 training requires the credit-workbench-v2 data contract")
    if version["task"] != spec["task"]:
        raise ValueError("Task/version mismatch")
    model = Path(spec["model"]["path"])
    config = json.loads((model / "config.json").read_text())
    quant = config.get("quantization") or config.get("quantization_config") or {}
    if quant.get("bits") != 4:
        raise ValueError("Expected a 4-bit base")
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            model, local_files_only=True, trust_remote_code=False
        )
    configure_non_thinking(tokenizer)
    token_lengths = []
    assistant_tokens = []
    chats = {"train": [], "valid": []}
    for case in cases:
        if case.split not in ("train", "validation"):
            continue
        if case.target is None:
            raise ValueError("Training/validation targets required")
        _metrics, failures, parsed = assess_training_target(case, case.target, version)
        if not parsed or failures:
            raise ValueError("Target failed validation: " + case.case_id)
        turns = messages(case, version) + [
            {"role": "assistant", "content": json.dumps(case.target, ensure_ascii=False)}
        ]
        from credit_risk.tokenization import verify_training_tokens

        full, prefix = verify_training_tokens(tokenizer, turns)
        if len(full) > spec["config"]["max_seq_length"]:
            raise ValueError(
                f"Sequence too long for {case.case_id}: requires {len(full)} tokens; "
                f"configured limit is {spec['config']['max_seq_length']}"
            )
        if len(full) <= len(prefix):
            raise ValueError("Assistant target has no trainable tokens: " + case.case_id)
        token_lengths.append(len(full))
        assistant_tokens.append(len(full) - len(prefix))
        chats["train" if case.split == "train" else "valid"].append({"messages": turns})
    if not all(chats.values()):
        raise ValueError("Nonempty train and validation splits required")
    cfg = {
        **spec["config"],
        "model": str(model),
        "train": True,
        "fine_tune_type": "lora",
        "mask_prompt": True,
        "grad_checkpoint": True,
        "adapter_path": str(
            PROJECT / "adapters/candidates" / spec["task"] / Path(spec["output"]).name
        ),
    }
    target_modules = cfg.pop("target_modules")
    if target_modules == "attention":
        cfg["lora_parameters"]["keys"] = ATTENTION_KEYS
    elif target_modules == "attention_mlp":
        cfg["lora_parameters"]["keys"] = ATTENTION_KEYS + MLP_KEYS
    elif target_modules != "all_linear":
        raise ValueError("Unknown target-module preset")
    cfg["iters"] = iterations_for(
        len(chats["train"]), cfg["batch_size"], cfg["grad_accumulation_steps"], cfg.pop("epochs")
    )
    optimizer_updates = cfg["iters"] // cfg["grad_accumulation_steps"]
    schedule = cfg.pop("schedule")
    warmup_ratio = cfg.pop("warmup_ratio")
    min_lr_ratio = cfg.pop("min_lr_ratio")
    weight_decay = cfg.pop("weight_decay")
    cfg["optimizer_config"] = {
        "adam": {},
        "adamw": {"weight_decay": weight_decay},
        "muon": {},
        "sgd": {},
        "adafactor": {},
    }
    if schedule == "cosine_decay":
        warmup = min(optimizer_updates - 1, round(optimizer_updates * warmup_ratio))
        cfg["lr_schedule"] = {
            "name": "cosine_decay",
            "warmup": max(0, warmup),
            "warmup_init": cfg["learning_rate"] * min_lr_ratio,
            "arguments": [
                cfg["learning_rate"],
                max(1, optimizer_updates - max(0, warmup)),
                cfg["learning_rate"] * min_lr_ratio,
            ],
        }
    else:
        cfg["lr_schedule"] = None
    if Path(cfg["adapter_path"]).exists():
        raise ValueError("Candidate output already exists")
    metadata = {
        "dataset_manifest": spec["dataset"]["hash"],
        "version_id": version["id"],
        "base_snapshot": model.name,
        "model_config_hash": digest(config),
        "chat_template_hash": digest(tokenizer.chat_template),
        "chat_template_mode": CHAT_TEMPLATE_MODE,
        "max_tokens": max(token_lengths),
        "min_assistant_tokens": min(assistant_tokens),
        "token_lengths": token_lengths,
        "micro_batches": cfg["iters"],
        "optimizer_updates": optimizer_updates,
        "quantization": quant,
        "target_module_preset": target_modules,
        "resolved_target_modules": cfg["lora_parameters"].get("keys", "all eligible linear modules"),
        "optimizer": cfg["optimizer"],
        "optimizer_config": cfg["optimizer_config"][cfg["optimizer"]],
        "lr_schedule": cfg["lr_schedule"],
        "task": spec["task"],
        "promotable": False,
    }
    if write:
        data = Path(spec["output"]) / "chat_data"
        data.mkdir()
        for split, rows in chats.items():
            (data / (split + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))
        cfg["data"] = str(data)
        (Path(spec["output"]) / "training.yaml").write_text(yaml.safe_dump(cfg))
    return cfg, metadata


def query_checker(spec):
    from credit_risk import data_service as ds
    from credit_risk.query_guard import GuardedQueryCompiler, SchemaRegistry
    from credit_risk.schemas import QueryPlan

    manifest = spec["dataset"]["manifest"] if spec.get("dataset") else {}
    snapshot = manifest.get("synthetic_snapshot")
    registry_entry = manifest.get("schema_registry")
    registry_path = (
        (Path(spec["dataset"]["path"]).parent / registry_entry["file"])
        if registry_entry
        else PROJECT / "configs/schema_registry.yaml"
    )
    registry = SchemaRegistry(registry_path)
    compiler = GuardedQueryCompiler(registry)

    def check(case, output):
        plan = QueryPlan.model_validate(output)
        compiled = compiler.compile(plan)
        lineage = ds.build_sql_review_packet(compiled)
        lineage["schema_registry_version"] = registry.version
        metrics = {"compilation_success": 1.0}
        if case.expected.get("tables") is not None:
            metrics["table_correctness"] = float(case.expected["tables"] == [compiled.source_table])
        if case.expected.get("columns") is not None:
            metrics["column_correctness"] = float(
                set(case.expected["columns"]) == set(compiled.selected_columns)
            )
        if case.expected.get("joins") is not None:
            metrics["join_correctness"] = float(case.expected["joins"] == compiled.joins)
        if snapshot and "rows" in case.expected:
            if snapshot.get("classification") != "synthetic":
                raise ValueError("Query execution only supports declared synthetic snapshots")
            root = Path(spec["dataset"]["path"]).parent
            db = (root / snapshot["file"]).resolve()
            if not db.is_relative_to(root):
                raise ValueError("Snapshot outside registered dataset")
            with db.open("rb") as f:
                before = hashlib.file_digest(f, "sha256").hexdigest()
            if before != snapshot["sha256"]:
                raise ValueError("Synthetic snapshot changed")
            rows = ds._execute_with_limits(
                compiled, registry.data["query_controls"], database_path=db
            )
            ds.validate_result(rows, compiled, plan, registry.data["query_controls"])
            with db.open("rb") as f:
                after = hashlib.file_digest(f, "sha256").hexdigest()
            if before != after:
                raise ValueError("Snapshot changed during evaluation")
            metrics["result_agreement"] = float(
                sorted(map(digest, rows)) == sorted(map(digest, case.expected["rows"]))
            )
            lineage["snapshot_sha256"] = before
        return metrics, lineage

    return check


def run(spec):
    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    output = Path(spec["output"])
    if spec["kind"] == "train":
        cfg, metadata = prepare_training(spec, write=True)
        from credit_risk.training import execute_training

        execute_training(cfg, metadata)
        (output / "result.json").write_text(
            json.dumps({"adapter_path": cfg["adapter_path"], "metadata": metadata})
        )
        return
    cases = [c for c in load_cases(spec) if c.split in spec["splits"]]
    if not cases:
        raise ValueError("No evaluation cases in selected splits")
    if spec.get("adapter_path"):
        selected = Path(spec["adapter_path"]) / "adapters.safetensors"
        if hashlib.sha256(selected.read_bytes()).hexdigest() != spec["checkpoint_sha256"]:
            raise ValueError("Selected adapter changed after queuing")
    model, tok = load(
        spec["model"]["path"],
        adapter_path=spec.get("adapter_path"),
        tokenizer_config={"trust_remote_code": False},
    )
    configure_non_thinking(tok)

    attempt_by_case = {}

    def provider(case, version):
        attempt = attempt_by_case.get(case.case_id, 0)
        attempt_by_case[case.case_id] = attempt + 1
        seeds = spec["generation"]["seed_sequence"]
        mx.random.seed(seeds[min(attempt, len(seeds) - 1)])
        prompt = tok.apply_chat_template(
            messages(case, version), tokenize=False, add_generation_prompt=True
        )
        tokens = tok.encode(prompt)
        if len(tokens) + spec["generation"]["max_tokens"] > spec["context_limit"]:
            raise ValueError("Evaluation context exceeds budget")
        return generate(
            model,
            tok,
            prompt=prompt,
            max_tokens=spec["generation"]["max_tokens"],
            sampler=make_sampler(
                temp=spec["generation"]["temperature"],
                top_p=spec["generation"]["top_p"],
                top_k=spec["generation"]["top_k"],
            ),
        )

    identity = {
        "model": spec["model"],
        "adapter_path": spec.get("adapter_path"),
        "checkpoint_sha256": spec.get("checkpoint_sha256"),
        "version_id": spec["version"]["id"],
        "schema_hash": digest(spec["version"]["schema"]),
        "prompt_hash": digest(spec["version"]["prompt"]),
        "generation": spec["generation"],
    }
    result = evaluate(
        cases,
        provider,
        spec["version"],
        identity,
        query_checker(spec) if spec["task"] == "query_plan" else None,
    )
    store = Store(spec["workspace"])
    for case, row in zip(cases, result["cases"], strict=True):
        first = row["attempts"][0]
        store.add(
            "answer",
            {
                "case": case.model_dump(),
                "output": first["output"],
                "sql_lineage": first["sql_lineage"],
                "version_id": spec["version"]["id"],
                "schema_hash": digest(spec["version"]["schema"]),
                "prompt_hash": digest(spec["version"]["prompt"]),
                "identity": identity,
                "metrics": row["metrics"],
                "failures": row["failures"],
            },
        )
    (output / "result.json").write_text(json.dumps(result, indent=2))


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    # Survives a dashboard crash: a restarted scheduler cannot overlap an orphan GPU job.
    with (Path(spec["workspace"]) / "gpu.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            run(spec)
        except Exception as exc:
            (Path(spec["output"]) / "failure.json").write_text(
                json.dumps({"error": type(exc).__name__, "detail": str(exc)})
            )
            raise


if __name__ == "__main__":
    main()
