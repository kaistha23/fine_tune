"""Browser smoke with temporary fixtures only; starts no model or training job.
Run: uv tool run --from playwright python scripts/check_workbench_browser.py
"""

import json
import socket
import sqlite3
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

PROJECT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="credit-ui-test-") as temporary:
    root = Path(temporary)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    command = "from pathlib import Path; import uvicorn; import credit_risk.workbench.server as s; s.PROJECT=Path(__import__('sys').argv[1]); fake=Path(__import__('sys').argv[1])/'fake-base'; fake.mkdir(exist_ok=True); s.model_catalog=lambda: [{'id':'fake-base','path':str(fake),'label':'Fake base (no weights)'}]; s.embedding_catalog=lambda: []; uvicorn.run(s.create_app(Path(__import__('sys').argv[1])/'state'),host='127.0.0.1',port=int(__import__('sys').argv[2]))"
    with (root / "server.log").open("w") as log:
        proc = subprocess.Popen(
            [str(PROJECT / ".venv/bin/python"), "-c", command, str(root), str(port)],
            cwd=PROJECT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            base = f"http://127.0.0.1:{port}"
            for _ in range(100):
                try:
                    urllib.request.urlopen(base + "/api/session", timeout=1).close()
                    break
                except OSError:
                    time.sleep(0.1)
            else:
                raise RuntimeError((root / "server.log").read_text())
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page(viewport={"width": 1440, "height": 1000})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(base)
                page.wait_for_function(
                    "() => document.querySelector('#status').textContent.startsWith('Ready')"
                )
                assert "No datasets registered" in page.locator("#datasetList").inner_text()
                # Source data: initialize load 1, validate and append a synthetic month as load 2.
                # The Playwright interpreter cannot import the project; use its virtualenv.
                prep = [str(PROJECT / ".venv/bin/python"), "-m", "credit_risk.data_prep.cli", "fixture"]
                curated = root / "data/curated/credit_risk.duckdb"
                curated.parent.mkdir(parents=True)
                small = ["--obligors", "4", "--seed", "17"]
                subprocess.run([*prep, "--out", str(curated), *small], cwd=PROJECT, check=True, capture_output=True)
                page.wait_for_function("() => !document.querySelector('#initSources').hidden")
                page.locator("#initSources").click()
                page.wait_for_function(
                    "() => document.querySelector('#sourceStatus').textContent.startsWith('Watermark load 1')"
                )
                subprocess.run(
                    [*prep, *small, "--month", "2026-01", "--out-dir", str(root / "incoming")],
                    cwd=PROJECT, check=True, capture_output=True,
                )
                month = root / "incoming/obligor_monthly-2026-01.parquet"
                page.locator("#sourceTable").select_option("obligor_monthly")
                page.locator("#sourceFile").set_input_files(str(month))
                page.locator("#validateSource").click()
                page.wait_for_function("() => !document.querySelector('#appendSource').disabled")
                page.locator("#appendSource").click()
                page.wait_for_function(
                    "() => document.querySelector('#sourceStatus').textContent.startsWith('Watermark load 2')"
                )
                assert "obligor_monthly-2026-01.parquet" in page.locator("#sourceLedger").inner_text()
                # Documents: register a policy (indexing is not started, so no model loads).
                policy = root / "policy.md"
                policy.write_text(
                    "# Monitoring\n\n7.2 Enhanced monitoring\n\nStage 2 obligors are reviewed every quarter.\n"
                )
                page.locator("#documentFile").set_input_files(str(policy))
                page.locator("#documentId").fill("SAMA-MONITORING")
                page.locator("#documentVersion").fill("1.0")
                page.locator("#documentFrom").fill("2025-01-01")
                page.locator("#registerDocument").click()
                page.wait_for_function(
                    "() => document.querySelector('#documentList').textContent.includes('SAMA-MONITORING@1.0')"
                )
                assert "registered" in page.locator("#documentReport").inner_text()
                # Ask: template plan → run → the fake base has no weights, so the step fails
                # visibly instead of loading a model.
                page.get_by_role("button", name="Ask", exact=True).click()
                page.locator("#askQuestion").fill("What is the current credit stage of OBL-0002?")
                page.locator("#askAsOf").fill("2026-01-15")
                page.locator("#askManual").click()
                page.wait_for_function("() => !document.querySelector('#askPlanPanel').hidden")
                plan = json.loads(page.locator("#askPlan").input_value())
                assert plan["obligor_id"] == "OBL-0002" and plan["as_of_date"] == "2026-01-15"
                page.locator("#askRun").click()
                page.wait_for_function(
                    "() => document.querySelector('#askStatus').textContent.startsWith('Failed')",
                    timeout=60000,
                )
                assert "OBL-0002" in page.locator("#askHistory").inner_text()
                page.screenshot(path=str(PROJECT / "outputs/workbench-browser/ask.png"), full_page=True)
                page.get_by_role("button", name="Datasets", exact=True).click()
                for title, section in [
                    ("Runs", "runs"),
                    ("Evaluation", "evaluation"),
                    ("Answers & feedback", "answers"),
                ]:
                    page.get_by_role("button", name=title, exact=True).click()
                    assert page.locator("#" + section).is_visible()
                state = page.request.get(base + "/api/state").json()
                version = state["active"]["credit_analysis"]
                sample = {
                    "case": {
                        "case_id": "UI-ONLY",
                        "group_id": "UI-GROUP",
                        "task": "credit_analysis",
                        "split": "development",
                        "question": "Are inputs sufficient?",
                        "portfolio": "retail",
                        "jurisdiction": "SAMA",
                        "as_of_date": "2025-01-01",
                        "facts": {},
                        "evidence": [],
                        "expected": {"fields": {"answer_status": "INSUFFICIENT_EVIDENCE"}},
                        "consistency_paths": ["answer_status"],
                        "provenance": {"classification": "synthetic"},
                    },
                    "output": {"answer_status": "ANSWERED", "executive_summary": ""},
                    "version_id": version,
                    "sql_lineage": {
                        "parameterised_sql": "SELECT value FROM synthetic WHERE id=?",
                        "parameter_values": ["private-fixture"],
                    },
                }
                page.get_by_text("Import a captured local answer", exact=True).click()
                page.locator("#importPayload").fill(json.dumps(sample))
                page.locator("#importAnswer").click()
                page.wait_for_function(
                    "() => document.querySelector('#answerSelect').options.length===2"
                )
                assert page.locator("#answerSelect option").nth(1).inner_text().startswith("imported")
                page.locator("#answerSelect").select_option(index=1)
                page.wait_for_function(
                    "() => document.querySelector('#answerInput').textContent.includes('UI-ONLY')"
                )
                assert "private-fixture" not in page.locator("#sqlView").inner_text()
                page.locator("#feedbackComment").fill("Wrong answer; insufficient inputs.")
                page.locator("#submitFeedback").click()
                page.wait_for_function(
                    "() => document.querySelector('#feedbackResult').textContent.includes('needs_expected_results')"
                )
                page.locator("#correction").fill(
                    json.dumps({"answer_status": "INSUFFICIENT_EVIDENCE", "executive_summary": ""})
                )
                page.locator("#cause").select_option("model_behaviour")
                page.locator("#submitFeedback").click()
                page.wait_for_function(
                    "() => document.querySelector('#feedbackResult').textContent.includes('eligible_for_training\": true')"
                )
                page.locator("#exportBatch").click()
                page.wait_for_function(
                    "() => document.querySelector('#feedbackResult').textContent.includes('eligible_cases\": 1')"
                )
                page.locator("#loadVersion").click()
                page.locator("#versionName").fill("Browser fixture version")
                page.locator("#promptEdit").fill(
                    "Use supplied facts and output schema. Identify missing information."
                )
                page.locator("#saveVersion").click()
                page.wait_for_function(
                    "() => document.querySelector('#editVersion').options.length===3"
                )
                after = page.request.get(base + "/api/state").json()
                assert after["active"]["credit_analysis"] == version
                # Feedback and version edits never start training or evaluation.
                assert not [j for j in after["jobs"] if j["spec"]["kind"] in ("train", "evaluate", "regression")]
                # A completed artifact fixture exercises separate scorecards and loss rendering.
                output = root / "fixture-run"
                output.mkdir()
                report = {
                    "identity": {"version_id": version, "generation": {}},
                    "case_set_hash": "fixture",
                    "evaluator_version": "field-checks-v2",
                    "splits": {
                        s: {"json_validity": {"value": v, "denominator": 2}}
                        for s, v in [("validation", 1), ("test", 0.5), ("oot", 0)]
                    },
                    "cases": [],
                }
                (output / "result.json").write_text(json.dumps(report))
                job = {
                    "id": "fixture-run",
                    "created_at": "2026-01-01",
                    "status": "completed",
                    "spec": {"output": str(output), "task": "credit_analysis", "kind": "evaluate"},
                }
                with sqlite3.connect(root / "state/workbench.sqlite3") as con:
                    con.execute(
                        "INSERT INTO items VALUES (?,?,?)", ("job", "fixture-run", json.dumps(job))
                    )
                page.reload()
                page.wait_for_function(
                    "() => document.querySelector('#evalRun').options.length===2"
                )
                page.get_by_role("button", name="Evaluation", exact=True).click()
                page.get_by_text("Advanced historical comparison").click()
                page.locator("#evalRun").select_option("fixture-run")
                page.locator("#loadEvaluation").click()
                page.wait_for_function(
                    "() => document.querySelector('#scorecards').textContent.includes('50.0%')"
                )
                assert "oot" in page.locator("#scorecards").inner_text()
                assert "Not evaluated" in page.locator("#releaseGates").inner_text()
                assert "standalone evaluation" in page.locator("#comparisonProgressText").inner_text()
                screenshots = PROJECT / "outputs/workbench-browser"
                screenshots.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(screenshots / "evaluation.png"), full_page=True)
                page.get_by_role("button", name="Runs", exact=True).click()
                page.evaluate(
                    "() => draw([{train:{iteration:8,train_loss:2}},{validation:{iteration:7,val_loss:2.2}},{train:{iteration:16,train_loss:1.7}},{validation:{iteration:15,val_loss:1.9}}])"
                )
                page.screenshot(path=str(screenshots / "runs.png"), full_page=True)
                page.set_viewport_size({"width": 390, "height": 844})
                assert page.evaluate("() => document.documentElement.scrollWidth <= innerWidth")
                assert not errors, errors
                browser.close()
            print(
                "Browser passed: four views, direct feedback, masking, versions, split scorecards, charts, mobile layout; zero JS errors. No models loaded."
            )
        finally:
            proc.terminate()
            proc.wait(timeout=10)
