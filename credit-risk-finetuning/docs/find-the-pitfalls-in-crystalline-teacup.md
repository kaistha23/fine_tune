# Plan — data prep for `credit-risk-finetuning`

## Context

The audit's pitfalls were mostly fixed in the current working tree on
`data_prep_code_updates`. Verified on that tree: **441 passed, 35 skipped,
`ruff check .` clean**, CI and pre-commit added. What is still missing is on the **data**
side: the model is trained and gated on 8 gold cases, thresholds live only in policy prose,
and no reviewed phase-2 dataset reaches the coverage gate. Governed units and consolidated
data generation are complete; taxonomy and diversity controls precede deterministic
derivations and policy rules.

The dedicated data-prep package now owns the fixture, gold, and spike generators. The remaining
phases build four things into it:
the tabular-LLM **task taxonomy as a coverage checklist**, **clone/diversity controls**,
**deterministic derivations**, and **hand-authored policy rules** evaluated before the model
answers. Extraction of rules from policy by LLM is deferred.

## Pitfall status (verified on current tree)

| Finding | Status |
|---|---|
| Untrained "best" checkpoint | Fixed — baseline required, best needs post-update improvement (`training.py:198-212`) |
| Abstention gate auto-fail on empty slice | Fixed — `abstention_recall: float \| None` |
| One predicate as filter + gate + metric | Fixed — `is_admissible_training_target`, `judge.py`, metric renamed `extractive_support_rate` |
| Eval stricter than prod | Fixed — `metrics.py:130` passes factsheet |
| Negative denominators / sign inversion | Fixed — `denominator <= 0` invalid, `previous <= 0` → `None` |
| Retrieval AND + truncate-then-filter | Fixed — both arms generate, one cosine gate, then top-k |
| Chunk-id collision | Fixed — full heading path (`ingest.py:100`) |
| Placeholder embedder shipped | Fixed — `CR_EMBEDDING_MODEL:?` required; `extra` forbids unknown keys |
| `promote` ignores gates | Fixed — reads `passed` (`runner.py:92`); Wilson bound, `MIN_ELIGIBLE = 30` |
| No CI | Fixed — `.github/workflows/credit-risk.yml` (untracked) |
| **Work uncommitted** (41 modified, 17 untracked) | **Fixed** — foundation landed in `c2d3ed2` |
| **Gold set = 8 cases** vs `MIN_ELIGIBLE = 30` per gated slice | **Open** — every gate reports insufficient |
| **No `unit` on `MetricValue`** | **Fixed** — required units, registry 1.4.0 |
| **Seed dataset has no template-family leakage check or cap** | **Fixed** — shared family/clone cap and split check |
| Workbench token via `GET /api/session`; owner-bound approval; self-certifying workbench feedback; open `/docs`; no SQLite migrations | Open — out of scope here, listed for tracking |

---

## Design

### 1. Separate data-prep folder

New package **`credit-risk-finetuning/src/credit_risk/data_prep/`**. A package under `src/`
rather than a top-level folder so it is importable by tests, packaged by hatch
(`packages = ["src/credit_risk"]`), and exposed as a CLI without `sys.path` hacks.

**Boundary:** `data_prep` *authors and measures* candidate payloads. `dataset.py`,
`feedback.py` and `guardrails.is_admissible_training_target` stay where they are and remain
the *admission gate*. Generators never admit their own output.

```
src/credit_risk/data_prep/
  __init__.py
  cli.py            # credit-risk-data-prep {fixture,gold,spike,coverage,diversity,rules}
  fixture.py        # synthetic DuckDB fixture
  gold.py           # frozen mechanics gold set
  spike.py          # mechanics training spike
  taxonomy.py       # task types + situations (§2)
  coverage.py       # coverage matrix report (§2)
  diversity.py      # clone detection, family caps (§3)
  derivation.py     # check_derivation (§4)
  rules.py          # PolicyRuleRegistry + evaluate_rules (§5)
configs/data_prep/
  coverage_targets.yaml   # minimum cases per cell
configs/policy_rules.yaml # hand-authored registry (§5), beside the other registries
```

Fixture callers use `python -m credit_risk.data_prep.fixture` or the unified CLI. No legacy
script shims remain.

### 2. Task taxonomy as a coverage checklist

Map each tabular-LLM task family to a credit task. Every training and gold payload carries
`task_type` and `situation`; the coverage report counts cells.

| Taxonomy | Credit task (`task_type`) | Example | Status |
|---|---|---|---|
| Table QA | `factsheet_qa` | "What is current DSCR?" → value + `case_id` citation | new |
| Table-to-Text | `factsheet`, `email_draft` | summary of position and trends | exists |
| Table Fact Verification | `claim_verification` | claim + factsheet + evidence → supported / refuted / insufficient | **new, highest value** |
| NL2SQL | `query_plan` | request → structured `QueryPlan` (never SQL) | exists (workbench) |
| Tabular Math Reasoning | `metric_interpretation` | cite governed metric + derivation; refuse to compute | new |
| Table Interpretation | `field_interpretation` | pp vs %, TTC vs PIT PD, units, grain | new |
| TAT-QA (table + text) | `ews_analysis`, `credit_deterioration`, `policy_qa` | factsheet + policy evidence + rule outcome | exists |

**Situations** (cross-cutting axis): `base`, `near_miss`, `mitigant`, `grain_trap`,
`superseded_policy`, `wrong_jurisdiction`, `missing_field`, `conflicting_evidence`,
`injection`.

`coverage.py` reports task × portfolio × jurisdiction × situation counts against
`coverage_targets.yaml`. Gold targets are driven by the existing gate: **≥ 30 eligible
cases per gated portfolio and task slice** (`gates.py` `MIN_ELIGIBLE = 30`), otherwise gates
stay insufficient by design.

### 3. Template clones vs. diversity

A template clone varies only digits ("Utilisation rose from 62% to 88%, above 85%" × 800)
and teaches one rule. Diversity varies the reasoning for the same concept: base breach,
limit cut with flat drawings, 84.6% near-miss, seasonal mitigant, facility-vs-obligor grain
trap, superseded trigger, missing `facility_limit`, stage 3 dominating.

Controls in `diversity.py`, applied to seed payloads and reported in the manifest:

- **Clone skeleton:** digest of question + target with numbers masked (`\d+(\.\d+)?` →
  `<n>`). A skeleton above `MAX_PER_TEMPLATE_FAMILY` fails the build.
- **Family cap:** promote `MAX_FEEDBACK_PER_TEMPLATE_FAMILY = 50`
  (`workbench/feedback.py:17`) to a shared `MAX_PER_TEMPLATE_FAMILY` used by both paths.
- **Family leakage:** seed `dataset.py` gains the split check the workbench already has
  (`contracts.py:185`); require `template_family` in provenance.
- **Negative controls:** each trigger family must have ≥ 1 `near_miss` case linked by the
  existing `distinct_from` field; paraphrases share `equivalence_id`.
- **Manifest:** `template_family_counts`, `skeleton_counts`, `situation_counts`, coverage cells.

### 4. Derivation change

Numbers are currently accepted either as exact extracts or through human `semantic_review`.
A derivation makes numeric claims **deterministically verifiable**, cutting reviewer load
without weakening admission.

**Prerequisite — units.** Add `unit` to `MetricValue` and to each metric in
`configs/schema_registry.yaml`; `factsheet.py` populates it. (`RiskDriverDetail` already asks
the model for a `unit` it is never shown.)

```python
class Derivation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metric: str                          # key in factsheet.calculated_metrics
    observed: float
    operator: Literal["gt", "gte", "lt", "lte", "eq"]
    threshold: float
    unit: str
    threshold_evidence_id: str
    rule_id: str | None = None           # set once §5 exists
    holds: bool

class SupportedClaim(BaseModel):
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)
    derivation: Derivation | None = None
```

`check_derivation` (deterministic, no model):
1. metric exists and `validation_status == "valid"`
2. `observed` equals `MetricValue.value`
3. `unit` equals `MetricValue.unit`
4. `threshold` appears in the text of `threshold_evidence_id` (or equals the rule threshold when `rule_id` is set)
5. `holds` equals the operator applied

Wiring: `is_admissible_training_target` accepts a claim whose derivation verifies without
`semantic_review` for the numeric content; prose still goes to `judge.py` in evaluation.
**Prompt/schema change → bump `PROMPT_VERSION`, `DATASET_VERSION`; prior adapters are not
comparable.**

### 5. Policy rules before answering (hand-authored)

Thresholds exist only in policy prose, and `risk_tiers.py` runs after answering using
hardcoded English substrings with no provenance. Add a registry mirroring `SchemaRegistry`:

```yaml
# configs/policy_rules.yaml
version: "1.0.0"
default_deny: true
rules:
  - rule_id: sama.watchlist.utilisation.v1
    metric: utilisation_pct
    operator: gte
    threshold: 85.0
    unit: pct
    action: flag_watchlist
    mandatory: true
    jurisdiction: SAMA
    portfolio: [sme, corporate]
    evidence_id: SAMA-CIRC-4#7.2
    document_version: "2.0"
    effective_from: 2025-07-01
    effective_to: null
    approved_by: reviewer-id
    quote: "Facilities utilised at or above 85% shall be placed on watchlist."
```

- **Load:** default-deny, `expected_version` pin, metric must exist in `CALCULATORS` /
  registry, unit must match metric unit, `quote` must be verbatim in the approved chunk.
- **Scope:** jurisdiction, portfolio, effective dates — reuse the `rag/filters.py`
  predicate shape.
- **Before model call:** `evaluate_rules(factsheet, context)`; a mandatory rule that cannot
  be evaluated returns `INSUFFICIENT_EVIDENCE` with no model call.
- **In prompt:** `rule_evaluations` beside `factsheet` and `evidence` (prompt version bump).
- **After:** a claim contradicting a fired rule fails; `Derivation.rule_id` must match.
- **Data prep use:** generators draw thresholds from the registry, so near-miss and breach
  cases are built around real triggers rather than invented numbers.
- **Deferred:** LLM extraction of candidate rules at ingest + SME approval screen.
- **Limit:** only numeric, unconditional clauses become rules; conditional clauses stay
  retrieved text.

---

## Steps

**Phase 0 — land current work**
1. Commit the working tree in coherent slices (CI/pre-commit; calculations + registry 1.3.0; training manifest v2; retrieval + ingest v2; evaluation v2 + judge + qualification; docs).
2. Push; confirm the GitHub workflow passes.
3. Remove duplicate `group_id` check in `dataset.py` (lines 36 and 45).

**Phase 1 — data-prep package (completed)**
4. `src/credit_risk/data_prep/` owns the fixture, gold, and spike generators.
5. `credit-risk-data-prep {fixture,gold,spike}` is the installed entry point.
6. Tests and operating docs use the package; old scripts were removed.

**Phase 2 — units (completed)**
7. `MetricValue`, registry 1.4.0, calculators, and direct factsheet metrics carry governed units.

**Phase 3 — coverage and diversity (completed)**
8. `taxonomy.py`: task types and situations as enums; validate payload `task_type`/`situation`.
9. `coverage.py` + `configs/data_prep/coverage_targets.yaml`; coverage cells written to the dataset manifest.
10. `diversity.py`: clone skeleton, shared `MAX_PER_TEMPLATE_FAMILY`, near-miss requirement.
11. Seed `dataset.py`: template-family split-leakage check and family cap; manifest counts.

**Phase 4 — derivation**
12. `Derivation` schema + `SupportedClaim.derivation`.
13. `data_prep/derivation.py` `check_derivation`; wire into `is_admissible_training_target`.
14. Bump `PROMPT_VERSION` and `DATASET_VERSION`; update prompt-alignment golden tests.

**Phase 5 — policy rules**
15. `configs/policy_rules.yaml` + `PolicyRuleRegistry` load validations.
16. Scope selection reusing the filters predicate; `evaluate_rules`.
17. `api.py`: evaluate before model call; mandatory-unevaluable → `INSUFFICIENT_EVIDENCE`.
18. `build_user_content`: add `rule_evaluations`; version bump.
19. Output check: contradiction with fired rule fails; `rule_id` matches.
20. Migrate `risk_tiers.py` constants into the registry with provenance.

**Phase 6 — generate data to the targets**
21. Extend `data_prep/gold.py` to cover every taxonomy row and situation, reaching ≥ 30 per gated slice; SME review before freeze.
22. Generate training candidates per family with the new situations; admit through `build_dataset`; iterate until coverage and diversity reports pass.

**Deferred**
23. LLM rule-candidate extraction at ingest + workbench approval.

---

## Verification

| Step | Check |
|---|---|
| 1–3 | `uv lock --check`; `uv run ruff check .`; `uv run pytest -q` = 441 passed / 35 skipped |
| 4–6 | Existing fixture consumers pass; the unified fixture and gold commands are covered; no legacy script remains |
| 7 | Every `calculated_metrics` entry in a built factsheet has a non-null `unit` |
| 8–11 | Payload with unknown `task_type` rejected; 51 cases in one family fail build; same family in train and test fails; digits-only variants collapse to one skeleton; a trigger family without a near-miss fails |
| 12–14 | Wrong `observed`, wrong unit, threshold absent from evidence, or wrong `holds` each rejected; a correct non-extractive numeric claim is admitted without `semantic_review`; prompt-alignment tests pass on the new version |
| 15–16 | Rule with unknown metric, mismatched unit, or non-verbatim quote rejected at load; rule effective 2026-03-01 does not fire for `as_of_date` 2026-01-15 |
| 17 | Factsheet missing `facility_limit` with a mandatory utilisation rule returns `INSUFFICIENT_EVIDENCE` and the model client is never called |
| 18–19 | Serialised user turn contains `rule_evaluations` with fixed key order; "within tolerance" at 88.4% vs 85% rule fails |
| 20 | Existing `risk_tiers` tests pass against registry-backed rules |
| 21–22 | Coverage report shows every target cell met; `evaluate_gates` on the new gold set returns gate results rather than insufficient for each slice with ≥ 30 |
