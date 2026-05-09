# Soccer TIPS Layer 1 v0.1 QA Cleanup Patch

This patch performs a QA and cleanup pass after the accepted Layer 1 Ingestion Infrastructure v0.1 milestone.

## Files changed or added

Replace these existing files:

```text
src/soccer_tips/data/skillcorner_loader.py
src/soccer_tips/data/build_layer1.py
tests/test_layer1_ingestion.py
```

Add this new file:

```text
src/soccer_tips/data/qa_layer1_outputs.py
```

## What changed

### SkillCorner warning cleanup

`load_dynamic_events()` now reads dynamic events with:

```python
pd.read_csv(events_path, low_memory=False)
```

This is intended to remove the mixed-dtype warning from the wide SkillCorner dynamic events CSV while preserving existing optional missing-file behavior.

### Controlled batch mode

The Layer 1 build runner now supports:

```powershell
$env:PYTHONPATH="src"
python -m soccer_tips.data.build_layer1 --provider all --batch
```

Batch scope:

- all SkillCorner matches discovered from `external/skillcorner/data/matches.json`
- `Metrica Sample_Game_1`
- `Metrica Sample_Game_2`

`Sample_Game_3` remains out of scope for v0.1.

### QA output inspection

Run:

```powershell
$env:PYTHONPATH="src"
python -m soccer_tips.data.qa_layer1_outputs
```

This inspects currently processed outputs and writes:

```text
data/processed/validation_reports/layer1_v01_qa_summary.json
```

It reports:

- output file names
- CSV shapes
- metadata JSON shape/key counts
- validation report status and warnings
- canonical tracking player schema compliance
- canonical tracking ball schema compliance
- validation report top-level schema compliance

## Sanity test

Run:

```powershell
$env:PYTHONPATH="src"
python tests/test_layer1_ingestion.py
```

Expected:

```text
All Layer 1 ingestion sanity checks passed.
```

## Scope boundary

This patch does not add Layer 2 metrics, compactness, transition stability, tactical identity modeling, dashboards, AI interpretation, or reporting.
