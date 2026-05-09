# Soccer TIPS Layer 1 Ingestion Infrastructure v0.1

This package contains a starter implementation for reusable Layer 1 ingestion infrastructure.

## Scope

This implementation covers:

- Foundation config, schemas, and common I/O helpers
- SkillCorner provider-specific loader
- Metrica Sample_Game_1 and Sample_Game_2 provider-specific loader
- Coordinate normalization utilities
- Structural validation helpers
- Layer 1 build runner
- Lightweight sanity checks

This implementation does **not** cover:

- compactness metrics
- transition stability metrics
- tactical identity modeling
- dashboards
- AI interpretation
- recommendation engines
- reporting layers
- Layer 2 feature engineering

## Expected project structure

```text
project-root/
  external/
    skillcorner/
      data/
        matches.json
        matches/
          <match_id>/
            <match_id>_match.json
            <match_id>_tracking_extrapolated.jsonl
            <match_id>_phases_of_play.csv
            <match_id>_dynamic_events.csv
    metrica/
      data/
        Sample_Game_1/
          RawEventsData.csv
          RawTrackingData_Home_Team.csv
          RawTrackingData_Away_Team.csv
        Sample_Game_2/
          RawEventsData.csv
          RawTrackingData_Home_Team.csv
          RawTrackingData_Away_Team.csv
  src/
    soccer_tips/
      config.py
      schemas.py
      data/
        common.py
        normalize.py
        skillcorner_loader.py
        metrica_loader.py
        validation.py
        build_layer1.py
  tests/
    test_layer1_ingestion.py
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run sanity checks

```bash
python tests/test_layer1_ingestion.py
```

## Run Layer 1 processing

From project root, depending on your Python setup:

```bash
python -m soccer_tips.data.build_layer1 --provider skillcorner --skillcorner-match-id 2017461
python -m soccer_tips.data.build_layer1 --provider metrica --metrica-game Sample_Game_1
python -m soccer_tips.data.build_layer1 --provider all --skillcorner-match-id 2017461 --metrica-game Sample_Game_1
```

If Python cannot find `soccer_tips`, run with `PYTHONPATH=src` on Mac/Linux or `$env:PYTHONPATH="src"` in PowerShell first, or install the package in editable mode after adding a pyproject later.

## Outputs

Processed outputs are written to:

```text
data/processed/skillcorner/<match_id>/
data/processed/metrica/<sample_game>/
data/processed/validation_reports/
```

These directories are ignored by Git in this v0.1 setup.
