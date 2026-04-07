# Captain Desktop Companion

Captain is a persistent desktop companion prototype built as a synthetic organism experiment rather than a chatbot shell.

This repository currently starts from a **single-file Python prototype** and keeps the working code intact while making the project easier to use with GitHub and Codex.

## Current state

The current `captain_ai_v1.py` includes:

- Dream State / Sim World
- Vision cortex with webcam, desktop, panel, and dream views
- Drive system with energy, stress, curiosity, social bond, comfort, fatigue, and familiarity
- Raw cortex persistence and externalized runtime data folders
- Shared control panel / shell experiments
- PyQt desktop UI
- Journal, DB, cache, snapshot, and weight directories created next to the script at runtime

## Repo goals

- Keep the first public repo simple and honest
- Preserve the one-file working prototype for now
- Make the codebase readable enough for future refactors
- Give Codex clear instructions so it stops thrashing the UI or architecture

## Run

```bash
python captain_ai_v1.py
```

## Dependencies

Install:

```bash
pip install -r requirements.txt
```

## Notes

This repo intentionally keeps the current prototype in one file while the design is still moving.

The next cleanup step after this starter pack should be:
1. stabilize one visual/body direction
2. freeze top-level UI behavior
3. split the file into modules only after behavior stops changing every session

## Data files

The current code creates runtime data folders such as:

- `captain_v1_data/identity/`
- `captain_v1_data/logs/`
- `captain_v1_data/snapshots/`
- `captain_v1_data/cache/`
- `captain_v1_data/identity/weights/`

These are ignored in `.gitignore` and should not be committed.

## Short project summary

Captain is a persistent desktop computer companion prototype with visible embodiment, internal state, memory, multi-eye perception, and a future migration path into other bodies such as RC crawler hardware.
