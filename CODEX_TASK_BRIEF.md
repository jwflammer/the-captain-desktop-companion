# CODEX_TASK_BRIEF.md

## What this project is

Captain is a persistent desktop companion prototype built in Python and PyQt.
It is not a chatbot wrapper.

## What Codex should optimize for

1. Stability
2. Minimal breakage
3. Preserving the one-file working prototype
4. Keeping perception, state, and persistence intact
5. Small, testable changes

## First tasks Codex is allowed to do

- clean imports
- remove dead code only when safe
- improve naming where it does not break behavior
- improve README / repo hygiene
- add small helper functions
- tighten shutdown and persistence logic carefully
- improve comments and structure for maintainability

## First tasks Codex should avoid unless explicitly asked

- full architecture rewrite
- UI redesign
- replacing the companion concept
- large dependency changes
- moving everything into packages and submodules
- changing the sensory model or internal drives without instruction

## How to work

- read `AGENTS.md` first
- inspect `captain_ai_v1.py`
- propose small diffs
- keep behavior the same unless asked to change behavior
- prefer one change set at a time
