# AGENTS.md

## Project purpose

Captain is a persistent desktop companion prototype.
Do not turn Captain into a chatbot puppet, generic assistant shell, or LLM wrapper.
Preserve the idea of Captain as a synthetic organism with state, memory, perception, and visible embodiment.

## Current repo rules

- Keep the main working app in `captain_ai_v1.py` unless the task explicitly asks for a refactor.
- Make surgical changes first.
- Do not redesign the whole UI unless asked.
- Do not remove persistence features unless asked.
- Prefer stability over flashy additions.
- Preserve the current data folder behavior.

## Coding rules

- Use clear, boring Python over cleverness.
- Keep new helper functions small and explicit.
- Add comments only where they help maintainability.
- Avoid introducing extra dependencies unless clearly justified.
- If changing shutdown, saving, or persistence behavior, keep changes minimal and easy to revert.

## Review rules

Before finishing:
- confirm the app still boots
- confirm keyboard shortcuts still exist if relevant
- confirm persistence paths are unchanged unless the task explicitly requires otherwise
- mention any tradeoffs or risks plainly

## Current architecture landmarks

Important classes in the current file:
- `SimWorld`
- `DriveSystem`
- `VisionCortex`
- `RawCortex`
- `CommandTerminal`

## What not to do

- Do not convert this into a web app.
- Do not replace the core with an API chatbot.
- Do not silently change the companion concept.
- Do not split into modules unless the task explicitly asks for repository restructuring.
