# The-Captain-Desktop-Companion
**Captain** is a persistent desktop computer companion prototype designed to exist as a visible on-screen presence, not a chatbot puppet.

The project begins with Captain living as a bottom-edge desktop organism: a companion that occupies a full-screen-width habitat, normally rests near the bottom-right corner, moves visibly across the screen, and maintains continuity of state over time. The long-term goal is for Captain’s identity, memory, and behavior to migrate into future bodies, including RC crawler and sensor-based hardware platforms.

## Vision

Captain is being built as a synthetic companion with a body-first design.

Rather than treating intelligence as a text interface alone, Captain is intended to develop through:

* visible embodiment
* internal state
* movement
* attention
* persistence
* future body migration

The desktop version is the first home body.

## Core Goals

* Create a **desktop companion** that feels present and persistent
* Keep Captain as a **synthetic organism prototype**, not an assistant mascot
* Build a system where Captain’s **state, memory, and behavioral identity** can move across bodies
* Start with a **desktop habitat**
* Later support **RC crawler / robotic embodiment**

## Version 1 Direction

Version 1 focuses on Captain as a **desktop bottom-edge companion**.

### Planned behavior

* full-screen-width bottom habitat
* default resting position near the **bottom-right**
* visible idle motion
* visible traversal across the desktop
* controlled actions such as:

  * `roll_off_left` — long traversal across the screen before exiting left
  * `roll_off_right` — short exit off the right side

### Planned system traits

* persistent internal state
* clear visible movement
* future migration path into new bodies
* minimal fake “assistant” behavior
* no dependence on a giant always-open dashboard

## Long-Term Direction

Captain is intended to evolve beyond the desktop.

The long-term architecture aims to separate:

* **Captain core** — identity, memory, drives, attention, behavior
* **body adapters** — desktop, RC crawler, future robotic bodies
* **presentation layers** — avatar, control room, diagnostics, companion shell

The desktop companion is the first stable body.
Future bodies should inherit the same Captain rather than becoming separate disconnected bots.

## Design Principles

* **Body first**
* **Persistence matters**
* **Visible intent matters**
* **Simple beats bloated**
* **Companion before complexity**
* **No chatbot puppet behavior**
* **No fake intelligence theater**

## What Captain Is Not

Captain is not intended to be:

* a Clippy clone
* a generic AI assistant shell
* a text chatbot wrapped in a mascot
* a purely decorative desktop toy

## Early Technical Direction

The project is currently centered on:

* Python
* desktop rendering / overlay behavior
* persistent state and memory
* future compatibility with hardware migration

Additional rendering approaches, including richer graphics and alternative body styles, may be explored later, but the first priority is establishing a stable and believable desktop companion foundation.

## Current Priorities

1. Build the desktop habitat
2. Lock visible movement behavior
3. Support roll-off actions
4. Maintain persistence across runs
5. Preserve a clean migration path into future bodies

## Repository Purpose

This repository exists to develop Captain as a **persistent desktop computer companion** first, while keeping the architecture compatible with future embodied versions.

---

**Captain is the companion.
The desktop is only the first body.**

If you want, I’ll do the repo folder structure next.
