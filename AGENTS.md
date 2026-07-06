# AGENTS.md

Rules for coding agents working on `mini-vla-from-scratch`.

This project is a minimal Vision-Language-Action framework built from scratch for learning robot action prediction.

Current project direction:

```text
Toy2D data
-> Dataset / Collate
-> MiniVLA forward
-> Behavior Cloning training
-> Inference / Visualization
-> Fake Robot Rollout
-> Future Open-kaka Adapter
```

The project is intentionally small, explicit, and educational. Do not turn it into a full LeRobot / LIBERO / OpenVLA / SmolVLA reproduction.

---

## Language

* Reply to the user in Chinese unless another language is requested.
* Use English for code, comments, filenames, identifiers, CLI names, config keys, and commit messages.
* Keep explanations concise and stage-oriented.

---

## Project Scope

This project is:

* A from-scratch mini VLA implementation.
* A learning-oriented engineering skeleton.
* A small controllable pipeline for:

  * toy data generation
  * dataset loading
  * model forward
  * behavior cloning
  * inference
  * fake rollout
  * future robot adapter skeleton

This project is not:

* A full SmolVLA reproduction.
* A full OpenVLA / Octo / π0 implementation.
* A LeRobot fork.
* A LIBERO benchmark integration.
* A MuJoCo / Isaac Sim project.
* A real robot control stack.
* A real safety-certified robot system.

Reference materials such as TinyVLA, SmolVLA, Every-Embodied, and Open-kaka-robot may guide design, but do not copy their complexity into this project unless explicitly requested.

---

## Current Architecture

The current MiniVLA architecture is:

```text
image
-> SmallCNNVisionEncoder
-> image_feat

input_ids + attention_mask
-> LLM-ready TextBackbone
-> text_feat

state
-> StateEncoder
-> state_feat

concat(image_feat, text_feat, state_feat)
-> FusionMLP
-> ActionHead
-> action_pred
```

Important rules:

* Do not reintroduce GRU-only language encoding as the main path.
* The text backbone must remain LLM-ready.
* `MockLLMTextEncoder` is used for tests and no-network environments.
* `LLMTextEncoder` is reserved for future HuggingFace/local LLM integration.
* Pytest must not require downloading real LLM weights.
* Text backbone is frozen by default during Stage 3 training.

---

## Config Rules

Use config as a single source of truth.

Current model config:

```text
configs/model/mini_vla_cnn_llm.yaml
```

Expected behavior:

* `build_model(config)` builds the MiniVLA model.
* `model.name` must be `mini_vla`.
* Unsupported component types must raise `ValueError`.
* Dimension mismatches must raise explicit errors.
* `model.action_dim` controls action output dimension.
* Do not duplicate action dimension under `action_head.output_dim`.

Do not silently ignore config fields.

Do not leave old config paths such as `mini_vla_cnn_gru` active in training flows.

---

## Stage Discipline

Work stage by stage.

### Completed

* Stage 0: project skeleton
* Stage 1: Toy2D data pipeline
* Stage 2: MiniVLA model forward and config-driven builder

### Current next stage

Stage 3: Behavior Cloning training loop.

Stage 3 should be implemented in small steps:

```text
Stage 3-0: training design and config cleanup
Stage 3-A: loss / metrics / optimizer
Stage 3-B: single training step
Stage 3-C: minimal trainer loop
Stage 3-D: checkpoint save/load
Stage 3-E: train CLI integration
Stage 3-F: docs and roadmap cleanup
```

Do not skip directly to a large trainer.

Do not implement inference, rollout, MuJoCo, LeRobot, LIBERO, or real robot adapter during Stage 3.

---

## Think Before Coding

Before editing code:

* Identify the current stage.
* Identify the smallest responsibility boundary that owns the change.
* Check whether an existing file should be extended before creating a new file.
* State assumptions if the task is ambiguous.
* Push back if the requested change expands scope unnecessarily.
* Prefer direct, readable code over clever abstractions.

Do not guess silently.

---

## Existing Structure First

New behavior does not automatically require new files.

Follow this order:

1. Extend an existing function if the behavior naturally belongs there.
2. Add a small helper to an existing file if the file already owns the responsibility.
3. Add a new file only when the responsibility is clearly distinct and likely to grow.
4. Add a new directory only when multiple related files form a stable feature domain.
5. Do not add a new top-level directory unless explicitly requested.

When unsure, choose the smaller change and explain the tradeoff.

---

## Surgical Changes

* Touch only files required by the task.
* Do not refactor unrelated code.
* Do not reformat unrelated code.
* Do not update docs unless the task asks for docs or stage cleanup.
* Do not silently fix unrelated issues.
* Mention unrelated issues in the summary instead.
* Preserve existing public behavior unless the task explicitly asks to change it.

---

## Testing Rules

Prefer targeted tests.

For new behavior:

* Add focused tests.
* Run the smallest relevant test command first.
* Run full `pytest` only for broad changes, stage completion, or release readiness.
* Always report the exact validation command used.
* If validation cannot be run, explain why.

For Stage 3 training:

* Test loss as a scalar.
* Test backward works.
* Test optimizer updates only trainable parameters.
* Test frozen text backbone remains frozen.
* Test a single DataLoader batch can run through model, loss, backward, and optimizer step.

Do not add slow integration tests unless explicitly needed.

---

## Training Rules

During Stage 3:

* Use Behavior Cloning.
* Use MSE loss:

```text
loss = MSE(action_pred, action_gt)
```

* Default trainable modules:

  * `vision_encoder`
  * `state_encoder`
  * `fusion`
  * `action_head`

* Default frozen module:

  * `text_encoder`

Do not add:

* LoRA
* real LLM finetuning
* diffusion policy
* flow matching
* action chunking
* LeRobot dataset
* LIBERO benchmark
* WandB
* multi-GPU training
* real robot rollout

unless explicitly requested.

---

## Dependency Rules

Do not introduce new heavy dependencies unless explicitly requested.

Avoid adding:

* `transformers` as a required dependency
* LeRobot
* LIBERO
* MuJoCo
* Isaac Sim
* WandB
* distributed training frameworks

Optional adapters may exist, but tests must not require network downloads or large model weights.

---

## Reference Project Rules

Reference projects may guide design, but they must not control implementation.

### TinyVLA / SmolVLA

Use for:

* lightweight VLA design
* LLM-ready backbone idea
* action decoder concept
* training strategy inspiration

Do not copy:

* full model scale
* diffusion / flow matching
* action chunking
* large-scale training pipeline

### Every-Embodied

Use for:

* learning roadmap
* SmolVLA / LIBERO / LeRobot background
* MuJoCo and embodied AI learning references

Do not copy now:

* LIBERO integration
* LeRobot training
* MuJoCo dependency
* full reproduction scripts

### Open-kaka-robot

Use later for:

* robot interface structure
* safety rules
* teleoperation concepts
* adapter design

Do not use now for:

* Stage 3 training
* model architecture
* real robot control

---

## Code Quality

* Keep functions short.
* Keep control flow explicit.
* Prefer explicit errors over silent fallbacks.
* Avoid duplicate logic and second sources of truth.
* Do not swallow errors with broad `try/except`.
* Follow existing project style.
* Keep data schemas simple and explicit.
* Use clear tensor shape comments where useful.
* When returning result dictionaries, expose output paths if the caller or CLI needs them.
* Avoid path ambiguity by writing outputs into stable subdirectories.

---

## Security

* Never hardcode secrets, tokens, API keys, or credentials.
* Use environment variables for sensitive values.
* Do not log secrets.
* Validate external input at system boundaries.
* Treat external model/provider output as untrusted.
* Do not concatenate user input into shell commands.

---

## Planning

For trivial edits, proceed directly.

For non-trivial tasks, give a short plan:

```text
Goal:
Files:
Approach:
Validation:
```

For large tasks, split work into small verifiable stages.

Stop and ask when the goal or scope is unclear.

---

## Final Response Format

After code changes, report only:

```text
Changed:
- file: what changed

Validated:
- command used

Notes:
- risks or follow-up only if relevant
```

Do not include long explanations unless asked.

If no code was changed, summarize the recommendation directly.
