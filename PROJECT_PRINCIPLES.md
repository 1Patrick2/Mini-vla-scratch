# Mini VLA from Scratch - Project Principles

## 1. Project Goal

This project is a minimal Vision-Language-Action framework built from scratch for learning robot action prediction.

The goal is not to reproduce a large-scale state-of-the-art VLA model. The goal is to understand the full pipeline of a VLA system:

- configuration management
- dataset and episode format
- image / language / state encoding
- multimodal fusion
- continuous action prediction
- behavior cloning training
- validation and inference
- fake robot rollout
- future robot interface integration

## 2. Core Scope

The first version focuses on a toy 2D manipulation task:

```text
image + instruction + state -> action
```

The model predicts a continuous action vector from visual observation, language instruction, and current state.

## 3. Non-Goals for V0

The first version does not include:

- real robot control
- real-time robot deployment
- large pretrained VLA models
- diffusion policy
- action tokenization
- multi-camera input
- depth camera input
- LeRobot data collection
- OpenVLA / SmolVLA fine-tuning
- complex web UI

These features may be added in later stages after the minimal pipeline is fully understood.

## 4. Directory Responsibility Rules

Each directory must have a clear responsibility.

### `configs/`

Only stores configuration files. No Python logic should be placed here.

### `mini_vla/config/`

Loads, validates, and merges configuration files.

### `mini_vla/datasets/`

Converts raw data into tensors that the model can consume. Dataset code must not depend on model internals.

### `mini_vla/models/`

Defines neural network modules only. Model code must not read files, save checkpoints, or run training loops.

### `mini_vla/training/`

Contains training loops, validation loops, losses, metrics, optimizers, and checkpoints.

### `mini_vla/inference/`

Loads trained models and performs prediction, rollout, and visualization.

### `mini_vla/robot_interface/`

Defines fake robot, action adapter, safety wrapper, and future robot integration interfaces.

### `scripts/`

Provides command-line entry points only. Scripts should call package modules and should not contain core business logic.

### `tests/`

Contains unit tests and minimal integration tests.

### `docs/`

Stores design documents, architecture notes, data format descriptions, and integration plans.

## 5. Code Generation Rule

Large language models may be used to assist development, but the project must remain human-readable and human-controlled.

For every new module:

1. Define the responsibility of the file.
2. Implement the minimal working version.
3. Read and understand the code.
4. Add or update tests.
5. Run the relevant command.
6. Update documentation if the design changes.

Do not generate large amounts of unrelated code at once.

## 6. Minimalism Rule

Prefer simple, explicit implementations over clever abstractions.

A module should only be added when it has a clear responsibility. Avoid meaningless directories, unused helper files, and premature framework design.

## 7. Stage-Based Development Rule

Each stage must have:

- a clear goal
- a runnable command
- an expected output
- a test or validation method
- a short note in the documentation

A stage is not complete until it can be reproduced from the command line.

## 8. Reproducibility Rule

Training and evaluation should be reproducible.

The project should provide:

- fixed random seed support
- config-driven experiments
- checkpoint saving
- log saving
- clear output directories
- documented commands

## 9. Safety Rule

No real robot should be controlled by default.

Any future real robot execution must go through:

- explicit enable flag
- action clipping
- joint / workspace limit checks
- speed limit checks
- emergency stop interface
- dry-run mode
- logging and replay support

## 10. Learning Priority

This project prioritizes understanding over performance.

A simple model that is fully understood is better than a large model that only runs as a black box.
