# Roadmap

## V0: Toy 2D Mini VLA

Goal:

```text
image + instruction + 2D state -> 2D action
```

Tasks:

- Project skeleton
- Config system
- Toy 2D dataset generation
- Dataset loader
- MiniVLA model
- Training loop
- Evaluation script
- Inference visualization
- Fake robot rollout

## V1: Action Chunk Prediction

Goal:

```text
image + instruction + state -> future action sequence
```

Tasks:

- Add action horizon
- Update dataset format
- Update action head
- Add rollout metrics

## V2: Episode Robot Dataset

Goal:

Support robot-style episode data.

## V3: Robot Interface Layer

Goal:

Prepare for virtual or real robot integration.

## V4: Pretrained Encoder Upgrade

Goal:

Replace small encoders with pretrained components.

## V5: Real / Virtual Robot Rollout

Goal:

Connect model output to robot simulation or virtual arm interface.
