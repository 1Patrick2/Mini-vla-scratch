# Policy Inference and Robot Learning Notes

## 1. Where This Project Fits in ML/DL/IL/RL

| Term | Full Name | Relation to This Project |
|------|-----------|--------------------------|
| **ML** | Machine Learning | The project trains a model to predict actions from observations — a supervised machine learning task. |
| **DL** | Deep Learning | The MiniVLA model uses neural networks (CNN, MLP, embeddings) — a deep learning approach. |
| **IL** | Imitation Learning | The model learns by imitating expert demonstrations (ground-truth actions from Toy2D episodes). |
| **BC** | Behavior Cloning | The specific IL method used: `loss = MSE(action_pred, expert_action)`. The model clones the expert's behaviour. |
| **VLA** | Vision-Language-Action | The model fuses image, text instruction, and state to predict an action — a classic VLA paradigm. |

This project is a **minimal Toy2D VLA with Behavior Cloning training and policy inference**.

## 2. What This Project Is Not (Yet)

| Area | Why Not Included |
|------|------------------|
| **RL (Reinforcement Learning)** | RL requires a reward function, environment interaction, and online policy improvement. Toy2D currently has no reward, no environment loop, and no online evaluation metric. |
| **ACT / Action Chunking Transformer** | ACT predicts a sequence of future actions and requires temporal context. The current Toy2D data is single-step, not sequential. |
| **Diffusion Policy** | Diffusion-based action generation adds significant complexity (denoising, multi-step sampling) not needed for Toy2D regression. |
| **Real Robot Control** | The robot interface is a placeholder skeleton (Stage 6). No real robot is controlled by default. |
| **LeRobot** | LeRobot is referenced for design ideas (policy API, checkpoint inference) but is not imported as a dependency. |

All of the above are candidates for V1+ stages after the core pipeline (Stage 0-6) is stable.

## 3. Stage 3 vs Stage 4

```
Stage 3: Train a policy with Behavior Cloning
  data → model → MSE loss → backward → optimizer.step → checkpoint

Stage 4: Run policy inference from a checkpoint
  checkpoint → Predictor / Policy → select_action → action → visualization
```

Stage 3 builds the **training** side of a robot learning pipeline.
Stage 4 builds the **deployment/inference** side.

Together they form a complete "train → deploy → visualize" loop for Toy2D.

## 4. Predictor / Policy API

The `Predictor` class in `mini_vla/inference/predictor.py` follows a LeRobot-inspired design:

- `predict(sample)` — project-internal name, single-sample inference
- `select_action(observation)` — policy-style alias, closer to robot learning notation

```
observation = {
    "image": Tensor[3, H, W],
    "input_ids": Tensor[T],
    "attention_mask": Tensor[T],
    "state": Tensor[state_dim],
}

action = Predictor.select_action(observation)
# Tensor[action_dim] — e.g. [dx, dy]
```

This separation is useful: project code can use `predict()`, while integration with robot environments or evaluation scripts can use `select_action()` for readability.

## 5. How LeRobot Inspired This Design

LeRobot (Hugging Face) describes a standard robot learning workflow:

1. **Record dataset** — collect teleoperated episodes
2. **Train policy** — IL/BC training on the dataset
3. **Run inference / evaluate policy** — load checkpoint, run policy, evaluate

Toy2D + MiniVLA replicates this workflow at minimal scale:

| LeRobot Step | Toy2D Equivalent |
|--------------|-------------------|
| Record dataset | `generate_toy_data.py` |
| Train policy | `scripts/train.py` + `Trainer` |
| Load checkpoint | `Predictor(config, checkpoint_path)` |
| Run policy | `Predictor.select_action(observation)` |
| Evaluate | `infer_one.py` + L1 error + visualization |

The current project does **not** use LeRobotDataset, LeRobot policy classes, or real robot record/eval. The design borrows the **API pattern** (policy → select_action → action) without importing LeRobot.

## 6. Why Transformer / RL Are Deferred

| Why not now? | Transformer | RL |
|--------------|-------------|-----|
| Data format | Toy2D is single-step; Transformer needs sequences or action chunks | No reward function or environment interaction exists |
| Model scope | Adding Transformer would require Stage 2/3 rework (model, training, checkpoint) | RL needs Stage 5 rollout metrics before it can optimize |
| Stage plan | Reserved for Stage 7 (Tiny Transformer Policy Experiment) | Reserved for Stage 8 (Fake RL / Reward Evaluation) |

The current MLP fusion is sufficient for Toy2D. Transformer / RL experiments belong in future stages after the full pipeline (data → train → inference → rollout → adapter) is stable.

## References

- [LeRobot: Imitation Learning on Real-World Robots](https://huggingface.co/docs/lerobot/il_robots)
- [LeRobot: Overview](https://huggingface.co/docs/lerobot/index)
- [TinyVLA: Compact VLA](https://arxiv.org/abs/2409.12514)
- [SmolVLA: Affordable VLA](https://arxiv.org/abs/2506.01844)
