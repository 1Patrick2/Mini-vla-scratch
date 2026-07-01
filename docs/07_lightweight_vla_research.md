# Lightweight VLA Research Notes

## Why This Document

Stage 2/3 need external references for model forward and training design.
Open-kaka-robot is useful for robot control and safety, but not for model architecture.
TinyVLA and SmolVLA are the primary references for lightweight VLA design.
This project keeps a minimal implementation, not a full reproduction.

## TinyVLA

**Reference:** [TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation](https://arxiv.org/abs/2409.12514)

**Key idea:** Compact VLA, fast inference, data efficiency.

### What we adopt
- Keep model lightweight.
- Keep vision-language-action components explicit.
- Treat action decoder as a separate module.
- Avoid huge robot pretraining in early stages.

### What we do NOT copy now
- Diffusion policy decoder.
- Real robot benchmark.
- Large-scale robot data pipeline.

## SmolVLA

**Reference:** [SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics](https://arxiv.org/abs/2506.01844)

**Key idea:** Affordable and efficient VLA, lower training/inference cost, future async inference and action chunking.

### What we adopt
- Use LLM/VLM-ready text backbone.
- Keep inference and robot execution separable for later rollout.
- Keep architecture config-driven.
- Design for future LeRobot-style data compatibility.

### What we do NOT copy now
- Full 450M model reproduction.
- Flow matching.
- Action chunking.
- Async inference stack.
- Full LeRobot training.

## Design Decision: Stage 2 Current Model

```
image -> SmallCNNVisionEncoder -> image_feat    Tensor[B, 128]
input_ids + attention_mask -> LLM-ready TextBackbone -> text_feat    Tensor[B, 128]
state -> StateEncoder -> state_feat    Tensor[B, 128]
concat(image_feat, text_feat, state_feat) -> FusionMLP -> fused_feat    Tensor[B, 128]
fused_feat -> ActionHead -> action_pred    Tensor[B, action_dim]
```

### Key points
- Text backbone is LLM-ready, not a fixed GRU.
- `MockLLMTextEncoder` is used for testing (no transformers dependency).
- `LLMTextEncoder` is reserved for HuggingFace/local LLM integration.
- Text backbone is frozen by default in Stage 3 training.
- Action is continuous regression (not tokenization, not diffusion, not flow matching).

## Design Decision: Stage 3 Training Strategy

- Behavior cloning with MSE loss.
- Freeze text backbone by default.
- Train vision_encoder, state_encoder, fusion, action_head.
- Do not full-finetune LLM in early stage.
- Do not implement LoRA yet, but leave config space for it.

## Future References

- **Kaka-VLA:** To be added when concrete materials are available.
- **Open-kaka-robot:** Used in Stage 5/6 for safety, rollout and adapter design.
