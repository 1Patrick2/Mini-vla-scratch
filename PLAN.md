# Mini VLA from Scratch — 长期开发计划

> 本文档基于 `interact.md` 专家方案和 `PROJECT_PRINCIPLES.md` 核心原则制定，
> 是项目的最高级路线图。每个阶段完成后在下方标记日期。

---

## 项目总览

```
image + instruction + state → action
```

| 阶段 | 内容 | 状态 |
|------|------|------|
| Stage 0 | 项目骨架 + 基准文件 | ✅ 完成 |
| Stage 1 | Toy 2D 数据管道 (generator → dataset → collate) | ✅ 完成 |
| Stage 2 | MiniVLA 模型 forward | ✅ 完成 |
| Stage 3 | Behavior Cloning 训练闭环 | ✅ 完成 |
| Stage 4 | 推理 + 可视化 (Policy Inference) | ✅ 完成 |
| Stage 5 | PushT Dataset Evaluation | ✅ 完成 |
| Stage 6 | Open Kaka Adapter 预留 | ⏳ |
| V1+ | action chunk / episode / pretrain | 🔮 远期 |

---

## Stage 0：项目骨架 ✅

**当前状态：已完成**

### 已创建文件

```
README.md
PROJECT_PRINCIPLES.md
SETUP.md
ROADMAP.md
CHANGELOG.md
pyproject.toml
requirements.txt
requirements-dev.txt
environment.yml
setup_env.sh
setup_env.ps1
.env.example
.gitignore
configs/base.yaml
configs/model/mini_vla_cnn_llm.yaml
configs/data/toy_2d.yaml
configs/data/episode_robot.yaml
configs/train/debug.yaml
configs/train/default.yaml
mini_vla/__init__.py
mini_vla/config/__init__.py
mini_vla/config/loader.py
mini_vla/config/schema.py
mini_vla/datasets/__init__.py  (placeholder)
mini_vla/datasets/collate.py
mini_vla/datasets/episode_dataset.py  (placeholder)
mini_vla/datasets/toy_2d_dataset.py
mini_vla/datasets/transforms.py
mini_vla/models/__init__.py  (placeholder)
mini_vla/models/action_head.py
mini_vla/models/fusion.py
mini_vla/models/language_encoder.py
mini_vla/models/mini_vla.py
mini_vla/models/state_encoder.py
mini_vla/models/vision_encoder.py
mini_vla/training/__init__.py
mini_vla/training/trainer.py
mini_vla/training/losses.py
mini_vla/training/metrics.py
mini_vla/training/optimizer.py
mini_vla/training/checkpoint.py
mini_vla/inference/__init__.py
mini_vla/inference/predictor.py
mini_vla/inference/rollout.py
mini_vla/inference/visualizer.py
mini_vla/robot_interface/__init__.py
mini_vla/robot_interface/action_adapter.py
mini_vla/robot_interface/fake_robot.py  (placeholder)
mini_vla/robot_interface/safety_wrapper.py
mini_vla/robot_interface/open_kaka_adapter.py  (placeholder)
mini_vla/utils/__init__.py
mini_vla/utils/seed.py
mini_vla/utils/logging.py
mini_vla/utils/paths.py
mini_vla/utils/tensor.py
scripts/train.py
scripts/generate_toy_data.py
scripts/evaluate.py
scripts/infer_one.py
scripts/rollout_fake_robot.py
tests/test_config_loader.py
tests/test_action_adapter.py
tests/test_trainer.py
docs/00_project_scope.md ~ docs/05_open_kaka_integration.md
```

### 验收命令

```bash
pip install -e .
python -c "import mini_vla; print('OK')"
pytest                          # 4 passed
python scripts/train.py --config configs/train/debug.yaml --dry-run
```

---

## Stage 1：Toy 2D 数据生成 ✅

**目标已完成。** 拆为 3 个子阶段：

- Stage 1-A ✅ Toy 2D Data Generator (`scripts/generate_toy_data.py`)
- Stage 1-B ✅ Toy2DDataset (`mini_vla/datasets/toy_2d_dataset.py`)
- Stage 1-C ✅ DataLoader Collation (`mini_vla/datasets/collate.py`)

### 目标

生成合成 2D manipulation episode 数据，供后续模型训练使用。

### 设计

场景：固定大小画布上有一个红点（object）和一个绿点（target）。
语言指令指示移动方向：

```
"move red object to target"
"move red object left"
"move red object up"
"move red object right"
"move red object down"
```

每个 episode 包含多个 step，每个 step 包含：
- image: 当前画布状态 (RGB, 64×64)
- state: object 当前坐标 [x, y]
- action: 下一步位移 [dx, dy]
- instruction: 自然语言指令文本
- target: 目标位置坐标

### 输出格式

```
data/toy_2d/
└── episodes/
    ├── ep_000000/
    │   ├── frames/
    │   │   ├── 000000.png
    │   │   └── ...
    │   └── episode.json
    └── ep_000001/
        ├── frames/
        └── episode.json
```

Each `episode.json`:
```json
{
  "episode_id": "ep_000000",
  "instruction": "move red object to target",
  "steps": [
    {
      "frame": "frames/000000.png",
      "state": [0.25, 0.40],
      "action": [0.03, -0.01],
      "target": [0.80, 0.20]
    }
  ]
}
```

### 文件变更

| 文件 | 操作 |
|------|------|
| `scripts/generate_toy_data.py` | 重写：数据生成入口 |
| `mini_vla/datasets/toy_2d_dataset.py` | 重写：读取生成的数据 |
| `mini_vla/datasets/transforms.py` | 按需更新 |
| `mini_vla/datasets/collate.py` | 按需更新 |
| `tests/test_dataset.py` | 新增：dataset 加载测试 |
| `configs/data/toy_2d.yaml` | 按需更新 |

### 约束

- 不引入外部图片
- 全合成数据，确保可复现
- seed 固定
- 验收命令必须包含 `--num-episodes` 参数

### 验收

```bash
python scripts/generate_toy_data.py --config configs/data/toy_2d.yaml --num-episodes 100
python -c "
from mini_vla.datasets.toy_2d_dataset import Toy2DDataset
ds = Toy2DDataset('data/toy_2d')
sample = ds[0]
print(sample.keys())
print('image:', sample['image'].shape)
print('action:', sample['action'].shape)
"
pytest tests/test_dataset.py
```

---

## Stage 2：MiniVLA 模型 Forward ✅

### 目标 ✅ 已完成

当前真实架构：

```
image → SmallCNNVisionEncoder → image_feat    Tensor[B, 128]
input_ids + attention_mask → MockLLMTextEncoder → text_feat    Tensor[B, 128]
state → StateEncoder → state_feat    Tensor[B, 128]
concat(image_feat, text_feat, state_feat) → FusionMLP → ActionHead → action_pred    Tensor[B, 2]
```

### 已完成组件

| 文件 | 操作 |
|------|------|
| `mini_vla/models/vision_encoder.py` | SmallCNNVisionEncoder |
| `mini_vla/models/language_encoder.py` | MockLLMTextEncoder (LLM-ready) |
| `mini_vla/models/state_encoder.py` | StateEncoder MLP |
| `mini_vla/models/fusion.py` | Concat + FusionMLP |
| `mini_vla/models/action_head.py` | Continuous MLP ActionHead |
| `mini_vla/models/mini_vla.py` | 完整 MiniVLA 组装 |
| `mini_vla/models/builder.py` | Config-driven build_model() + strict validation |
| `tests/test_model_forward.py` | Forward shape 测试 |
| `tests/test_model_builder.py` | Builder config/dim validation 测试 |

### 验收

```bash
pytest tests/test_model_forward.py -v
```

预期输出 shape：

```python
batch = {
    "image":    torch.randn(4, 3, 64, 64),
    "input_ids": torch.randint(0, 128, (4, 16)),
    "state":    torch.randn(4, 2),
}
output = model(batch)  # shape: (4, 2)
```

---

## Stage 3：Behavior Cloning 训练闭环 ✅

### 目标

跑通完整训练流程：加载数据 → forward → loss → backward → checkpoint。

### 核心组件

| 文件 | 职责 |
|------|------|
| `mini_vla/training/losses.py` | action MSE loss |
| `mini_vla/training/metrics.py` | MSE / L1 / direction accuracy |
| `mini_vla/training/optimizer.py` | Adam / scheduler 配置 |
| `mini_vla/training/checkpoint.py` | save / load / best model tracking |
| `mini_vla/training/trainer.py` | 完整 train / val loop |
| `tests/test_training.py` | 新增：完整训练管线测试 |

### 训练流程

```
for epoch in range(epochs):
    for batch in train_loader:
        pred = model(batch)
        loss = loss_fn(pred, batch["action"])
        loss.backward()
        optimizer.step()
    
    eval on val_loader
    save checkpoint if best
```

### 验收

```bash
python scripts/train.py --config configs/train/debug.yaml
```

预期输出：

```
Epoch 1/5  train_loss=0.042  val_loss=0.039
Epoch 2/5  train_loss=0.021  val_loss=0.018
...
Checkpoint saved: outputs/checkpoints/best.pt
```

---

## Stage 4：推理 + 可视化 (Policy Inference) ✅

### 目标

加载训练好的模型，对单条样本预测 action，并可视化结果。

### 文件变更

| 文件 | 操作 |
|------|------|
| `mini_vla/inference/predictor.py` | 实现单样本预测 |
| `mini_vla/inference/visualizer.py` | 实现预测轨迹可视化 |
| `scripts/infer_one.py` | 重写：推理入口 |

### 可视化输出

```
image + predicted_action direction arrow → prediction.png
```

### 验收

```bash
python scripts/infer_one.py \
  --ckpt outputs/checkpoints/best.pt \
  --config configs/train/debug.yaml
```

输出：`outputs/predictions/prediction.png`

---

## Stage 5：PushT Dataset Evaluation ✅

### 目标

使用 `lerobot/pusht` 数据集验证 MiniVLA 在真实机器人学习 benchmark 上的动作预测能力。

### 子阶段

| 阶段 | 内容 |
|------|------|
| Stage 5-A | PushT 数据集检查 (inspection) |
| Stage 5-B | PushT Adapter (将 PushT sample 转为 MiniVLA 格式) |
| Stage 5-C | PushT config + batch forward |
| Stage 5-D | PushT 训练 smoke test (Trainer 通过 factory 接入 PushT) |
| Stage 5-E | PushT 离线评估 + baselines (zero/mean/previous-action) |
| Stage 5-F | 评估报告 + 可视化输出 |
| Stage 5-G | 文档收口 |

### 核心文件

- `docs/12_pusht_dataset_evaluation_design.md` — 设计文档
- `configs/data/pusht.yaml` — 数据配置
- `configs/train/pusht_debug.yaml` — 训练配置
- `mini_vla/datasets/pusht_adapter.py` — 适配器
- `mini_vla/datasets/pusht_inspection.py` — 数据检查
- `mini_vla/datasets/factory.py` — dataset factory
- `mini_vla/evaluation/action_metrics.py` — 评估指标
- `mini_vla/evaluation/baselines.py` — baseline 策略
- `mini_vla/evaluation/evaluator.py` — 离线评估引擎
- `scripts/inspect_pusht_dataset.py` — 检查 CLI
- `scripts/evaluate_pusht.py` — 评估 CLI

### 验收

```bash
python scripts/inspect_pusht_dataset.py --repo-id lerobot/pusht --max-samples 128
python scripts/train.py --config configs/train/pusht_debug.yaml
python scripts/evaluate_pusht.py \\
  --config configs/train/pusht_debug.yaml \\
  --ckpt outputs/checkpoints/best.pt
```

验收标准：
- inspect 能输出 PushT schema
- MiniVLA 能训练 PushT 子集（loss finite）
- MiniVLA test MAE < ZeroActionBaseline MAE
- MiniVLA test MAE < MeanActionBaseline MAE
- finite output ratio == 1.0
- report.json / predictions.jsonl 保存成功
```

---

## Stage 6：Open Kaka Adapter 预留 ⏳

### 目标

不连真机，只做接口预留和设计文档。

### 文件变更

| 文件 | 操作 |
|------|------|
| `mini_vla/robot_interface/open_kaka_adapter.py` | 接口定义 + 占位实现 |
| `docs/05_open_kaka_integration.md` | 更新：集成方案 |

### 验收

```python
from mini_vla.robot_interface.open_kaka_adapter import OpenKakaAdapter
adapter = OpenKakaAdapter()
action = [0.1, 0.0]
adapted = adapter.adapt(action)  # 不报错即可
```

---

## V1+：远期方向 🔮

| 方向 | 说明 |
|------|------|
| V1 action chunk | 预测未来 N 步动作序列 |
| V2 episode dataset | 支持真实机器人 episode 格式 |
| V3 pretrained encoder | 替换为 ResNet / TinyBERT |
| V4 真机 rollout | Open Kaka 虚拟从臂 + 安全闭环 |

---

## 目录职责速查

| 目录 | 职责 |
|------|------|
| `configs/` | 只放 YAML 配置，不放 Python 逻辑 |
| `mini_vla/config/` | 配置加载、校验、合并 |
| `mini_vla/datasets/` | 数据 → tensor，不依赖模型结构 |
| `mini_vla/models/` | 网络结构，不读文件、不存结果 |
| `mini_vla/training/` | 训练循环、loss、metrics、checkpoint |
| `mini_vla/inference/` | 预测、rollout、可视化 |
| `mini_vla/robot_interface/` | 动作适配、安全限制、机器人接口 |
| `mini_vla/utils/` | 通用工具：seed、logging、tensor |
| `scripts/` | CLI 入口，不写核心逻辑 |
| `tests/` | 单元测试 + 最小集成测试 |
| `docs/` | 设计文档，不堆代码 |
