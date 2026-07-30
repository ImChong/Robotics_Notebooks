# PRISM（Polynomial Representations for Interaction-Structured Motor Control）

> 来源归档

- **标题：** PRISM
- **类型：** repo
- **来源：** University of Michigan, Ann Arbor（Seung Hyun Lee / Stella X. Yu）
- **链接：** <https://github.com/lsh3163/prism>
- **项目页：** <https://lsh3163.github.io/prism/>
- **论文：** <https://arxiv.org/abs/2607.23473>（Submitted 2026-07-26）
- **许可：** 顶层 LICENSE 标注 *being finalized*（见 `NOTICE.md`）；BFM-Zero / LeRobot 补丁遵循各自上游条款
- **入库日期：** 2026-07-30
- **一句话说明：** 官方实现：独立 PyTorch `PRISMConditioner` + 针对 **BFM-Zero** / **SmolVLA（LeRobot）** 的源码补丁与论文对齐训练/评测配置。
- **沉淀到 wiki：** [`wiki/entities/paper-prism.md`](../../wiki/entities/paper-prism.md)

---

## 核心定位

只改**本体感觉条件通路**（factorized polynomial interactions），保留下游策略、动作接口与低层控制器；可插入 RL locomotion 与 VLA / Diffusion 模仿学习栈。

---

## 仓库入口（README，截至 2026-07-30）

| 组件 | 说明 |
|------|------|
| 安装 | `pip install -e ".[test]"`（Python 3.10+ / PyTorch 2.1+） |
| 单测 | `python -m unittest discover -s tests -v` |
| 核心包 | `src/prism_robot/` — `PRISMConditioner`；`polynomial_features()` 可检视中间多项式特征 |
| 集成 | `integrations/` — `bfm-zero.patch`、`bfm-zero-evaluation.patch`、`lerobot-smolvla.patch`（钉住上游 commit） |
| 配置 | `configs/bfm_zero_prism.env`、`configs/smolvla_prism.env` |
| 结果 / 复现 | `RESULTS.md`、`REPRODUCIBILITY.md` |
| 分析 | `analysis/` — 表征 / t-SNE 等脚本 |

### 最小用法

```python
from prism_robot import PRISMConditioner
conditioner = PRISMConditioner(
    input_dim=32, output_dim=1152, hidden_dim=1152,
    degree=2, interaction_mode="gated", gate_init=1e-2,
    post_mlp_layers=2, use_rmsnorm=True,
)
```

### 复现 BFM-Zero（需先装上游）

1. clone [LeCAR-Lab/BFM-Zero](https://github.com/LeCAR-Lab/BFM-Zero)，checkout 钉住 commit，apply `integrations/bfm-zero.patch`（评测再 apply evaluation patch）。
2. `source configs/bfm_zero_prism.env` → `uv run python -m humanoidverse.train`
3. 评测：`uv run python -m humanoidverse.tracking_eval ...`（低摩擦 / payload 覆盖见 `REPRODUCIBILITY.md`）

### 复现 SmolVLA @ LIBERO

1. clone [huggingface/lerobot](https://github.com/huggingface/lerobot)，钉住 commit，apply `lerobot-smolvla.patch`。
2. `lerobot-train ... --policy.state_conditioner_type=prism ...`（见 README / `configs/smolvla_prism.env`）
3. `lerobot-eval` 对齐 80K checkpoint 的官方 `eval50` 协议。

---

## 开源边界

- **本仓已开：** conditioner 实现、测试、补丁、配置、结果表。
- **不随仓分发：** 上游 checkpoint、仿真资产、LAFAN / LIBERO 数据（按各上游条款自行获取）。
- **Humanoid-Gym / Diffusion Policy 任务专属实验：** 论文主表有结果；公开仓当前重点放出 **更强 backbone（BFM-Zero / SmolVLA）** 集成路径。

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-prism](../../wiki/entities/paper-prism.md) | 论文实体与选型结论 |
| [paper-bfm-zero](../../wiki/entities/paper-bfm-zero.md) | 更强 RL backbone 集成对象 |
| [humanoid-gym](../../wiki/entities/humanoid-gym.md) | 论文主表 locomotion 评测栈 |
| [diffusion-policy](../../wiki/methods/diffusion-policy.md) | IL 侧替换线性本体条件的基线 |
