# perceptron-isaac

> 来源归档（repo）

- **标题：** perceptron-ai-inc/isaac
- **类型：** repo
- **组织：** Perceptron Inc.
- **代码：** https://github.com/perceptron-ai-inc/isaac
- **LeRobot fork / 子模块：** https://github.com/perceptron-ai-inc/lerobot（钉 `fe0ff792e07aea117057b0e193be6c57d45f2f30`）
- **权重 Hub：** https://huggingface.co/PerceptronAI/Isaac-0.5
- **许可证：** 代码 Apache License 2.0；权重条款以 Hugging Face 仓库为准
- **入库日期：** 2026-08-27
- **一句话说明：** Perceptron Isaac 0.5 的官方开源入口：根仓几乎只有 README + 钉死的 LeRobot 子模块；策略类型 `perceptron_isaac`，覆盖导入 checkpoint、LIBERO eval、微调与 SO100/SO101/YAM 真机 rollout。
- **沉淀到 wiki：** [Perceptron Isaac 0.5](../../wiki/entities/perceptron-isaac-05.md)

## 开源状态（README + policy 指南核查，2026-08-27）

| 项 | 状态 |
|----|------|
| 根仓 | 公开；Apache 2.0 |
| 训练 / 推理 | 在 `isaac/lerobot` 子模块；`uv sync --extra perceptron_isaac` |
| 权重 | Hub 页 **COMING SOON** |
| mHarmony / TensorStream | **单独维护，未 vendor**；`perceptron_isaac` extra **尚未声明** 其可发布运行时 → 干净安装 **不能** 完成渲染/训练/推理 |
| CUDA extra | `perceptron_isaac_cuda`（Linux；`flash-linear-attention` + `causal-conv1d`）；生产 H100 动作对齐运行时钉 **PyTorch 2.10.0+cu128**，与仓默认 2.11 CPU 开发环境不同 |
| 结论 | **部分开源**（代码入口完整，运行时依赖与权重未齐） |

## 目录与入口（README）

```text
isaac/
├── lerobot/       # Isaac 0.5 训练与推理集成
├── LICENSE
└── README.md
```

```bash
git clone --recurse-submodules https://github.com/perceptron-ai-inc/isaac.git
cd isaac/lerobot
uv sync --locked --extra perceptron_isaac
uv run python -c "from lerobot.policies.perceptron_isaac import PerceptronIsaacPolicy"
```

文档（子模块内）：

- Fine-tuning：`lerobot/docs/PERCEPTRON_ISAAC_FINETUNING_GUIDE.md`
- Policy / runtime：`lerobot/src/lerobot/policies/perceptron_isaac/README.md`
- Provenance / 包边界：`lerobot/docs/PERCEPTRON_ISAAC_PROVENANCE.md`

## 可运行表面（policy README 归纳）

- `PerceptronIsaacConfig` / `PerceptronIsaacPolicy`；NTP、FAST、flow-matching 原生损失。
- `lerobot-isaac-import`：一次性把原始 HF export 打成可部署包（契约 JSON + shard 流式转换 + Qwen3.5 RMSNorm 校正 + FAST processor）。
- `lerobot-eval`：LIBERO（每策略实例 **batch_size=1**）。
- `lerobot-train`：标准 v0.6 trainer；策略自有联合目标。
- `lerobot-rollout`：同步 SO100 / SO101 / 双臂 YAM；YAM 默认 torque-off，运动需显式安全开关。
- `lerobot-isaac-parity`：与 Genesis oracle 的损失/梯度对齐检查。

**关键工程边界：** mHarmony 与 TensorStream 不在 LeRobot 内；sibling checkout / 自定义 `PYTHONPATH` / 直接 `import genesis.data.mharmony` 被标为 **release blocker**。

## 对 wiki 的映射

- [perceptron-isaac-05](../../wiki/entities/perceptron-isaac-05.md)
- [lerobot](../../wiki/entities/lerobot.md)
- [vla](../../wiki/methods/vla.md)
