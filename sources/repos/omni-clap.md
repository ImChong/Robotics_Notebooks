# CLAP（omni-CLAP/clap）

> 来源归档

- **标题：** CLAP
- **类型：** repo
- **来源：** Princeton IRoM / omni-CLAP
- **链接：** <https://github.com/omni-CLAP/clap>
- **论文：** <https://arxiv.org/abs/2608.27406>
- **项目页：** <https://omni-clap.github.io/>
- **权重：** <https://huggingface.co/omni-CLAP/CLAP>
- **许可：** MIT
- **入库日期：** 2026-08-29
- **一句话说明：** 跨本体动作条件视频 WM 全栈：训练、评测、回放、键盘遥操作、policy-in-the-loop 部署，以及双臂 YAM / G1 适配入口。
- **沉淀到 wiki：** [`wiki/entities/paper-clap-cross-embodiment.md`](../../wiki/entities/paper-clap-cross-embodiment.md)

---

## 仓库入口（README）

| 组件 | 说明 |
|------|------|
| 安装 | `uv venv` + `uv pip install -e .` 或 `uv sync`；Python ≥3.9（in-process openpi 用 3.11） |
| 权重 | `hf download omni-CLAP/CLAP`；默认按需落到 `model_ckpt/` |
| 回放 | `bash examples/getting_started/replay.sh`（`clap-rollout-replay`） |
| 键盘遥操作 | `bash examples/getting_started/teleop.sh`（`clap-teleop`） |
| 闭环部署 | `bash examples/getting_started/deploy.sh` / `deploy_yam.sh`（`clap-rollout-deploy`；需 openpi 或 MolmoAct-2） |
| G1 适配 | `clap-preprocess-g1` → `examples/adapt/adapt_g1_humanoid.sh` → `clap-eval` |
| 训练 | `clap-train` + `examples/slurm/*.slurm`（跨本体约 8×80GB GPU） |
| 样例数据 | `sample_data/oxe/`（约 232 MB：droid / bridge / taco_play / bimanual_yam / g1_humanoid） |

## 检查点族（HF `omni-CLAP/CLAP`）

| 族 | 代表 | 条件 |
|----|------|------|
| 跨本体 | `clap-curr` / `clap-ee` / `clap-lam` / `clap-lang` | EE / LAM / language |
| 单平台后训练 | `clap-*-droid` / `clap-*-bridge` | 一律 EE |
| 新本体适配 | `adapt-yam`（14-D）、`adapt-g1`（26-D） | 关节空间 |

## 开源边界（截至 2026-08-29）

- **已开源**：训练、评测、回放、遥操作、部署与适配脚本可跑。
- **数据**：OXE / EgoDex 需自备；仓内只带样例 episode。
- **策略后端**：openpi / MolmoAct-2 为可选依赖，不随 CLAP 权重发布。
