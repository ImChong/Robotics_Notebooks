# trungdt880/THAW-VLA

> 来源归档（repo）

- **名称：** THAW-VLA
- **类型：** repo / vla / world-model-distillation / starvla
- **URL：** <https://github.com/trungdt880/THAW-VLA>
- **论文：** [arXiv:2609.24682](../papers/thaw_vla_arxiv_2609_24682.md)
- **项目页：** <https://thaw-vla.trung-dt.com/> — [`sources/sites/thaw-vla-trung-dt-com.md`](../sites/thaw-vla-trung-dt-com.md)
- **机构：** UW–Madison、UIUC
- **许可证：** MIT（基于 StarVLA MIT）
- **入库日期：** 2026-09-23
- **一句话说明：** THAW-VLA 官方实现：四 venv 隔离依赖；Cosmos3 特征预计算 → StarVLA QwenGR00T 蒸馏训练 → LIBERO/RoboCasa-GR1 评测与 websocket 部署。

## 运行入口（README）

| 步骤 | 脚本 / 模块 |
|------|-------------|
| 环境 | `./scripts/00_setup_env.sh`（+ `--teacher` / `--libero` / `--robocasa`） |
| Teacher 特征 | `./scripts/01_precompute_teacher.sh {libero\|gr1}` → `tools/cosmos3_precompute_targets.py` |
| 训练 | `./scripts/02_train.sh configs/{libero,gr1}_{distill,baseline}_qwen35_0p8b.yaml` |
| LIBERO 评测 | `./scripts/03_eval_libero.sh <checkpoint.pt>` |
| GR1 评测 | `./scripts/04_eval_gr1.sh <checkpoint.pt>` |
| 部署 | `deployment/` websocket policy server（评测 harness 共用） |

## 关键代码路径

| 路径 | 作用 |
|------|------|
| `starVLA/model/modules/distill/fastwam_repa.py` | cosine 对齐损失与 projector |
| `starVLA/dataloader/gr00t_lerobot/fastwam_cache.py` | 读取预计算 teacher cache |
| `tools/cosmos3_precompute_targets.py` | 冻结 Cosmos3-Nano 特征提取 |
| `configs/` | distill/baseline 成对配置（仅 `use_repa` 与 cache 差异） |
| `examples/` | StarVLA 数据准备（LIBERO LeRobot、RoboCasa GR1） |

## 开源边界（2026-09-23）

- **已有：** 完整训练/评测/部署管线、四环境 setup 脚本、config 与文档
- **需自备：** Cosmos3-Nano checkpoint（~33 GB）、LIBERO / RoboCasa 数据、Qwen3.5-0.8B base
- **HF 权重：** `termanteus/THAW-VLA-Qwen3.5-0.8B-{LIBERO,Robocasa-GR1}` — **private**，需 `hf auth` + 申请访问

## 对 wiki 的映射

- [paper-thaw-vla](../../wiki/entities/paper-thaw-vla.md) — 源码运行时序图对齐本 README
