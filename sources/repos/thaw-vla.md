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
- **复核日期：** 2026-11-12
- **核对范围：** 已阅读官方 README、`scripts/01_precompute_teacher.sh`、`scripts/02_train.sh`、`scripts/03_eval_libero.sh`、`tools/cosmos3_precompute_targets.py`、`starVLA/model/modules/distill/fastwam_repa.py`、`starVLA/dataloader/gr00t_lerobot/fastwam_cache.py` 的主要实现及运行路径；未实际执行训练。
- **通读：** 已完成
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
- **HF 权重：** `termanteus/THAW-VLA-Qwen3.5-0.8B-{LIBERO,Robocasa-GR1}` 页面可公开浏览到 `final_model.pt`（2026-11-12）；官方 README 仍写 `private`，属未同步的历史描述。实际下载/推理未测试

## 代码运行核查补充

- `01_precompute_teacher.sh` 根据 `libero` / `gr1` 选择配置和相机数，按 GPU 分片运行特征提取，逐进程检查退出码和 cache `meta.json`。
- `cosmos3_precompute_targets.py` 从理解塔指定层取 image hidden states，按相机池化并打包缓存。
- `fastwam_cache.py` 定义 `targets.f16.memmap`、`valid.npy`、`meta.json` 和 `steps_hash`，用于检查训练数据排序一致性。
- `fastwam_repa.py` 通过 `RepaProjector` 双层 MLP、camera pooling 和 valid mask 计算加权 cosine alignment loss；该模块只用于训练。
- `02_train.sh` 用 Accelerate / DeepSpeed ZeRO-2 启动 `starVLA/training/train_starvla.py`；`03_eval_libero.sh` 启动套件评测并汇总日志。
- **验证边界：** 本次为静态阅读源码，没有运行 CUDA 预计算、GPU 训练、仿真或真机实验。

## 对 wiki 的映射

- [paper-thaw-vla](../../wiki/entities/paper-thaw-vla.md) — 源码运行时序图对齐本 README
