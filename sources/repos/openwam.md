# OpenWAM（OpenWAM-Official/OpenWAM）

> 来源归档

- **标题：** OpenWAM
- **类型：** repo
- **来源：** NUS / Tsinghua / PKU / HKU / ZJU / CUHK / SJTU 等
- **链接：** <https://github.com/OpenWAM-Official/OpenWAM>
- **论文：** <https://arxiv.org/abs/2609.07398>
- **项目页：** <https://openwam-official.github.io/>
- **权重集合：** <https://huggingface.co/OpenWAM>
- **许可：** 见仓库 LICENSE（README 末段）
- **入库日期：** 2026-09-09
- **一句话说明：** 模块化 WAM 研究栈：Hydra 配置、训练 / 部署 / 多基准 WebSocket 评测、OpenWAM-α 与 Study 检查点下载器。
- **沉淀到 wiki：** [`wiki/entities/paper-openwam.md`](../../wiki/entities/paper-openwam.md)

---

## 仓库入口（README）

| 组件 | 说明 |
|------|------|
| 安装 | `pip install -e .`（Python ≥3.10；推荐 torch 2.7.1 + cu128） |
| 资产下载 | `scripts/download_assets/download_video_backbone.py`、`download_benchmark_data.py`、`download_openwam_checkpoints.py` |
| 训练 | `bash scripts/train.sh dataloader=libero model=dual_system model/video_backbone=wan22_ti2v_5b …` |
| 微调 OpenWAM-α | `training.finetune_ckpt_path=<foundation_ckpt_dir>` |
| 部署 | `bash scripts/deploy.sh <ckpt_dir_path>` → WebSocket `ws://127.0.0.1:8848` |
| 推理自检 | `scripts/inference_test/inference_single_test.py`、`inference_continuous_test.py` |
| LIBERO 评测 | `benchmarks/libero/` WebSocket 客户端 |
| RoboTwin / LIBERO-plus / RoboCasa365 / VLABench / EBench / RoboDojo | 各 `benchmarks/<name>/` |
| 参考硬件 | 推荐 8×80GB GPU 训练 Wan2.2-5B 级骨干 |

## 架构变体（README Support Status）

| 族 | 变体 | 说明 |
|----|------|------|
| `single_system` | `vanilla` / `moe` | 单 DiT 承载 video + action + state |
| `dual_system` | `joint_self_attn` | 分离 Video DiT + ActionDiT，层间 mixed self-attention（OpenWAM-α 默认） |
| `dual_system` | `joint_cross_attn` / `idm` | 跨注意力桥接或 IDM 两阶段 |
| `tri_system` | `joint_self_attn` | 追加冻结 VLM 理解专家 |

## 开源边界（截至 2026-09-09）

- **已开源**：训练、部署、8+ 基准评测客户端、资产与检查点下载脚本、用法文档 `assets/openwam_usage_docs/`。
- **权重**：HF `OpenWAM`（46 模型，含 OpenWAM-α Foundation 与下游微调）。
- **外部依赖**：RoboDojo 真机评测经 [XPolicyLab](https://github.com/XPolicyLab/XPolicyLab)；部分基准需单独拉环境。
- **计划中**：SimplerEnv、Calvin 等标注为 Planned。
