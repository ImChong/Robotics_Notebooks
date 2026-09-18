# LaWAM（RLinf/LaWAM）

> 来源归档

- **标题：** LaWAM
- **类型：** repo
- **组织：** RLinf
- **链接：** <https://github.com/RLinf/LaWAM>
- **论文：** <https://arxiv.org/abs/2606.15768>
- **项目页：** <https://rlinf.github.io/LaWAM/>
- **HF 集合：** <https://huggingface.co/collections/jialei02/lawam-checkpoints>
- **LeRobot SFT 权重：** <https://huggingface.co/jialei02/lawam-libero-sft-lerobot>
- **LeRobot 数据集：** <https://huggingface.co/datasets/jialei02/libero_merged_no_noops_20hz>
- **许可：** MIT（基于 StarVLA）
- **入库日期：** 2026-09-18
- **一句话说明：** 潜空间 WAM：LaWM + Alternate-DiT VLA；LIBERO/RoboTwin 训练与 auto_eval 脚本；LeRobot 原生对接。
- **沉淀到 wiki：** [`wiki/entities/paper-lawam.md`](../../wiki/entities/paper-lawam.md)

---

## 仓库入口（README）

| 组件 | 说明 |
|------|------|
| 核心模型 | `starVLA/` — 训练循环、配置、LaWAM 框架 |
| LaWM | `latent_action_model/` — LAM / LaWM 与 DINOv3 特征 |
| 部署 | `deployment/` — 评测用 policy server |
| LIBERO 评测 | `examples/LIBERO/eval_files/auto_eval_scripts/run_libero_benchmark.sh` |
| RoboTwin 评测 | `examples/Robotwin/eval_files/auto_eval_scripts/auto_eval_robotwin.sh` |
| 单节点训练 | `train_lawam.sh` + `starVLA/config/training/train_*.yaml` |
| 多节点训练 | `train_lawam_distributed.sh` |
| 依赖 VLM | Qwen3-VL-2B-Instruct |
| 依赖视觉编码 | DINOv3 ViT-B/16 + `jialei02/lawam_lam` |

## 开源边界（截至 2026-09-18）

- **已开源：** 训练、推理、LIBERO/RoboTwin sweep、HF 检查点与 LeRobot 格式数据。
- **LeRobot：** 官方文档页 `docs/source/lawam.mdx` 已收录。
- **真机：** 项目页有 Franka / Quanta X1 结果；主复现路径为仿真 benchmark + HF 权重。
