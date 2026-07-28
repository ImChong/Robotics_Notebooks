# DexUMI（real-stanford/DexUMI）

> 来源归档

- **标题：** DexUMI
- **类型：** repo
- **来源：** Stanford / Columbia / J.P. Morgan AI Research / CMU / NVIDIA
- **链接：** <https://github.com/real-stanford/DexUMI>
- **项目页：** <https://dex-umi.github.io/>
- **论文：** <https://arxiv.org/abs/2505.21864>
- **数据：** <https://umi-data.github.io/>
- **许可：** MIT
- **入库日期：** 2026-07-28
- **一句话说明：** 完整开放外骨骼设计优化、45 FPS 示范采集、SAM2/ProPainter 视觉适配、数据打包、Diffusion Policy 训练与 XHand/Inspire 真机评测。
- **开源状态：** **已开源**。
- **沉淀到 wiki：** [`paper-notebook-dexumi-using-human-hand-as-the-universal-manipul.md`](../../wiki/entities/paper-notebook-dexumi-using-human-hand-as-the-universal-manipul.md)

## 仓库概况（2026-07-28）

| 阶段 | README 入口 |
|------|-------------|
| 环境 | `mamba env create -f environment.yml` |
| 采集 | `real_script/data_collection/record_exoskeleton.py` |
| 回放/处理 | `real_script/data_generation_pipeline/process.sh` |
| 视觉适配 | `render_all_dataset.py`（SAM2 + ProPainter + robot render） |
| 数据生成 | `6_generate_dataset.py` |
| 训练 | `accelerate launch .../train_diffusion_policy.py` |
| 部署 | `open_server.py` + `eval_xhand.py` / `eval_inspire.py` |

复现依赖外部 `sam2`、`ProPainter`、`record3D`、SAM2 checkpoint、ARKit iPhone 与目标手硬件；仓库提供样例数据用于先验证离线数据生成链。

## 对 wiki 的映射

- 项目页：[`dexumi.md`](../sites/dexumi.md)
- 论文来源：[`humanoid_pnb_dexumi-using-human-hand-as-the-universal-manipul.md`](../papers/humanoid_pnb_dexumi-using-human-hand-as-the-universal-manipul.md)
- 遥操作路线：[`depth-teleoperation.md`](../../roadmap/depth-teleoperation.md)
