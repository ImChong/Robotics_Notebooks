# OpenHLM-project / OpenHLM

> 来源归档（ingest）

- **标题：** OpenHLM — Whole-Body Humanoid Loco-Manipulation Recipe
- **类型：** repo
- **官方入口：** <https://github.com/OpenHLM-project/OpenHLM>
- **项目页：** <https://openhlm-project.github.io/>
- **论文：** <https://arxiv.org/abs/2606.22174>
- **数据集 / 权重：** HF `OpenHLM/OpenHLM-data` · `OpenHLM/OpenHLM-ckpts`
- **许可：** Apache-2.0
- **机构：** 清华大学；上海期智研究院；千寻智能
- **入库日期：** 2026-07-22
- **一句话说明：** 全身原生人形 VLA 全栈：G1 全身控制与采集（基于 GR00T-WBC 改）、HuMI 异构数据管线、openpi 系训练/服务/真机推理客户端。

## 仓库三分块（README）

| 目录 | 职责 |
|------|------|
| `src/GR00T-WholeBodyControl4OpenHLM`（或仓内同名块） | 机器人侧全身控制、VR 遥操作、同步采集、部署接口（对齐 NVlabs GR00T-WBC） |
| `src/HuMI4OpenHLM` | HuMI（人手/身体 tracker + 腕部相机）采集与共训数据转换 |
| `src/openpi4OpenHLM` | LeRobot 转换、norm stats、`train_pytorch.py`、`serve_policy.py`、推理文档 |

## 关键复现路径

1. **采集：** 按 README 装 GR00T-WBC 部署栈 + PICO VR；`scripts/` 下采集；HuMI 走 Vive tracker + GoPro 指南。
2. **训练：** `cd src/openpi4OpenHLM` → 装环境、拉 `pi05_base` → `convert_g1_data_to_lerobot*.py` → `compute_norm_stats.py` → `torchrun … scripts/train_pytorch.py <config>`。
3. **部署：** `serve_policy.py --env SONICG1 …`；机器人侧 `deploy_stream.sh` + `openpi-eval/main.py`（经 WBC 闭环）。

## 硬件要点（README）

- Unitree G1；腕部 ChangingTek 夹爪 + RealSense D405；头部 Unitree SV1-25；PICO4U（头显 + 手柄 + 腿部 tracker）
- HuMI：HTC Vive trackers（双夹爪/骨盆/双足）+ 腕部 GoPro

## 对 wiki 的映射

- [OpenHLM 实体页](../../wiki/entities/paper-loco-manip-161-154-openhlm.md)
- [项目页](../sites/openhlm-project-github-io.md)
- [GR00T-WholeBodyControl](./gr00t_wholebodycontrol.md) — 低层栈上游
