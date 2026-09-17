# DexCap（j96w/DexCap）

> 来源归档（ingest · 2026-09-17）

- **标题：** DexCap
- **类型：** repo
- **来源：** Stanford TML / SVL
- **链接：** <https://github.com/j96w/DexCap>
- **项目页：** <https://dex-cap.github.io/>
- **论文：** <https://arxiv.org/abs/2403.07788>
- **数据集：** <https://huggingface.co/datasets/chenwangj/DexCap-Data>
- **许可：** MIT
- **入库日期：** 2026-09-17
- **一句话说明：** RSS 2024 DexCap 官方实现：NUC 端 Rokoko 手套流 + 胸挂相机采集，Ubuntu 工作站处理/建 HDF5，robomimic 训练点云 Diffusion Policy。
- **开源状态：** **已开源** — `STEP1_collect_data`、`STEP2` 处理脚本、`STEP3_train_policy/robomimic` 训练配置与文档齐全；硬件 BOM 见项目页与 2024-08 Google Doc。
- **沉淀到 wiki：** [`paper-notebook-dexcap-scalable-and-portable-mocap-data-collecti.md`](../../wiki/entities/paper-notebook-dexcap-scalable-and-portable-mocap-data-collecti.md)

## 仓库结构（复现入口）

| 阶段 | 路径 / 命令 | 说明 |
|------|-------------|------|
| NUC 环境 | `install/env_nuc_windows.yml` | Windows + Rokoko Studio + Anaconda |
| 工作站环境 | `install/env_ws_requirements.txt` + `STEP3_train_policy` editable install | Python 3.8 |
| 采集 | `STEP1_collect_data/redis_glove_server.py` + `data_recording.py -s --store_hand` | 手套 JSON 流 + 多相机帧 |
| 可视化/标定 | `replay_human_traj_vis.py`、`transform_to_robot_table.py` | SLAM 漂移修正、对齐机器人桌面系 |
| 切分 demo | `demo_clipping_3d.py` | 长 episode → 任务片段 |
| 建数据集 | `demo_create_hdf5.py` | PyBullet 指尖 IK → robomimic HDF5 |
| 训练 | `STEP3_train_policy/robomimic/scripts/train.py --config training_config/*.json` | 点云 Diffusion Policy，46 维动作 |

## 依赖栈（README 致谢）

- 策略训练：[robomimic](https://github.com/ARISE-Initiative/robomimic)、[Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
- 机械臂：[Deoxys](https://github.com/UT-Austin-RPL/deoxys_control)
- LEAP Hand：[LEAP_Hand_API](https://github.com/leap-hand/LEAP_Hand_API)

## 对 wiki 的映射

- 项目页：[`dexcap.md`](../sites/dexcap.md)
- 论文来源：[`humanoid_pnb_dexcap-scalable-and-portable-mocap-data-collecti.md`](../papers/humanoid_pnb_dexcap-scalable-and-portable-mocap-data-collecti.md)
- 灵巧数据采集指南：[`dexterous-data-collection-guide.md`](../../wiki/queries/dexterous-data-collection-guide.md)
