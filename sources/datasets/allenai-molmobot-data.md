# MolmoBot-Data：Ai2 仿真机器人专家操作数据集

- **官方数据集：** https://huggingface.co/datasets/allenai/molmobot-data
- **官方数据说明：** https://huggingface.co/datasets/allenai/molmobot-data/blob/main/README.md
- **原论文：** https://arxiv.org/html/2603.16861v3
- **数据生成源码：** https://github.com/allenai/molmospaces
- **模型训练与部署源码：** https://github.com/allenai/MolmoBot
- **核查日期：** 2026-11-17
- **核对范围：** MolmoBot 原论文 v3 §§3、5.3 与附录 A.3–A.7、Table 3/4；HF 数据集 README 中八个官方配置和扩展配置、bulk_download.py 下载方式及过滤/统计入口；官方 MolmoBot 仓 README 的后处理说明。
- **通读：** 已完成
- **一句话说明：** 采用 MolmoSpaces 场景程序生成 1.7M 条、5,704 小时的 Franka FR3 与 Rainbow RB-Y1 仿真专家数据，覆盖刚体抓放与可动关节物体操作。

## 核实结果

- 论文 v3（2026-11-08）Table 3/4 明列 **1.7M** 轨迹、**5,704 小时**、**94.2k** 独立环境、八种数据配置、两种机器人。
- 八种训练数据配置：`DoorOpeningDataGenConfig`、`RBY1OpenDataGenConfig`、`RBY1PickDataGenConfig`、`FrankaPickOmniCamConfig`、`RBY1PickAndPlaceDataGenConfig`、`FrankaPickAndPlaceOmniCamConfig`、`FrankaPickAndPlaceColorOmniCamConfig`、`FrankaPickAndPlaceNextToOmniCamConfig`。
- HF README 另列 `FrankaPickAndPlaceOmniCamConfig_ObjectBackfill` 扩展数据，明确不属于发布模型的训练数据。
- 下载需用 `bulk_download.py`；部分 shard 为压缩包索引，下载后需 `validate_trajectories.py` 创建有效轨迹索引，再由 `calculate_stats.py` 统计动作/qpos 分布。
- **任务数口径：** 数据集八种配置包含机器人与环境版本区分；论文方法 §3.2 的任务语义是六项（Pick、PnP Into/Next-To/Color、Open、Open-door），不能推断八类完全不同的语义技能。
- **许可：** 当前 HF 数据集页标识 `odc-by`，实际再分发资产与场景还需检查数据卡及依赖资源的授权条款。

## 对知识库的映射

- [论文通读归档](../papers/molmobot_arxiv_2603_16861.md)
- [论文实体](../../wiki/entities/paper-molmobot.md)
- [数据集节点](../../wiki/entities/dataset-molmobot-data.md)
