# uni-vta/UVTA（Unified Visual-Tactile-Action）

- **标题：** UVTA: Unified Visual-Tactile-Action Modeling from Human Demonstrations for Dexterous Manipulation
- **类型：** repo / training + evaluation + robot rollout
- **仓库：** <https://github.com/uni-vta/UVTA>
- **维护方：** uni-vta
- **许可证：** MIT（仓库 README）
- **配套论文：** <https://arxiv.org/abs/2609.34182>
- **项目页：** <https://uni-vta.github.io/>
- **数据集：** <https://huggingface.co/datasets/Chopper233/UVTA>
- **入库日期：** 2026-10-16
- **一句话说明：** UVTA 官方实现，提供视触觉条件扩散策略训练、离线动作质量评估与机器人 rollout；公开数据集独立托管于 Hugging Face。

## 开源范围

仓库 README 提供的可运行组件包括：

- uvta/：策略、数据集读取、归一化和推理。
- configs/、scripts/：任务训练配置与入口；scripts/train.py 可训练五项任务。
- scripts/eval_action_quality.py：对留出的机器人轨迹评估腕部 RMS 误差与动作块粗糙度。
- cet/：机器人 rollout、观测时间同步与轨迹插值。
- deform_to_human/：触觉变形到人手 FSR 区域的转换与映射。
- assets/、third_party/：腕部相机环与重定向工具资源。

训练默认 300 epochs、batch size 250，学习率采用 cosine schedule；灯泡任务配置为 500 epochs。默认动作用腕部 RGB 与 20-D 指尖触觉，预测 16-step、31-D 动作块（9-D 相对腕姿 + 22 维手指关节）。

## 数据与硬件边界

- Hugging Face 数据集以 28 个 Zarr v2 tar shards 发布，合计约 482 GB；应按任务挑选下载并保留 Zarr 元数据与目录层级。
- 每个任务约 1,000 条人类 episode 和 150 条机器人 episode；flip_book 有 149 条机器人 episode。
- cet/ 提供 rollout 逻辑，但目标机器人 RobotEnv、手臂运动学及厂商 SDK 不包含在仓库，需要适配；真机执行前应核验时间同步、标定与关节运动限位。
- 数据集许可未在本次来源核查中独立确认；不要把代码 MIT 许可推及数据集。

## 最小复现入口

环境要求 Python 3.10 或更新版本、PyTorch 2.5.1 与 torchvision 0.20.1（按 CUDA 版本安装）。安装仓库依赖并设置 UVTA_ROOT 与 UVTA_DATA_ROOT 后，可用下列入口开始：

    python scripts/train.py task=switch

初次验证可限制加载的数据量（README 建议 dataset.max_episode=[1,1]）；完整数据加载会占用大量磁盘与内存。离线评估入口为 scripts/eval_action_quality.py。真机入口是 cet/relative_policy_rollout.py，运行前需提供匹配的机器人接口。

## Robotics_Notebooks 映射

- 论文：[sources/papers/uvta_arxiv_2609_34182.md](../papers/uvta_arxiv_2609_34182.md)
- 项目页：[sources/sites/uvta.md](../sites/uvta.md)
- Wiki 实体：[wiki/entities/paper-uvta-human-tactile-dexterous-manipulation.md](../../wiki/entities/paper-uvta-human-tactile-dexterous-manipulation.md)

## 一手来源

- 仓库首页 / README：<https://github.com/uni-vta/UVTA>
- 数据集：<https://huggingface.co/datasets/Chopper233/UVTA>
- 项目页：<https://uni-vta.github.io/>
- 论文：<https://arxiv.org/abs/2609.34182>
