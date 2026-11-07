# RoboSet 官方项目与数据页（RoboSet Dataset）

- **类型：** 项目页 / 数据集
- **项目：** RoboSet — Large-scale real-world multi-task robotics dataset
- **原始地址：** https://robopen.github.io/roboset/
- **核对日期：** 2026-11-07
- **核对范围：** 官网完整公开文本（Dataset composition、Source of dataset、Camera viewpoints、Different Tasks、Python starting code、数据分项下载与许可链接）。未下载数据或复现策略训练。
- **通读：** 已完成
- **代码：** 项目页仅提供 `h5py` HDF5 最小读取样例；未在该页核实独立训练/推理 GitHub 仓库。
- **数据：** 官网按 Teleoperation、Kinesthetic、Autonomous 分项展示入口；本次未实测可下载性。

## 原文可核对事实

1. **领域与场景**：真实世界厨房场景的多任务操作；活动分为若干语言描述任务，同一活动可由多项技能串联。
2. **数据采集**：一部分为操作者使用 Oculus Quest 2 遥操作；另一部分为变换环境配置后执行动作示教轨迹回放。
3. **规模**：官方页宣称 28,500 条轨迹，其中 9,500 遥操作、19,000 动作示教回放。页面也出现 Autonomous 分项入口，但没有给出可与前述分项相加的独立轨迹数，**不可额外累加**。
4. **多视角**：每帧包含四个不同相机视角，主要用于视角多样性研究。
5. **存储读取**：项目页示例采用 `h5py.File(path, 'r')`，按 `Trial0/data`、`Trial0/derived`、`Trial0/config` 等分组访问。
6. **许可**：项目页底部标注 MIT License；实际下载包可能涉及单独条款，本次未逐文件核实。

## 原文可支持的最小流程

人类遥操作 / 动作轨迹回放 → 厨房场景数据采集 → 四路视觉与动作轨迹 → HDF5 试次数据 → Python 读取 → 下游多任务策略研究。

## 局限与尚未核对

- 官网呈现采集方式与数据格式入口，但没有足以推导具体策略训练及部署调用关系的官方源码入口，因此**不编造源码运行时序**。
- 未读取任何 HDF5 样本，字段名称及关节/动作单位要以下载包实际 schema 为准。
- 此文件记录官网，不代表论文/补充材料已通读。

## 关联

- [跨项目一手入口核对](../datasets/robot-manipulation-primary-sources-2026-11-07.md)
- [RoboSet 实体](../../wiki/entities/roboset-dataset.md)
