# zhourui9813/DexRoam

> 来源归档

- **类型：** repo
- **链接：** <https://github.com/zhourui9813/DexRoam>
- **项目页：** <https://dexroam.github.io/>
- **许可证：** MIT
- **入库日期：** 2026-10-04
- **一句话说明：** 提供 Quest 3 + ZED Mini 第一视角全身示教采集，以及面向 Astribot / XHand 的人体动作对齐与时间重采样。

## 主要目录

| 路径 | 用途 |
|---|---|
| ego_wholebody_mocap | Quest / WebXR 与 ZED 同步采集、episode 管理和 HDF5 记录 |
| human_robot_alignment | 全身动作重定向、XHand 配置与时间重采样 |
| scripts | 轨迹可视化工具 |
| docs | 采集和对齐参考 |

## 开源状态

**已开源。** 官方仓库公开采集和对齐代码及安装、运行说明。策略训练代码见 [DexRoam-Policy-Training](./zhourui9813-dexroam-policy-training.md)；公开示例数据见 [HF 数据卡](https://huggingface.co/datasets/zhourui9813/DexRoam_Realworld_Data)。

## 对 wiki 的映射

- [论文实体页](../../wiki/entities/paper-dexroam-mobile-bimanual-manipulation.md)
- [项目页归档](../sites/dexroam-github-io.md)
