# IAIE-PhysicalAI/Bimanual-Teleoperation

> 来源归档（ingest）

- **类型：** repo
- **官方仓库：** <https://github.com/IAIE-PhysicalAI/Bimanual-Teleoperation>
- **论文：** <https://github.com/chaotan2003/human-like-redundancy/blob/main/paper.pdf>
- **项目页：** <https://chaotan2003.github.io/human-like-redundancy/>
- **入库日期：** 2026-10-12
- **一句话说明：** PICO 双臂遥操作参考管线：SWGR 重定向、AEAC 肘偏好、J-PARSE 速度 IK 与 Viser 可视化。

## README 运行入口

- 环境：Python 3.10 / 3.11；`bash setup_conda.sh --install`
- 启动：`python scripts/simulation/teleop_jaka_k1.py`
- 测试：`python -m unittest discover tests`
- 组件：`BaseTeleopController` 读取 XR / 身体跟踪、做输入健康检查、SWGR 映射和 engagement latch；`ViserJparseController` 负责目标滤波、AEAC/J-PARSE IK、插值与可视化。
- 频率：控制约 60 Hz；插值开启时约 200 Hz；Viser 约 30 Hz。

## 实现边界

README 描述 JAKA K1 机器人模型的 Viser 集成；未列物理仿真后端或真实硬件命令发布通道。因此该仓适合理解/运行可视化算法栈，不等于具备安全真机控制接口。

## 对 wiki 的映射

- [实体页](../../wiki/entities/paper-human-like-redundancy-resolution.md)
- [论文归档](../papers/human-like-redundancy-resolution.md)
- [项目页归档](../sites/human-like-redundancy.md)
