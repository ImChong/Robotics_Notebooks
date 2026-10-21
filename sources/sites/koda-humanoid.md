# KODA 官方项目页

> 来源归档（ingest）

- **标题：** KODA: Kinematics–Observation Data Alignment for Humanoid Learning from Human Egocentric Demonstration
- **类型：** official project page
- **项目页：** <https://koda-humanoid.github.io/koda/>
- **论文：** <https://arxiv.org/abs/2610.21152>
- **代码：** <https://github.com/Gamma-Robot/KODA>
- **入库日期：** 2026-10-17（发布状态复核：2026-10-20）
- **一句话说明：** 官方页展示以共享控制时钟对齐人类第一视角、机器人全身轨迹、合成机载视角和控制器兼容动作标签的转换框架。

## 开源核查（2026-10-20）

- 官方仓库现提供 M1/M2：RGB-D/MCAP 转换、手部估计与重定向、接触感知修正、SCO 跟踪、手臂细化，以及机器人机载视角合成；README 含安装步骤和 `koda run` CLI。
- M4 部分开放：公开两个示例 MCAP 与微调的 ScaleBFM tracking checkpoint，资产下载入口为 <https://huggingface.co/datasets/TeleEmbodied/koda-assets>。
- M3 控制器动作标签、motion-token 编码、控制器对齐数据导出，以及 M5 训练配置和模型细节仍标为计划中；因此属于**部分开源**，目前不能复现完整论文训练流程。
- README 未声明对集成代码及各第三方组件统一适用的项目级许可；模型、机器人网格和依赖需分别检查其条款。

## 对 wiki 的映射

- [KODA 论文来源](../papers/koda_arxiv_2610_21152.md)
- [官方仓库归档](../repos/gamma-robot-koda.md)
- [KODA 实体页](../../wiki/entities/paper-koda-egocentric-vla.md)
