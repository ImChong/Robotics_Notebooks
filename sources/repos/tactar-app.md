# TactAR_APP（AR 触觉遥操作应用）

> 来源归档（ingest；依据公开 README 摘要）

- **类型：** repo / Unity application
- **官方仓库：** <https://github.com/xiaoxiaoxh/TactAR_APP>
- **配套论文：** <https://arxiv.org/abs/2503.02881>
- **入库日期：** 2026-10-05
- **一句话说明：** RDP 论文配套的 TactAR 触觉/力遥操作 APP 源码，支持 Quest 3 AR 呈现及多路相机/触觉流。

## README 所示内容

- 实时将 tactile/force sensor 的三维形变/力场渲染在增强现实中，并附着于机器人末端的虚拟坐标。
- 支持多个 RGB 相机与光学触觉相机视频流，供接触丰富任务遥操作和采数时观察。
- README 面向 Meta Quest 3；提供预构建 APK release 入口、Unity 源码构建指南、标定说明和用户指南。
- Quest、工作站和机器人需在同一 LAN；机器人端硬件与依赖参见 RDP 仓库。

## 边界

- 这是遥操作/显示应用，不是 RDP 学习策略本身，也不单独包含整套机器人控制栈。
- 设备兼容、构建环境与许可应以仓库当前 README、Docs 和 LICENSE 为准。

## Wiki 映射

- [RDP 实体页](../../wiki/entities/paper-sa-2503-02881-reactive-diffusion-policy-slow-fast-visual-tacti.md)
- [RDP 代码仓归档](./reactive_diffusion_policy.md)
