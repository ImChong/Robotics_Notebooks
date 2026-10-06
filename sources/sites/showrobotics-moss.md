# MOSS 官方项目页（Show Robotics）

> 来源归档

- **标题：** MOSS · litter-picking rover
- **类型：** site / 项目页
- **来源：** Show Robotics
- **链接：** <https://www.showrobotics.ai/moss/>
- **入库日期：** 2026-10-06
- **核查日期：** 2026-10-06
- **一句话说明：** MOSS 项目展示与构建进度页，介绍履带式拾取机器人、硬件组成、实机进展及 Jev 仿真演示入口。
- **代码：** <https://github.com/metrox-eth/moss>
- **演示：** <https://www.showrobotics.ai/moss-jev/>
- **沉淀到 wiki：** [MOSS 实体页](../../wiki/entities/moss.md)

## 页面核查

项目页介绍的 MOSS 是以 3D 打印件为主的履带移动底盘，带垃圾箱与 SO-101 衍生机械臂。页面列出 Jetson Orin Nano Super、RealSense D455、ESP32-S3 电机控制、双编码器电机及机械臂/夹爪，并链接主仓库和 Jev 演示。

项目页称代码、设计文件与数据开放，但**设计文件发布状态与仓库 README 的最新说明不一致**：截至核查日，主仓库仍称 CAD 正在结合实机反馈定版、STL / STEP 将在设计冻结后发布。因此知识页按仓库当前状态记为「软件与固件已公开；硬件 CAD 尚待正式发布」，不把项目页较早的概括文案当作下载凭据。

## 演示边界

项目页将 MOSS × Jev 描述为 MuJoCo 中生成并录制、再在浏览器播放的物理轨迹；演示页明确显示「LIVE API OFF · REPLAY ONLY」，并写明 Jev decisions 为记录结果、没有 API 请求。详见 [演示页](https://www.showrobotics.ai/moss-jev/) 与 [仿真仓库归档](../repos/metrox-eth-moss-jev.md)。

## 对应仓库

- [MOSS 主仓库](../repos/metrox-eth-moss.md)
- [MOSS × Jev 仿真仓库](../repos/metrox-eth-moss-jev.md)
