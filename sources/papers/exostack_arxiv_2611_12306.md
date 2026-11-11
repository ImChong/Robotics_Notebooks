# ExoStack: A Modular ROS 2 Framework for Wearable Robotics
- **原文：** https://arxiv.org/pdf/2611.12306
- **论文编号：** arXiv:2611.12306
- **项目文档：** https://debadrata.github.io/ExoStack_documentation/
- **代码：** https://github.com/debadrata/ExoStack
- **日期：** 2026-11-11（本次核对）
- **核对范围：** 6 页论文 PDF（架构、实验、图表与局限），官方 README、开源许可、安装步骤、ROS 2 setup.py 和主要节点代码，以及项目技术文档。
- **通读：** 已完成

## 核心摘录
1. ExoStack 对外骨骼设备抽象为 ESA（ESP32/micro-ROS 多传感/执行 hub）、ExoCore（ROS 2 同步/步态检测/轨迹规划/辅助 profile / cost / HIL 优化）、资源配置和插件、PASE 图形控制台四部分。
2. ESA 一块控制板各传感通道打包为单条 ROS 消息；主机按时间基准同步多个 hub。设备配置与控制插件由 YAML/文件声明而非大幅修改节点代码。
3. HIL 路径：CMA-ES/Bayesian 产生辅助候选→ profile builder→ motion planner→ 人体穿戴装置→同步传感与 gait/cost→为候选 ID 回传代价。部署路径可读已有 profile 查表，需区分。
4. 官方实验：8 个感知 hub 的 100Hz 实测约 99.5±0.018Hz；测试的 planner 200–300Hz 有至少 97% 达标，时延报告小于 18ms，平台包括 Jetson AGX Orin/Orin Nano/LattePanda Delta。均是作者实验配置的数字。
5. 在膝 SoftEXOwalk 与髋 flexion–extension 两类气动外骨骼上报告运行/个性化流程；论文未给出广泛的临床功效、安全性或长期康复疗效证据。
6. ROS 2 Humble / Ubuntu 22.04 / Python 3.10；ESP32 micro-ROS 需外部 agent 和实际硬件。正式人机系统必须由人负责安全上限与急停。

## 重要许可与版本核验
- [代码仓](../repos/exostack-official.md) 的 `LICENSE` 为 **PolyForm Noncommercial 1.0.0**，仅可按许可证允许的非商业用途使用，**不**是无商业限制的 OSI 开源许可。
- 文档 `LICENSE-DOCS.md` 为 **CC BY-NC 4.0**。
- 官方 README 的 arXiv 徽章/论文入口仍有 `[ARXIV_ID]` 占位符，本库以用户给出的真实 arXiv URL 与论文为准。
- [项目文档核验](../sites/exostack-documentation.md)；[主实体](../../wiki/entities/paper-exostack.md)。
