## [2026-11-22] ingest | sources/repos/cyberbeast-robot-mc.md — CyberBeast 关节电机全软件栈（5 项目 + 总览）

- 录入 robot_mc、OpenBootloader、JointSDK、JointROS、canable2-fw-n32 五个独立开源项目，附官方源码链接、控制流程与运行时序 Mermaid 图。
- 归档 Motor Wizard 官方手册调试入口；EXE 文件大小和现时下载状态未作事实认定。
- 形成 firmware 40 kHz FOC → CAN/CAN-FD C99 SDK → ROS 2 1 kHz tick 的分层认知，区分 CAN Bootloader Classic 500 kbit/s 升级链与 SLCAN USB2CAN 诊断链。
- 注册 CyberBeast 所属机构，链接力矩电机与人形硬件纵深路线，节点健康度由 GitHub CI 核验。
