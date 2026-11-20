# Lyapunov 机器人 3D Agent 工作台官方代码

- **类型：** repo / desktop robotics workbench
- **代码：** https://github.com/ManfredCh/Lyapunov
- **项目页：** https://vorynel.com/lyapunov/
- **手册：** https://vorynel.com/lyapunov/manual.html
- **版本说明：** https://github.com/ManfredCh/Lyapunov/blob/main/docs/releases/0.2.5.md
- **核查日期：** 2026-11-20
- **核对范围：** 官方 README 的部署与操作入口、0.2.5 发布说明、https://vorynel.com/lyapunov/manual.html 手册第 0–15 节及附录 A–C（Scene、Camera/Capture/Recording、RGB-D、标定、world ready、导出、故障排查）并对照仓库结构；未部署/逐个审计全部源码模块
- **通读：** 已完成

## 原文记录

项目由 Vorynel 维护，Apache-2.0。README 描述 LLM-first 3D 桌面工作区（对话、代码、文件、三维 Viewer、物理引擎）。Linux x64 默认版本 0.2.5 及 MuJoCo 环境；Isaac Sim/Newton 为可选。推荐先 `prepare` → `initialize` → `ready` → 显式执行动作，再观察 world state、相机和数据导出；可见模型并不等于物理已就绪。Guest 模式无默认模型，需自己配置外部供应商；在线模型/积分与本地运行不可混为一谈。0.2.5 原生场景、文件、终端标签与物理引擎碰撞展示更新。**手册重点核对：** Camera 是绑定真实 robot body 的持久机位，Capture 是自由/固定机位的单帧或同物理步 RGB-D 采集，Recording 仅对特定可用的命名机位录制；相机 Depth 是光轴 Z、LiDAR range 是射线欧氏距离，不能混用。Scene revision、world generation/step、camera readback 与采集 frame ID 需一致；无有效深度时应拒绝 3D 标注而不是补值。官方手册中部分 Linux 固定包校验例子、版本核验字段仍指向 **0.2.0-alpha.1**，虽然页眉称 0.2.5：*当前下载版本和 SHA 以正式 0.2.5 release note/manifest 为准，不复用旧 alpha 校验值*。这里的“通读”指已列出的 README/手册/发行说明，不代表逐行审计整个大型 monorepo，也不代表实际安装通过。

## 对 wiki 的映射

- [独立详情节点](../../wiki/entities/lyapunov-vorynel.md)
