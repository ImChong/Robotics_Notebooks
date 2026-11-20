# Lyapunov 机器人 3D Agent 工作台官方代码

- **类型：** repo / desktop robotics workbench
- **代码：** https://github.com/ManfredCh/Lyapunov
- **项目页：** https://vorynel.com/lyapunov/
- **手册：** https://vorynel.com/lyapunov/manual.html
- **版本说明：** https://github.com/ManfredCh/Lyapunov/blob/main/docs/releases/0.2.5.md
- **核查日期：** 2026-11-20
- **核对范围：** README 的安装、版本、产品结构、robot/camera/physics 等章节与 0.2.5 发布说明；完整手册及源码未通读
- **通读：** 未完成

## 原文记录

项目由 Vorynel 维护，Apache-2.0。README 描述 LLM-first 3D 桌面工作区（对话、代码、文件、三维 Viewer、物理引擎）。Linux x64 默认版本 0.2.5 及 MuJoCo 环境；Isaac Sim/Newton 为可选。推荐先 `prepare` → `initialize` → `ready` → 显式执行动作，再观察 world state、相机和数据导出；可见模型并不等于物理已就绪。Guest 模式无默认模型，需自己配置外部供应商；在线模型/积分与本地运行不可混为一谈。0.2.5 原生场景、文件、终端标签与物理引擎碰撞展示更新。**未完成对主仓所有源码包与完整手册的审计**，不写“软件已本机验证”。

## 对 wiki 的映射

- [独立详情节点](../../wiki/entities/lyapunov-vorynel.md)
