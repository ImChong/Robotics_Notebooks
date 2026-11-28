# Lyapunov 机器人 3D Agent 工作台官方代码

- **类型：** repo / desktop robotics workbench
- **代码：** https://github.com/ManfredCh/Lyapunov
- **项目页：** https://vorynel.com/lyapunov/
- **手册：** https://vorynel.com/lyapunov/manual.html
- **版本说明：** https://github.com/ManfredCh/Lyapunov/blob/60da31056e0b6d47bb2b361f0407e3e481a8536b/docs/releases/0.2.7.md
- **版本：** 0.2.7，公开源码 `60da31056e0b6d47bb2b361f0407e3e481a8536b`；发布清单标明构建输入 `a8d21e55e1bb4af43eb07376215dcf204b1cc4a2`
- **核查日期：** 2026-11-28
- **核对范围：** 官方 README、0.2.7 说明、安装清单、Agent Graph 文档和相机合同；对照组合技能和三项实验的文档、摘要、SO-101 配对和离线评估入口源码；未安装工作台、未运行仿真或审计全仓
- **通读：** 已完成

## 原文记录

Vorynel 的 Apache-2.0 公开工作台把场景编辑、脚本、Agent 和物理状态放在一个桌面产品中。0.2.7 发布通道列出 Linux x64、WSL2 和 Apple Silicon；官网当前清单与历史 alpha 手册例子的大小、校验和、批次不同。Linux 清单最低 glibc 2.39，macOS 清单最低 macOS 15.0，便携包未公证。游客有本地工作区，没有默认模型；外部模型须由用户单独配置。MuJoCo 为默认产品安装路径，Isaac Sim 等后端另需匹配的软件和硬件。

## 核对入口

- [Agent Graph](https://github.com/ManfredCh/Lyapunov/blob/60da31056e0b6d47bb2b361f0407e3e481a8536b/docs/AGENT_GRAPH.md)与[组合技能范围](https://github.com/ManfredCh/Lyapunov/blob/60da31056e0b6d47bb2b361f0407e3e481a8536b/research/agy-robotics/README.md)：独立会话中的协调、调试、复现与确定性任务判定；上游 AGY 实现锁定 0.1.11，而不是本次并列收录的 0.1.13。
- [SO-101 配对入口](https://github.com/ManfredCh/Lyapunov/blob/60da31056e0b6d47bb2b361f0407e3e481a8536b/script/gates/run-so101-paired.ts)与[离线策略评估入口](https://github.com/ManfredCh/Lyapunov/blob/60da31056e0b6d47bb2b361f0407e3e481a8536b/research/agy-sim-234/run-policy.ts)：前者比较同条件假设，后者以独立 Python 环境通过 Jobs 回收权重评估结果。配对前须固定 MJCF、网格、扰动和成功谓词。
- [实验摘要](https://github.com/ManfredCh/Lyapunov/blob/60da31056e0b6d47bb2b361f0407e3e481a8536b/research/agy-sim-234/summary.json)：10 个有效的 SO-101 同种子配对，两种方法各 5 次放入箱中；YAM 标记工位视觉误差约 39.60 → 0.92 px；当前 SO-101 Playground 权重在 6 个保留种子上 0 次达成成功谓词，零动作对照同为 0 次。均为限定场景的仿真证据。
- [相机合同](https://github.com/ManfredCh/Lyapunov/blob/60da31056e0b6d47bb2b361f0407e3e481a8536b/docs/SCENE_CAMERA_PIPELINE.md)：Camera 管挂载机位；Capture 管拍摄及同帧采集；Recording 管支持的固定来源录制。深度轴向 Z 与射线长度不是同一个量，核对应绑定实际渲染机位、帧和 world 代次。
- 官网平台清单：[Linux/WSL2](https://vorynel.com/lyapunov/releases/latest/linux-x64.json) · [macOS](https://vorynel.com/lyapunov/releases/latest/darwin-arm64.json)。

## 对 wiki 的映射

- [独立详情节点](../../wiki/entities/lyapunov-vorynel.md)
