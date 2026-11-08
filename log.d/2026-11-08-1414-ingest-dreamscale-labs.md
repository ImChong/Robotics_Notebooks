## [2026-11-08] ingest | dreamscale-labs | 云端机器人基础模型推理平台

- **意图：** 收录 “I want to run a model on a robot to do something” 的官方产品与技术实现，区别机器人模型和云端推理服务。
- **一手来源：** Dreamscale 官网、Quick start / Models / Python SDK / SO-101 / YAM 文档，以及 2026-07-24、2026-10-16 两篇官方完整技术文章。
- **开放结论：** CLI/SDK 文档公开；HUD 评测适配代码 MIT 开放；商业平台 serving 后端未被官方仓库完整公开，不将其误标为全部开源。
- **工程关注：** 模型与 embodiment 观测/动作合同、动作块与控制频率、云推理和本地安全层分离；官方展示与独立复现口径分开。
- **输出：** Dreamscale Labs 主节点、网站归档、技术博客归档、HUD 源码审阅、频率解耦交叉引用；包含原文流程图和开源代码运行时序图。
- **验证边界：** 未启动收费推理会话、仿真或实体机器人；以 PR CI 检查页面健康度与 Mermaid 语法。
