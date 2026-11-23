# TensorLean：CertiK 与哥伦比亚大学联合公布生产级 GPU Kernel 形式化验证

> 一手来源：研究参与方 CertiK 的官方技术公告；不是论文全文，也不是独立复现实验。

- **原文：** https://www.certik.com/blog/tensorlean-certik-and-columbia-university-researchers-formally-verified-production-ai-kernels-from-deepseek-and-flashattention
- **机构：** CertiK、哥伦比亚大学（Columbia University）
- **公开日期：** 2026-11-20
- **项目：** https://tensorlean.ai/（本次网页抓取失败，项目细节未直接核对）
- **官方仓库：** https://github.com/TensorLean/TensorLean
- **仓库归档：** [TensorLean README](../repos/tensorlean.md)
- **知识节点：** [TensorLean](../../wiki/entities/tensorlean.md)
- **核对范围：** 已完整阅读 CertiK 2026-11-20 官方公告正文，包括「Beyond Testing」「The Breakthrough」「Continuous Assurance」三部分及引述；未将不可访问的项目主页、尚未发布的论文/证明文件视为已读。
- **通读：** 已完成

## 公告确认的核心事实

- 研究团队称 TensorLean 为生产级 AI GPU 算子提供机器可检查的形式化证明，**证明的范围受明示的输入前提与数值误差容限约束**，不等于任何配置、任何模型或任何硬件平台都正确。
- 方法直接面向经过编译的 **PTX 表示**，而不是仅检查高层张量表达式；使用 Lean 开发与机器检查证明。
- 报告统计 **149 个已编译 GPU Kernel 实例**；其中 **83 个**满足相应 Lean 规格；共报告 **38 个缺陷，分布于六个库**，其中 **DeepSeek DeepGEMM 2 个、FlashAttention 23 个**。
- 深入问题包括并行执行协调、数据布局、低精度算术误差与异常配置；公告强调在定义好的条件内证明正确，而非用有限的随机测试推断正确。
- 官方将「世界首次」作为研究方的声明；目前尚不能以其自身公告独立证明先发性。数字不是「83 个不同开源库」或「所有 DeepSeek 算子都已被验证」。

## 版本与差异

- 公告于 2026-11-20 写 **38** 个缺陷；2026-11-23 核查的 [官方仓库 README](https://github.com/TensorLean/TensorLean) 已列出 **39** 个缺陷条目。不同快照不可无说明地混用。
- 论文、完整 TensorLean 框架源码、PTX 语义模型与具体证明尚待发布；公告结果是**研究方报告**，本次未自行运行证明检查。

## 工程意义

对面向机器人/具身模型的 GPU 推理或训练 Kernel，性能优化之后的正确性检查应同时明确：张量规格、允许误差、输入前提、具体目标架构与编译产物。形式化验证可补足测试对稀有并发与极端尺寸的覆盖盲区，但不能代替系统级硬件故障、实时控制安全或模型输出语义的验证。
