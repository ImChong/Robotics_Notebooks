# EmbodiRSI 与 InstAgent：待完成的资料归档

> 草稿：尚未完成原文及项目页核验，不可视为完整 ingest。

## EmbodiRSI

- 论文：<https://arxiv.org/abs/2609.38905v3>
- 项目页：<https://embodirsi.instadapt.com>
- 标题：EmbodiRSI: Recursive Self-Improvement for Data-Efficient Robot Adaptation
- 指定的 v3 PDF 因文件大小超过抓取限制而未能读取，项目页也暂时无法访问。可读取的 [v4 HTML](https://arxiv.org/html/2609.38905v4) 为后续修订版本；尚未逐节完成通读和版本差异核查。
- 摘要称系统在目标场景构建仿真，并用策略执行反馈指导纠错轨迹及自适应数据采集，再利用少量真实数据微调。
- 源码与数据开放状态：项目页无法访问，待核实。

## InstAgent

- 论文：<https://arxiv.org/html/2610.23098v1>
- 项目页：<https://instagent01.github.io/>
- 标题：InstAgent: A Recursively Self-Improving Agent Operating System for Heterogeneous Robot Collaboration
- 论文介绍通过行为树组织异构机器人技能、依赖与交接条件；从执行反馈修订并复用经验证的行为树，结合多模态时空记忆。
- 已获取论文 HTML；尚未确认逐节通读完成。项目页暂时无法访问，源码与数据开放状态待核实。

## 后续工作

1. 获取并通读 EmbodiRSI v3 全文；核对 v4 差异。
2. 通读 InstAgent 论文全文，并核对两项目官方代码、数据及运行入口。
3. 完成各自 sources 和 wiki 实体、基于原文的 Mermaid 项目流程图；若有可运行官方源码，阅读后补充真实运行时序图。
4. 运行 `make ci-preflight` 和必要检查；完成前 PR 保持草稿。
