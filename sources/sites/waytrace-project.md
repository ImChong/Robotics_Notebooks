# WayTrace — 官方项目主页核验记录

- **项目：** WayTrace: A Map-to-Language-to-Trace Hierarchy for Embodied Urban Navigation
- **中文：** WayTrace：面向具身城市导航的地图—语言—轨迹分层架构
- **主页：** <https://way-trace.github.io/>
- **一手论文：** <https://arxiv.org/html/2611.05129v1>
- **核查日期：** 2026-11-07
- **核对范围：** 已从论文摘要及正文验证项目主页 URL；尝试直接打开主页，网页读取服务报错；也尝试浏览器访问但当前会话没有可用的原始浏览器上下文。尚未取得主页正文 / Footer / Code 区块，未完成项目页源码与数据外链核查。
- **通读：** 未完成
- **网页状态：** **本次读取受阻**，不等于页面已下线或不可访问。
- **源码状态：** **待核实公开链接**。论文 Introduction 仅承诺未来开放代码、模型、基准、数据及仿真环境；不能写成“已开源”，也不能反向断言“确认未开源”。
- **官方源码地址：** 未能核实；不建立占位 repo 源文件，不生成臆造的源码运行时序图。

## 可核实的论文实施信息（非项目网页引用）

- 论文写明远端 Orchestrator VLM、车载 Trace Predictor、深度投影及 5 Hz RL 导航策略的分层异步结构。
- 提供具体模型与机器人配置以及提示词（附录 F），但**论文算法描述不等于公开仓库可运行入口**。
- 上述实施细节见 [WayTrace 原文记录](../papers/waytrace_arxiv_2611_05129.md)。

## 对 wiki 的映射

- [WayTrace 唯一实体](../../wiki/entities/paper-waytrace.md)

## 待后续核对

- 主页 Resources/Code/Models/Dataset 按钮实际 URL；官方 GitHub、模型权重或数据集是否发布。
- 若出现可运行训练、推理或部署代码，按实际 README 与入口读源码、补 sources/repos/ 与 Mermaid sequenceDiagram。
