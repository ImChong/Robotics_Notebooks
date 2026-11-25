# NavMCP 论文来源归档

> 来源归档（ingest）

## 元数据

- **题目：** Scaffolding Foundation Models into Physical-World Agents Pushes the Frontier of Long-Horizon Navigation
- **方法 / 系统：** NavMCP
- **作者：** Zixing Lei, Gengze Zhou, Xiong-Hui Chen, Jiazhao Zhang, Yiyang Huang, Hang Yin, Haoqi Yuan, Qi Wu, Weixin Li, Siheng Chen
- **类型：** paper
- **预印本：** https://arxiv.org/abs/2608.30396
- **提交日期：** 2026-08-31（arXiv v1）
- **原文下载/核对日：** 2026-11-21
- **通读：** 已完成
- **核对范围：** 全文 1530 行（Abstract、§1–§6 NavMCP 三通道协议 `intent / observation / memory`、Table 1 跨基准 EQA 主结果、Table 2 匹配骨干对照、Table 3 架构消融、Unitree Go2 实机实验）
- **作者学术页：** https://yhqpkueecs.github.io/
- **代码 / 数据：** 截至 2026-11-21 复核，arXiv 条目与作者学术页仍未列代码仓库或数据集下载入口；当前公开材料不足以复现完整系统。

## 论文摘录

### Scaffolding Foundation Models into Physical-World Agents Pushes the Frontier of Long-Horizon Navigation（Lei et al., 2026）

- **问题：** 视觉语言模型（VLM）可做高层推理和计划调整，但反复将语义目标落到导航上并不稳定；导航基础模型（NFM）能闭环执行语义目标，却通常按有限回合运行，缺少跨任务的持久推理与记忆。
- **核心贡献：** 提出 NavMCP，将 VLM 推理 agent 与 NFM 执行器组织成长时程探索系统，不重新训练任一基础模型。协作由三个通道组织：**intent** 将取证需求转成导航调用；**observation** 将导航 rollout 转为可追溯的轨迹证据；**memory** 跨调用累积已发现信息、否定证据与未解决目标。
- **论文报告的结果（Table 1–3 全文核对）：** 在 HM-EQA 上准确率 **76.7%**（超 FAST-EQA 7.5 个百分点，Normalized Agent Steps **0.15**）；在 MT-HM3D 上 **54.4%**（+3.9 个百分点，Normalized Agent Steps **0.19**）；在 EXPRESS-Bench 上 LLM Score **79.27**。在统一骨干（`Qwen3.5-397B-A17B` + 相同感知工具集）对照下（Table 2），HM-EQA 领先 FAST-EQA 10.5 个百分点（相对 episodic interface 提升 14.9 个百分点）。Unitree Go2 实机成功率为 **78.3%**，任务时程增长时相对最强基线优势从 10 个百分点扩大到 45 个百分点。
- **对 wiki 的映射：**
  - [NavMCP 论文实体页](../../wiki/entities/paper-navmcp-long-horizon-navigation.md)
  - [视觉–语言导航](../../wiki/tasks/vision-language-navigation.md)
  - [Qwen-Robot Suite](../../wiki/entities/qwen-robot-suite.md) — agent 与专用基础模型组合的系统设计背景

## 复现与解释边界

- 上述数值与消融已对照论文全文（Table 1–3）核实；NavMCP 定位为不重训基础模型的协议/脚手架层，具体 NFM（如 Qwen-RobotNav-8B、InternVLA 等）与上层 VLM 骨干配置见 Table 3。
- 截至 2026-11-21，官方入口仍未提供可运行源码与数据下载链接。
