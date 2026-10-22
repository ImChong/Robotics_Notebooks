# DuplexAgent-RSI 论文来源归档

## 元数据

- **题目：** DuplexAgent-RSI: Recursive Harness Improvement for Full-Duplex Voice Agent Collaboration
- **作者：** Yingda Shen, Yuxiang Wang, Kunyu Feng, Qinke Ni, Jiaqi Li, Minghao Hsu, Junan Zhang, Dekun Chen, Yutong Bian, Zhizheng Wu
- **单位：** 香港中文大学（深圳）；Amphion Technology（据官方项目页）
- **预印本：** <https://arxiv.org/abs/2610.11299v1>（cs.AI / cs.SD；2026-10-08 提交）
- **HTML：** <https://arxiv.org/html/2610.11299v1>
- **PDF：** <https://arxiv.org/pdf/2610.11299v1>
- **项目页与演示：** <https://amphionteam.github.io/DuplexAgent-RSI-Demo/>
- **代码：** 项目页截至 2026-10-22 标注 “Code coming soon”，未提供代码仓库链接
- **入库日期：** 2026-10-22

## 摘录 1：全双工协作系统

- **问题：** 全双工交互模型能持续听说、处理中断，却难以独立承担长程搜索与复杂工具任务；串行 coding agent 擅长执行，但运行期间不能维持实时对话。
- **DuplexAgent：** 交互模型留在实时对话通道，推理模型和 coding agent 组成委派池；harness 以六个可编辑模块协调两者：Policy、Intake、Hold、Worker、Deliver、Memory。
- **运行时语义：** 系统需要处理任务接收、进度、取消、替换、过时结果丢弃及合适的结果播报窗口；“仍保持对话”并不等于用户必须等待委派任务完成后才能发言。
- **对 wiki 的映射：** 升格 [`wiki/entities/paper-duplexagent-rsi.md`](../../wiki/entities/paper-duplexagent-rsi.md)；与 [递归自改进概念页](../../wiki/concepts/recursive-self-improvement.md) 交叉引用。

## 摘录 2：Duplex-Harness-RSI 改进闭环

- **测试生成：** simulator 把请求类型、交互模式与输入条件组合成有时序的对话考试，并为行为设定预期；失败 trace 将偏差归因到 harness 模块。
- **闭环角色：** Exam Planner 根据归因失败与 repair archive 选择后续考试；Harness Editor 提出针对被诊断模块的修改。两者来自服务用户的委派池。
- **候选保留：** 候选与当前 harness 做配对运行，并经过开发集和独立 protection set 检查后才保留；附件模型权重固定，迭代对象是 harness。
- **迭代证据：** VoiceChat 固定核心评测中，十轮 RSI checkpoint r10 得分 85.3；基线 g0 为 54.6，one-shot editing 为 57.4，顺序重放十轮为 61.9。RSI 将漏委派率从 30.3% 降到 1.3%，额外委派由 3.9% 升至 6.6%。
- **另一交互模型：** Qwen-Realtime 基线 47.1，八个被保留版本后为 84.8（+37.7 分）。不同附件各自运行独立改进活动。
- **对 wiki 的映射：** 实体页“核心机制”“实验与评测”“结论”；此结果支持 harness 层持久改进的案例，不是模型权重自我训练或完整 RSI ignition 的证据。

## 摘录 3：结论边界与开放状态

- **有效结论：** 用可归因 trace、定向修改、配对评估和保护集，把全双工协作协调器中的经验式规则变成可检验的逐轮改进。
- **局限：** harness 修改无法直接修复交互模型本身的感知和时间能力；论文提出后续在 simulator 中对交互模型做 agentic RL。局部考试上的错误下降不能替代独立固定核心评测。
- **项目开放状态：** 官方网页含可交互演示及 VoiceChat、Qwen-Realtime 演示视频；页面写明技术报告与代码“coming soon”。截至入库日无官方代码 URL，故当前项目页可用于查看方法与演示，尚不能据此复现训练或改进闭环。
- **对 wiki 的映射：** 实体页“工程实践”与“局限与风险”。

## Robotics_Notebooks 映射

- 论文实体：[DuplexAgent-RSI](../../wiki/entities/paper-duplexagent-rsi.md)
- 概念：[递归自改进](../../wiki/concepts/recursive-self-improvement.md)
- 官方项目页：[sources/sites/duplexagent-rsi.md](../sites/duplexagent-rsi.md)

## 一手来源

- [arXiv 摘要与版本记录](https://arxiv.org/abs/2610.11299v1)
- [arXiv HTML 全文](https://arxiv.org/html/2610.11299v1)
- [DuplexAgent-RSI 官方项目页与演示](https://amphionteam.github.io/DuplexAgent-RSI-Demo/)
