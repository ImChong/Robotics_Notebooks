# GaP：原文通读与实现复核（2026-11-15）

- **类型：** paper
- **原文：** <https://arxiv.org/html/2607.05369v1>
- **版本：** 2026-11-15 获取的 HTML；具体正文、附录和代码范围如下。
- **代码：** <https://github.com/graph-robots/graph-as-policy>
- **核对日期：** 2026-11-15
- **核对范围：** arXiv:2607.05369v1 正文 §1–6、附录 A–G 的相关工作、ROS 插线接口、示例图说明、MORSL 接口与自学习伪代码/反馈说明；核对正文全部实验表与图注。另打开官方项目页，读主仓 README、Quickstart、`gap/cli/run.py` 全文及 `WorkflowExecutor.execute/_dispatch_node`，未执行仿真或真机。
- **通读：** 已完成

本次复核单独存档；[历史摘录](gap_arxiv_2607_05369.md)保留原有日期和发布状态，不能代替下述版本及协议核对。

## 原文与实现核对

- [原文](https://arxiv.org/html/2607.05369v1)的 VA 假设是固定工位、机器人/传感器与有界物体/位姿分布，不能外推成家庭开放世界保证。
- §5.2 主动说明 CaP-X 与 GaP 的先验条件不完全公平；无图或单 agent 的 0% 是该实验配置的消融，不是所有编码 agent 的普遍下界。杂货 I/II 的初始图已成功，未使用 self-learning；爆米花才展示 10 轮排练的 33%→94% 仿真增益。
- §5.7 洗箱是**双臂仿真**，150 次 GaP 成功 143 次（0.953），专家 148 次（0.987），不能写成真机产线实绩。插线 §5.6 是真机 121/130，逐次成功与整段长序列成功是不同指标。
- 杂货真机单次平均周期 GaP 67s、TipTop95s；结论说尚未达到约7s/件的工业吞吐目标，不能将“低于工业标准”误写成耗时少于7s。
- [官方项目页](https://graph-robots.github.io/gap/)仍提供主仓和技能库链接；2026-11-15 主仓 README 另外指向独立的 self-learning 仓。该后续仓不等于本论文主仓内已包含所有自学习路径，本次未读其论文或源码。
- 运行核对：[CLI](https://github.com/graph-robots/graph-as-policy/blob/main/gap/cli/run.py)的 `--validate-only` 只读工作流并验证；正常路径创建 sim/real connector 后调用 `gap.runtime.execute.execute`。`gap/runtime/executor.py` 的 `execute` 先静态检查，`_dispatch_node` 区分 tool/script/subgraph，记录输入、输出与错误 trace。主仓树 `e572392c`。

## 对 wiki 的映射

- [GaP 详情](../../wiki/entities/paper-gap-graph-as-policy.md)
