# Envisioning Agentic Robotics: A New Era of Robotic Systems｜展望智能体机器人：机器人系统的新纪元

> 来源归档（paper / position and research roadmap）

- **英文：** Envisioning Agentic Robotics: A New Era of Robotic Systems
- **中文：** 展望智能体机器人：机器人系统的新纪元
- **作者：** Grid Robotics：Qingfeng Lan、Duo Li、Siyao Li、Yiheng Lyu、Shijian Lu、Zhenghao Qi、Shizun Wang、Xinchao Wang、Ming Yang、Zhiqin Yang、Zuhao Yang、Tao Yu
- **论文：** https://arxiv.org/abs/2611.18359
- **全文 HTML：** https://arxiv.org/html/2611.18359v1
- **PDF：** https://arxiv.org/pdf/2611.18359
- **版本 / 日期：** arXiv v1，2026-11-13（提交日期，非会议接收日期）
- **项目主页：** https://grid-robotics-official.github.io/agentic-robotics-roadmap/
- **代码：** 截至 2026-11-16，官方项目页未列与该立场/研究路线论文对应的可运行训练、推理或评测仓库；论文不是一个可直接复现的新机器人模型。
- **核对范围：** arXiv v1 HTML 正文第 1–7 节（背景、S2/S1/S0 架构、具身 harness、时限与实时性、9 维评测、数字/物理 agent 对比、研究方向及局限）、附录 A（相关工作）、附录 B（应用场景）和附录 C（贡献者）；对照官方项目页的框架、指标、工作层级和公开资源；未独立重跑所引用研究。
- **通读：** 已完成

## 原文要点及 wiki 映射

1. **定义与分工（§2–3）：** Agentic Robotics 围绕用户意图构建感知、推理、执行、验证、修正闭环。Embodied S2 = foundation model + embodied harness；S1 是可调用技能/感知/规划模块，S0 是机体实时控制。S2→S1→S0、S2→S0、S2 编码生成 S1 三条执行路径可共存。
   - 对 wiki 的映射：[独立论文实体](../../wiki/entities/paper-envisioning-agentic-robotics.md)「核心原理」「流程总览」。
2. **Harness 与打断（§2.2/§3.3）：** 核心部分为 context manager、tool interface、planner、memory、verifier；可扩展安全门、预算控制、自我改进。实时事件监控器在 S2 之外监听用户/场景/机器人变化，可中断 S2，并在预设安全规则下使 S0 立即响应。取消命令不等于确认动作已停止；重新下发冲突动作前必须检查执行/机体状态。
   - 对 wiki 的映射：「核心原理」「工程实践」。
3. **运行约束（§3.4）：** 强调 capped scaling 与连续世界时钟；暂停仿真推理时间会掩盖场景过时和反应时限。文中 S0 约 50 Hz–1 kHz 是引用不同系统的范围，不是本论文新控制器的实测频率。
   - 对 wiki 的映射：「工程实践」「局限与风险」。
4. **评测框架（§4）：** 提出 8 项能力 + cost，共 **9 维、53 个子维度、29 个与 embodiment 相关子维度**；须区分固定 harness 比模型和固定模型比 harness。建议记录 wall-clock / observation age、实时与暂停条件、独立结果核查；多维画像不能随意汇总成单一得分。
   - 对 wiki 的映射：「评测与证据边界」。
5. **对比数据（§4.3）：** 论文汇集其他工作的条件化结果，例如 RT-Safe 的 paused 19.8% vs live 0.7% 安全成功率；不是本论文新跑的一组实验。不同任务、模型、评测协议的数字不可横向排名。
   - 对 wiki 的映射：「评测与证据边界」。
6. **责任阶梯（§6）：** L1 primitive、L2 task、L3 workflow、L4 standing role；空间作用域 workcell→room→building→site 是另一维。作者明确表示现阶段无足够公开证据证明非结构化环境中的可靠 L4 持续自主。
   - 对 wiki 的映射：「研究路线」「局限与风险」。
7. **局限（§4.4/§7、附录 A–B）：** 当前基准在职责范围、跨时长与真实连续时间方面覆盖不足；真实机器人结果昂贵且判定困难，综述所引用公司演示不等于统一协议下的验证。需要可靠的物理效果判定、标定仿真与安全监督。
   - 对 wiki 的映射：「结论」「局限与风险」。

## 资料定位

这是 **position / roadmap / evaluation proposal**，不应标记成已有可复现训练方案，也不应把汇总的外部 benchmark 数字当成本文独立实验结果。arXiv v1 与项目页可核验；项目页并未提供相应可运行源码。
