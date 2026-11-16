# Scaling by Structure: What Forms Can Scaling Take in Embodied AI?｜结构化扩展：具身智能如何通过可执行知识成长？

- **类型：** 2026 BEHAVIOR Challenge 冠军技术报告
- **英文标题：** Scaling by Structure: What Forms Can Scaling Take in Embodied AI?
- **中文标题：** 结构化扩展：具身智能如何通过可执行知识成长？
- **方案名称：** HiPhys-RSI（Hierarchical Physical Recursive Self-Improvement）
- **团队：** The North Star；华为诺亚方舟实验室具身智能团队（Huawei Noah EAI Team）
- **原文 PDF：** https://behavior1k-champion.com/assets/HiPhys-RSI_tech_report.pdf
- **官方项目：** https://behavior1k-champion.com/
- **官方竞赛：** https://behavior.stanford.edu/challenge/index.html
- **官方榜单：** https://huggingface.co/spaces/behavior-1k/2026-challenge-leaderboard
- **版本：** 官网所列冠军方案技术报告，18 页；2026-11-12 查阅，未见 arXiv 标识
- **入库日期：** 2026-11-12
- **核对范围：** 技术报告第 1–18 页：引言与相关工作、问题定义、方法 4.1–4.7、VLA 对照训练设置、实验 6.1–6.6、局限与评价完整性、结论、参考文献及附录 A/B/C（任务类别、失败模式、热图）；另核对官方项目主页与斯坦福竞赛评测规则。仅文本与图表阅读，未运行系统。
- **通读：** 已完成

## 核心贡献与证据

1. **结构化扩展**：冻结 coding agent 和 VLM，改进存储在可执行 Pipeline、工具封装、状态判断和失败恢复规则，不是更新 Agent 权重。可选地用 Pipeline 轨迹微调下游 VLA，这与「主 Agent 冻结」并不矛盾（报告 §4.6、§5）。
2. **构建循环（§4）**：每任务固定 10 个训练实例；读任务描述及示范 → 直接调用工具探索 → 构建自主 Pipeline → 用执行日志与 VLM 视频检查失败 → 修改程序；支持可选人工纠错。既有 Pipeline 可供后续任务分层复用。
3. **反馈规则（§4.4）**：令 s_k 为训练实例平均分，Δs_k=s_(k+1)−s_k；默认改进阈值 ε=0.02、耐心窗口 10 次修订（按任务调整）；达到停止条件或耗尽算力/交互预算即停止。
4. **Inference routing（§4.7）**：为 Pipeline 与 VLA 建 policy cards，记录功能、训练实例上的历史表现、失败边角案例；根据任务条件选择适用系统，然后由被选 Pipeline 的条件控制逻辑自主执行。
5. **官方成绩**：100 项任务的组织方隐藏测试 Q-score **0.683**、完整成功率 **48.2%**；官方第二名 Q-score **0.615**，差距约 0.068 / 11%（官网数值与排行榜说明；非单模型消融）。
6. **结构化技能规模效应（§6.2）**：开发实例上的 aggregate score 约达到 76/100；每任务改进轮数中位数 **25.5**，存在显著长尾；通用放置任务中位迭代 **46.5**，不能将其解释为所有任务耗时相同。
7. **不足（§6.3–6.6）**：隐藏实例较开发实例多数分数下降；遮挡搜索、狭窄可达空间、抓取、超时仍是瓶颈；论文还主动披露 Task 58 堆木头中利用接触判据的 benchmark hacking，Task 48 罐装食品中对部分得分的歧义行为。不能直接外推真机。

## 对照边界

- VLA 基线从 2025 年 BEHAVIOR 冠军的 π₀.₅ 系策略衍生。三种设置：Meta-1-Epoch、Meta-2-Epoch、Per-Task Post-Training；最后一种加入 Pipeline 生成轨迹和 3D 空间信息，故与单纯训练时长控制不是同一个变量。
- 官方评测允许 RGB、depth、proprioception，测试时不允许使用模拟器真值语义分割、物体状态与位姿等 privileged information；参见斯坦福评测规则。
- HiPhys-RSI 的工具组合示意不是已发布的软件实现；项目页截至 2026-11-12 没有给出可运行冠军代码的 GitHub/安装/推理入口。

## 对 wiki 的映射

- [HiPhys-RSI 冠军方案主节点](../../wiki/entities/paper-hiphys-rsi.md)
- [Agentic Manipulation Policies](../../wiki/methods/agentic-manipulation-policies.md)
- [BEHAVIOR-1K](../../wiki/entities/behavior-1k.md)

## 原始来源

- [官方技术报告 PDF（18 页）](https://behavior1k-champion.com/assets/HiPhys-RSI_tech_report.pdf)
- [冠军方案主页](https://behavior1k-champion.com/)
- [BEHAVIOR 2026 官方评测规则](https://behavior.stanford.edu/challenge/evaluation.html)
- [组织方排行榜](https://huggingface.co/spaces/behavior-1k/2026-challenge-leaderboard)
