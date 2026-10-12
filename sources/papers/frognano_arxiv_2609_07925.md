# FrogNano 论文来源归档

## 元数据

- **题目：** FrogNano: Training a 4B Coding Agent via Online Task Synthesis
- **作者：** Minseon Kim, Zhengyan Shi, Emiliano Penaloza, Christopher Cui, Roger Creus Castanyer, Maryam Hashemzadeh, Isadora White, Jonathan Light, Jeonghye Kim, Matheus Pereira, Darya Moldavskaya, Chinmay Singh, Fabio Vera, Baolin Peng, Xingdi Yuan, Marc-Alexandre Côté, Alessandro Sordoni
- **单位：** Froggy Team, Microsoft Research Montréal（HTML 页署名；部分联系邮箱含 Mila、UCSD 域名，页面未说明其角色）
- **预印本：** https://arxiv.org/abs/2609.07925
- **HTML：** https://arxiv.org/html/2609.07925v5
- **版本：** v1 2026-09-07；本次读取为 v5（2026-10-05 修订），cs.AI
- **项目页：** https://microsoft.github.io/debug-gym/#frognano
- **代码：** https://github.com/microsoft/FrogNano （MIT）；**模型：** https://huggingface.co/microsoft/FrogNano-4B-2609
- 附录/参考文献后半（HTML 前 100000 字符之后）未读到。

## 论文要解决的问题

小型编码智能体通常靠蒸馏前沿模型轨迹构建。FrogNano 面向最小硬件（排除前沿骨干与前沿教师），论证 4B 模型可仅用基于合成任务的 RL 后训练（无任何蒸馏的解题轨迹）达到有竞争力的表现，关键是任务要落在“当前检查点的可学习性边缘”。

## 方法与训练信息

1. **基座与数据：** Qwen3.5-4B；约 1500 个合成软件工程环境（基线混合）；TaskPilot 每轮约生成 300 个合成任务。
2. **TaskPilot 在线任务合成：** 从真实仓库快照（SWE-rebench、Scale-SWE）出发 → 生成模型写问题陈述、gold patch 与隐藏 fail-to-pass（F2P）测试 → 可执行性校验（F2P 补丁前失败、补丁后通过，P2P 保持稳定）→ 用当前策略跑 N 次 rollout 估计解决率 p̂ → 只收入目标区间内的任务，其余精炼或丢弃 → 在入选任务上 RL → 用新检查点生成下一批。迭代 1–4 目标 p̂=0.5；修订后的迭代 5 目标 0<p̂≤0.5 并换用更强的生成模型。
3. **Leaf 框架：** 五个类型化工具 Read / Write / Edit / Glob / Bash（模式取自 Claude Code 工具子集）；无工具调用的回复即结束回合；损失含推理、答案与工具调用，任务提示与工具输出被掩码。
4. **RL：** DPPO，组内相对优势（每组 8 条轨迹标准化，方差为零的组丢弃）；非对称重要性采样掩码（Δp>0.2 丢正优势 token，Δp<−0.2 丢非正优势 token）；无 KL、无熵奖励、无参考模型。每次 RL 迭代 200 次更新，每次 32 组×8 = 256 条轨迹。单节点 8×B200（2 卡训练、6 卡 SGLang 推理）。
5. **奖励：** 二值测试通过 + 成功门控的对数长度惩罚 R(N)=1−α·ln(n/N)（n>N）；截断但正确得 0.5，失败 0。N、α、学习率附录前文未给出。

## 论文报告的结果

- **评测设置：** Leaf、150 步、131K 上下文、temperature 0.6、3 个种子平均。
- **SWE-bench Verified（TaskPilot 迭代）：** 起点 39.4% → 49.1% → 53.1% → 56.7% → 59.1% → 61.5%（迭代 5）；同口径约 300 个真实 SWE-rebench 任务（按 4B 可学习性过滤）基线 48.0%。
- **留出集：** SWE-bench Pro 37.6%，Terminal-Bench v2 31.1%，PatchEval-Verified 47.3%；论文称可比肩 32B 到 100B+ 的模型，但本次读取的页面未给完整对比表。
- **框架适配：** Qwen3.5-4B 在 R2E-Gym 8.3% 与在 Leaf 上 37.2%；MiniMax-M2.5 两者均 66.5%。未见过的 mini-SWE-agent 框架 43.8%→56.4%。
- **推理期选择：** 3 候选下循环赛验证器 62.8%、单次验证 61.4%、随机 61.53%、pass@short 60.4%；整合阶段多工具调用强化平均步数 53.5→36.6，一版整合达 62.3%（pass@3 72.3%）。
- **上下文压缩：** 低预算最受益，64K 预算可恢复 131K 全上下文表现。
- **奖励破解分析：** 约 21.3% 轨迹被正则标记，2.5% 经 LLM 判官确认为破解尝试，均被脚手架阻止，FrogNano 无靠破解得分。
- **校准 vs 数量：** 本次读取的文本中没有单独变动“校准”与“任务数量”的受控消融；“校准优于数量”主要是论文的论断与迭代 5 机制变化的证据，不应当作已被受控实验量化。

## 复现与解释边界

- 仓库只发布 Leaf 框架与评测工具（见 sources/repos/frognano.md）；TaskPilot 与 RL 训练代码未见于仓库，合成任务数据集也未列链接。
- 仅限英文、偏 Python 的仓库任务；奖励与评测依赖测试，通过的补丁仍可能错误或不安全；他种语言、模糊需求、无可靠测试的工程未验证。
- 论文建议在隔离环境、受限凭据下运行并对每个补丁做人工审查。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-frognano-4b-coding-agent.md
- 概念入口：wiki/concepts/agentic-coding-software-fundamentals.md、wiki/methods/grpo.md

## 一手来源

- <https://arxiv.org/abs/2609.07925>
- <https://arxiv.org/html/2609.07925v5>
- <https://microsoft.github.io/debug-gym/#frognano>
