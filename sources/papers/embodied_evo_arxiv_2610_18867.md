# EmbodiedEvo 论文来源归档

## 元数据
- **题目：** EmbodiedEvo: Self-Evolving Coding Agent For Physical Intelligence
- **作者：** Wenyi Li、Yizhe Chi、Youjie Zheng、Chunyu Liu、Xiaoqiu Wang、Mingju Gao、Zheng Jiang、Deyao Hong、Calvin Xiao、Kaisen Yang、Qinhuai Na
- **机构：** Navers Lab、Einsia AI、清华大学（Tsinghua University）
- **预印本：** https://arxiv.org/abs/2610.18867
- **HTML：** https://arxiv.org/html/2610.18867v2
- **PDF：** https://arxiv.org/pdf/2610.18867
- **项目页：** https://lab.einsia.ai/embodiedevo/
- **代码：** https://github.com/Einsia/EmbodiedEvo
- **版本：** v2，2026-10-14；**入库日期：** 2026-10-15；**复核日期：** 2026-11-10
- **核对范围：** arXiv HTML v2 全文：§1–6、参考文献与附录 A–F（协议、逐任务结果、跨任务迁移、失败案例、成本核算、完整成功执行轨迹）。
- **通读：** 已完成

## 论文要解决的问题
具身 coding agent 能在单次任务中编写、执行并修订控制程序，但新场景中的 agent 无法继承此前摸索出的可行操作及其适用条件。反复探索会占用交互与推理预算，原始交互历史也不便复用。

## 方法与实验
EmbodiedEvo 固定模型权重、任务目标和机器人接口，把经验累积在 agent harness 中：
1. **Playground：** 在每个任务的一个场景中观察、编写并运行 Python 控制程序，根据执行反馈修订程序。
2. **Skills：** 将成功操作组合为可参数化、可执行的程序，并加入执行检查。
3. **Lessons：** 记录适用条件、已观察到的失败模式、诊断与恢复建议；未证实解释保留为假设。
4. **Freeze / Test：** 冻结 Skills 与 Lessons，交给新 agent 在未见布局测试。Test agent 可继续适配控制程序，但经验只读，不能 reset 或跨 Test 场景学习。

底层接口包括读取任务、观察图像和机器人状态、执行动作、Playground 重置。实验使用 RoboDojo-Sim；接口观测包含 RGB、关节状态、末端位姿和夹爪命令状态，不提供物体真值位姿、深度图或相机标定。

## 论文报告结果
- 2026-10-09 RoboDojo leaderboard 快照中，EmbodiedEvo 平均分 **44.48**、成功率 **37.82%**；项目页说明不同系统的评测协议存在差异。
- 配对设置下，继承经验后成功率从 **8.50%** 提高至 **30.17%**。
- Astra 经验迁移给 Sol / Luna 后，成功率分别提高 **8.86 / 5.50 个百分点**。
- Skills / Lessons 与原始交互历史的成功率为 **30.17% / 29.78%**，Test 单回合估算成本为 **$5.732 / $12.203**（降低 53.0%，按论文 token 费率估算）。
- 项目页报告 Precision 成功率 **13.25%**、Long-Horizon **23.50%**，精细接触和长时程执行仍是短板。

## 补充核对：实验协议与容易误读的结论
- **两种规模不能混写：** Full Evaluation 为 42 个任务，每任务 50 个 Test 场景，共 **2,100** 次测试；配对消融为每任务 5 个场景，共 **210** 次。Full Evaluation 使用任务描述 + 官方 wiki；主要配对实验使用任务描述，不应把 37.82% 与 8.50%→30.17% 当成同一条件直接相减。
- **单回合执行预算：** Playground 与 Test 均最多 100 次已接受的代码执行，另受任务原生动作数量限制；Playground 同场景 reset 不会恢复代码执行预算。
- **跨任务迁移：** 12 个固定目标任务中，相关任务经验相对无经验提高成功率 **14.72 个百分点**；标签不相似任务经验提高 **16.67 个百分点**。不应据此推断“随便迁移都有效”，逐任务结果存在回退与差异（§4.4、附录 C）。
- **失败探索仍可复用：** 传送带按图抓取任务的 Playground 在 100 次代码执行内未成功，但冻结的局部经验令后续 Test 的 Score / SR 从 **0 / 0** 提升到 **40 / 40%**（§4.5）。
- **成本比较对象：** 相对无经验 Test，Skills/Lessons 的估算成本从 **$4.058** 增至 **$5.732**，约 **+41.3%**；相对读入原始历史的 **$12.203** 才是 **-53.0%**。这两种说法都只统计 Test 阶段，**不包含 Playground 探索成本**（§4.6、附录 E）。

## 解读边界
这些结果来自 RoboDojo-Sim 仿真，不能据此推断真机性能。经验复用改善的是新 agent 的起点，不会自动增强底层动作原语；接触精度和多阶段任务收尾仍会失败。跨系统 leaderboard 的评测协议并不完全相同。

## 对 wiki 的映射
- 实体页：[EmbodiedEvo](../../wiki/entities/paper-embodied-evo-coding-agent.md)
- 项目页：[归档](../sites/embodied-evo.md)
- 代码仓：[归档](../repos/embodied-evo.md)

## 原始来源
- arXiv 摘要与版本信息：<https://arxiv.org/abs/2610.18867>
- arXiv HTML v2：<https://arxiv.org/html/2610.18867v2>
