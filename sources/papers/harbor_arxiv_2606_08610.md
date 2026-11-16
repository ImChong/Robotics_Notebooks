# HARBOR（arXiv:2606.08610）

> 来源归档（paper）

- **标题：** HARBOR: A Harness Framework for Agentic Robot Reinforcement Learning
- **类型：** paper
- **作者：** Zechu Li、Yufeng Jin、Xiaoyang Liu、Puze Liu、Vignesh Prasad、Carlo D'Eramo、Georgia Chalvatzaki（TU Darmstadt、Honda Research Institute Europe、Columbia University、同济大学、上海智能自主系统研究院、维尔茨堡大学、Hessian.AI）
- **arXiv：** <https://arxiv.org/abs/2606.08610>（v1，2026-06-07，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2606.08610v1>
- **代码：** 论文正文与附录未给出仓库或项目页链接（PDF 全文检索无 github / 项目页 URL）；arXiv 摘要页亦未核到，入库日（2026-11-12）按「未列链接」处理
- **入库日期：** 2026-11-12
- **一句话说明：** 把机器人 RL 自动化当作 harness 工程问题：用 LLM 编码智能体（agents）+ 命令 + 持久工件 + 可执行关卡（gates）+ 可复用知识，从依赖安装、任务/奖励/域随机化生成到 RL 训练与调参，在 6 个基准、16 个任务上自动化仿真 RL 流程。
- **核对范围：** 通读 arXiv v1 PDF 全文：摘要、第 1 引言、第 2 节问题建模、第 3 节方法、第 4 节实验（4.1 至 4.3）、第 5 节结论与局限、参考文献，以及附录 A 相关工作、B 背景、C 智能体与命令清单、D.1 真机设置、D.2 任务提示词、D.3 端到端学习曲线说明、D.4 调参详表说明、D.5 与 D.6 消融协议与 Sonnet 复现的文字；图中曲线与 Tab. 4/5 的逐格数值只读了正文已转述部分，未逐格核对。未运行任何代码，未核对第三方基线实现（Eureka、REvolve）的原始论文。
- **通读：** 已完成

## 核心摘录

1. **形式化（第 2 节）：** harness 记为 H_RL = (H_A, C, M, G, K)，即智能体、命令、可变工件、可验证关卡、可复用知识；论文明说 HARBOR「不保证最终策略的语义正确」，只把常见工程失败变成可观察的关卡失败。
2. **阶段（第 3 节 Tab. 1，附录 C）：** 依赖安装、任务生成、奖励生成、RL 集成、域随机化、RL 调参；每阶段由一个子智能体通过命令执行并产出工件，过关卡才推进，失败回传摘要重试，重试预算耗尽则标未解决并请用户介入。
3. **并行与经验（第 3 节）：** 主智能体集中控制，子智能体在隔离试验目录里分散执行；每轮把规律蒸馏成要点写入经验记忆，下轮按仿真器、任务、算法标签检索。
4. **端到端（第 4.1 节 Tab. 2）：** 4 个操作任务 × 4 个仿真器（IsaacLab、ManiSkill、Genesis、MJLab）；IsaacLab 仿真成功率 0.885 至 1.0；MJLab 明显偏低（dex-grasp 仿真 0.0）；真机每策略 10 次试验。
5. **奖励基线（第 4.1 节）：** 同一 Opus 骨干、同墙钟预算、固定任务实现，对比 Eureka 与 REvolve；insert-drawer 与 stack-cube 上两基线明显落后（Tab. 2、附录 D.3）。
6. **调参（第 4.2 节 Fig. 4）：** PPO、SAC、TD3 在 IsaacLab、Bi-DexHands、Loco-MuJoCo 共 12 个任务设置中，11 个持平或优于默认。
7. **消融（第 4.3 节 Tab. 3）：** 仅在 ManiSkill Push-Cube 单任务、每配置重复 10 次；完整 harness 48/50，Vanilla 29/50。
8. **局限（第 5 节）：** 受限于暴露给智能体的工具，新任务可能需大量脚手架修补或失败；sim-to-real 边界仍需人工；目前仅支持较简单的策略结构。

## 对 wiki 的映射

- [paper-harbor](../../wiki/entities/paper-harbor.md)
