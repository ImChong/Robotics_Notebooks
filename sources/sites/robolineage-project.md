# RoboLineage — 作者官方项目主页核验

- **标题：** RoboLineage: Agent-Native Data Lifecycle Governance Across Robot Policy Iterations
- **页面：** <https://robolineage.github.io/>
- **论文：** <https://arxiv.org/abs/2606.22142>
- **论文 HTML：** <https://arxiv.org/html/2606.22142v1>
- **代码：** <https://github.com/robolineage/robolineage>（官网 Code 官方指向）
- **机构：** TranscEngram（忆生科技）、The University of Hong Kong（香港大学）、Beijing Institute of Technology（北京理工大学）
- **会议：** 官网明确标注 CoRL 2026 和“Accepted to CoRL 2026”；本文引用时以该作者官网的录用声明为依据。
- **核验日期：** 2026-11-10
- **核对范围：** 官网标题、作者与机构、Paper/Code/Videos、Lifecycle Overview、Real-Robot Task Coverage、Review Reliability、Routine Lifecycle Effort、System Interface、Lineage-Guided Recollection、Paper and Citation。
- **通读：** 已完成
- **开源状态：** 已开源轻量治理运行时、frontend、schema、prompt contract、基础样例与测试（MIT）；作者 GitHub README 仍列 mock replay pipeline、打包基准评分脚本、更丰富 training adapter 等为待办，不能声称完全复现论文所有结果。

## 项目页关键证据

1. 用共享 artifact contract 串联 rollout、review、dataset、training record、evaluation、recollection；VSA 在线捕捉视觉语义锚点、离线 post-review 处理终态证据，raw capture 保持事实源。
2. ARX + ACT-style、Realman + Diffusion Policy、GALBOT G1 + VLA/LeRobot-style 均作为展示系统；不等于发布了相应策略训练权重。
3. Table 1 标注 25 tasks / 500 rollouts：完整系统 outcome 96.0%、failure phase 88.0%、data-use routing 96.4%、bad-data leakage 2.1%，基线 direct VLM 分别 78.4%、58.0%、79.6%、16.0%；route accuracy 由专家共识决定。
4. 12 组匹配 session：130.2→54.3 min，每轮 30 rollout 的 pre-training active human time 降低 58.3%；排除首次设备搭建、adapter 开发、GPU 训练时间。
5. 八任务匹配训练：92.7% vs 人工 93.1% 平均成功率；相同 candidate pool 与训练预算。
6. 三轮复采后 cube 83.3%、drawer 78.9%，对比人工 86.7%、81.1%，是特定真机任务案例而非通用机器人 benchmark。

## 对 wiki 的映射

- [RoboLineage 主实体](../../wiki/entities/paper-robolineage.md)
- [论文全文归档](../papers/robolineage_arxiv_2606_22142.md)
- [官方代码核查](../repos/robolineage-official.md)
