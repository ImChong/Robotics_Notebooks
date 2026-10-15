# EmbodiedEvo 项目页归档

> 来源归档（site / project-page）

- **标题：** EmbodiedEvo: Self-Evolving Coding Agent For Physical Intelligence
- **官方入口：** <https://lab.einsia.ai/embodiedevo/>
- **论文：** <https://arxiv.org/abs/2610.18867>
- **代码：** <https://github.com/Einsia/EmbodiedEvo>
- **团队：** Navers Lab、Einsia AI
- **页面核查日期：** 2026-10-15

## 页面公开信息
项目页展示 Playground 探索、冻结经验、Test agent 在未见场景复用的两阶段流程，并提供论文、GitHub、方法、结果和轨迹入口。

页面给出的 RoboDojo leaderboard **2026-10-09 快照**：Average Score / SR 为 **44.48 / 37.82%**；Memory 为 **71.90 / 69.67%**，Open 为 **55.52 / 53.00%**，Precision 为 **23.65 / 13.25%**，Long-Horizon 为 **37.20 / 23.50%**。页面注明不同系统评测协议可能不同。

## 开源核查
- **代码：已公开。** 项目页的 GitHub 链接指向官方 Einsia/EmbodiedEvo 仓库；README 提供安装、Playground / Test 运行与测试入口。
- **复现边界：** README 要求 Linux、Python 3.11+、Docker、单独准备 RoboDojo / Isaac Sim 环境与资产、NVIDIA GPU，并配置模型凭据；源码 checkout 不包含 simulator assets。
- **许可：** README 截至核查日说明尚未选择项目级许可证，使用或再分发前应重新核对。
- **数据：** 项目页未列独立可下载数据集链接；实验基于 RoboDojo-Sim。

## 对 wiki 的映射
- 实体页：[EmbodiedEvo](../../wiki/entities/paper-embodied-evo-coding-agent.md)
- 论文归档：[来源](../papers/embodied_evo_arxiv_2610_18867.md)
- 代码归档：[来源](../repos/embodied-evo.md)
