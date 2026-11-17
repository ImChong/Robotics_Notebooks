# GM-100 官方项目主页（RHOS AI）

- **网址：** https://www.rhos.ai/research/gm-100
- **论文：** https://arxiv.org/abs/2601.11421
- **官方仓库（Code）：** https://github.com/rhos-ai/GM-X
- **任务目录：** https://www.rhos.ai/research/gm-100/tasks/
- **结果页：** https://www.rhos.ai/research/gm-100/results/
- **官方数据：** https://huggingface.co/datasets/rhos-ai/gm100-cobotmagic-lerobot 、https://huggingface.co/datasets/rhos-ai/gm100-xtrainer
- **核查日期：** 2026-11-17
- **通读：** 已完成
- **核对范围：** 阅读官方网站首页说明、任务展示、Leaderboard 指标和官网更新说明，逐项核验 Code/Train/Test 资源入口，并对照论文 v1 §§3–7。
- **开源结论：** 数据集已开放（Cobot Magic 为 LeRobot，Xtrainer 为 HDF5）；Code 链接指向 GM-X 官方仓，但该仓目前仅 README、assets、.gitignore，没有可供直接运行的 GM-100 训练或评测脚本。官网明确为 preview 阶段，展示和结果尚可能不完整。

## 页面要点

- 强调从人–物交互 primitives、affordance 出发设计 100 个长尾细粒度任务，不是把 100 个泛化任务名称简单拼接。
- 提供任务视频、部分训练/测试下载入口、两种双臂平台切换和 Success Rate / Partial Success Rate 两种榜单口径。
- 2026-04-01 更新了 Cobot Magic 部分子任务评分判据；对比论文发表时的评分与最新网站时需要记录协议日期。
- 若上传自有结果，注意区分作者审核与未审核成绩，并保留 rollout 视频与机器本体说明。

## 对知识库映射

- [论文归档](../papers/gm100_arxiv_2601_11421.md)
- [官方 GitHub 归档](../repos/rhos-ai-gm-x.md)
- [独立论文节点](../../wiki/entities/paper-gm-100.md)
