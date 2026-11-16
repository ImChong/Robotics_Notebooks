# 4D-HOF 官方项目主页归档

- **项目：** 4D-HOF / Hand-Object Flow Matching for Feed-Forward 4D Interaction Reconstruction
- **官网：** https://tamu-visual-ai.github.io/4D-HOF/
- **论文：** https://arxiv.org/abs/2610.08782
- **代码：** https://github.com/TAMU-Visual-AI/4D-HOF
- **权重与数据入口：** https://huggingface.co/eggplant71/4D-HOF
- **机构：** Texas A&M University / NVIDIA
- **核验日期：** 2026-11-12
- **核对范围：** 页面摘要、方法图、定性对比、公开代码及 Hugging Face 按钮。
- **通读：** 已完成

## 项目页可核实内容

- 单目 RGB 视频先经场景分割、手部/物体 3D 初始化与尺度对齐，得到粗手物体交互轨迹，再通过条件桥/flow matching 恢复交互一致的 4D 状态。
- 页面提供互动式 3D 重建可视化、同步对比视频、运动重定向示例（五种机器人手）。
- **发布程度：部分开源。** 已能访问官方 GitHub 仓库与权重/准备数据页面，但代码仓库 README 明确当前只发布 HO3D inference；完整训练和零起点的全链前处理没有作为一键复现流水线公开。
- 对项目页额外的 GPT-6 agent 对照：仅为论文附录 B.4 的两个定性案例和资源开销描述，不能视为规模化 benchmark。

## 对 wiki 的映射

- [4D-HOF 主实体](../../wiki/entities/paper-4d-hof.md)
- [论文原文归档](../papers/4d_hof_arxiv_2610_08782.md)
- [代码归档](../repos/4d-hof.md)
