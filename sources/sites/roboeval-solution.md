# RoboEval 2026 冠军方案：作者项目主页

- **网站：** https://ilialarchenko.com/projects/roboeval2026/
- **论文：** https://arxiv.org/abs/2611.11464
- **代码：** https://github.com/IliaLarchenko/RoboEvalSolution
- **权重：** https://huggingface.co/IliaLarchenko/RoboEvalSolution
- **入库日期：** 2026-11-24
- **核对范围：** 项目页 Challenge、Model、Training、Inference、示范与恢复、训练历程、结果和结论；视频未逐帧核对。
- **复核日期：** 2026-11-25（项目页与 arXiv v2 §9 / 表 6 交叉核对）
- **通读：** 已完成

## 原始资料与开放边界

公开作者技术报告、Apache-2.0 训练/评测源码与提交权重；官方示范数据入口为 https://huggingface.co/datasets/helen9975/roboeval_ee_delta_20hz_scaled 。没有确认完整历史 rollout、合成轨迹和训练状态全部公开，未下载运行。主办方最终榜单 https://robo-eval.github.io/challenge.html#results 确认新种子复评结果 0.988，区别于该页上方作者自己的 0.990；项目页后部的 Final results 另外列主办方结果。项目页的最后调优用新留出种子比较权重平均和逐任务推理设置，不是继续训练或在公开种子优化。

## 对 wiki 的映射

- [论文实体](../../wiki/entities/paper-roboeval-solution.md)
- [论文归档](../papers/roboeval_solution_arxiv_2611_11464.md)
- [源码归档](../repos/roboeval-solution.md)
