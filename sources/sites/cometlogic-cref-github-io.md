# CReF 项目页（cometlogic.github.io/cref）

> 来源归档（ingest 附属）

- **标题：** CReF: Cross-modal and Recurrent Fusion for Depth-conditioned Humanoid Locomotion — Project Page
- **类型：** site / academic project page
- **链接：** <https://cometlogic.github.io/cref/>
- **源仓（Pages）：** <https://github.com/cometlogic/cref>（`index.html` / `scripts.js` / `styles.css` / `res/`；非训练代码）
- **关联论文：** [cref_arxiv_2603_29452.md](../papers/cref_arxiv_2603_29452.md)（arXiv:2603.29452）
- **入库日期：** 2026-08-18
- **一句话说明：** 浙大 / 山大 CReF 配套站：方法架构示意、仿真与真机实验视频、BibTeX；截至入库日 **无 GitHub 训练仓按钮**。

## 页面结构（2026-08-18 核查）

1. **Overview** — 框架一句话：单阶段深度条件人形行走。
2. **Abstract** — 与 arXiv 摘要一致。
3. **Method / Architecture** — 交叉注意力、门控残差融合、循环融合与落脚奖励示意。
4. **Experiments / Results** — 仿真消融与真机楼梯 / 高台 / 沟壑 / OOD 场景视频（资源在 `res/`）。
5. **BibTeX** — 引用块。
6. **Paper** — 链到 arXiv。

## 开源边界

- 该 GitHub 仓是 **项目页静态资源**，不是 Isaac Gym 训练或 Orin 部署仓库。
- 无 LICENSE、无 README 训练步骤、无 checkpoint。
- 结论：**确认未开源**（训练/推理代码）。

## 对 wiki 的映射

- 实体页演示入口：[paper-cref.md](../../wiki/entities/paper-cref.md)「推荐继续阅读」
