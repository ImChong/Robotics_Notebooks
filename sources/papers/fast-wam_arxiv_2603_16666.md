# Fast-WAM: Do World Action Models Need Test-time Future Imagination?

> 来源归档（paper）

- **标题：** Fast-WAM: Do World Action Models Need Test-time Future Imagination?
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2603.16666>
- **代码：** https://github.com/yuantianyuan01/FastWAM
- **项目页：** https://yuantianyuan01.github.io/FastWAM/
- **入库日期：** 2026-09-23
- **核对范围：** 已抓取并通读 arXiv HTML 全文（`https://arxiv.org/html/2603.16666v2`，Abstract、§1–§5 及 Appendix Table 3；未在本地硬件复现训练与真机评测）。
- **通读：** 已完成
- **一句话说明：** 训练期保留视频共训、推理期跳过未来视频去噪（单步直接读取首帧 clean latent 出动作），190 ms 延迟（>4× 快于 Fast-WAM-IDM 810 ms）；LIBERO 4 套件平均 97.6% / RoboTwin 2.0 50 任务平均 91.8%。
- **沉淀到 wiki：** [`wiki/entities/paper-fast-wam.md`](../../wiki/entities/paper-fast-wam.md)
- **触发 ingest：** [`sources/blogs/wechat_robot_engineer_embodied_frontier_algorithms_2026-09-23.md`](../blogs/wechat_robot_engineer_embodied_frontier_algorithms_2026-09-23.md)

## 一手论文要点摘录（`arXiv:2603.16666v2`）

- **模型规模与结构**：Wan2.2-5B 视频 DiT + 1B Action Expert（`d_a=1024`，总参数约 6B），通过 Mixture-of-Transformers (MoT) 共享注意力与结构化掩码交互（动作 token 不可见未来视频 token）；Flow Matching 联合损失 $\mathcal{L}=\mathcal{L}_{\text{act}}+\lambda \mathcal{L}_{\text{vid}}$，`h=32`，测试时仅 10 步去噪（CFG=1.0）。
- **消融与主实验指标**：
  - **LIBERO（Table 2）**：Spatial 98.2 / Object 100.0 / Goal 97.0 / Long 95.2，平均 **97.6%**（Joint 98.5%、Fast-WAM-IDM 98.0%、w/o video co-training 93.5%）。
  - **RoboTwin 2.0（Table 1，50 任务）**：Clean 91.88% / Rand 91.78%，平均 **91.8%**（Joint 90.6%、Fast-WAM-IDM 91.3%、w/o video co-training 83.8%）。
  - **真机与延迟**：Galaxea R1 Lite 叠毛巾等真实操作任务（60h 遥操作、30k 步训练，去掉视频共训成功率掉至 10%）；单步执行时间 RTX 5090D V2 约 **190 ms**（vs Fast-WAM-IDM **810 ms**）。

## 开源状态（步骤 2.5）

- **已开源**（核查日 2026-09-23）

## 对 wiki 的映射

- 实体页：[paper-fast-wam](../../wiki/entities/paper-fast-wam.md)
- 技术地图：[embodied-frontier-algorithms-technology-map](../../wiki/overview/embodied-frontier-algorithms-technology-map.md)
