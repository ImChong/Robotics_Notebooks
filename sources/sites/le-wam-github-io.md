# LeWAM 项目页（le-wam.github.io）

> 来源归档

- **标题：** LeWAM: End-to-End World and Action Modeling with JEPAs
- **类型：** site（项目页）
- **URL：** <https://le-wam.github.io/>
- **论文：** <https://arxiv.org/abs/2610.12818>（归档见 [lewam_arxiv_2610_12818.md](../papers/lewam_arxiv_2610_12818.md)）
- **代码：** <https://github.com/MinghaoFu/lewam>（归档见 [lewam.md](../repos/lewam.md)）
- **数据与检查点：** <https://huggingface.co/LeWAM>
- **入库日期 / 核查日期：** 2026-10-12
- **一句话说明：** UCSD / ETH / UBC / Brown 项目页：联合 JEPA 世界-动作模型，页面汇总仿真与真机结果。

## 页面核查结论

- 页面含论文 PDF、代码、数据与检查点、X 公告入口；称 arXiv 版本 “coming soon”（实际 arXiv:2610.12818 已上线）。
- 真机描述：七自由度机械臂，场景相机 + 腕部相机，30 Hz 关节目标，动作块在远程推理机计算；三项任务各 30 次，结果与论文表一致（LeWAM 冻结 V-JEPA 2.1 平均 35.6%，端到端 33.3%，Diffusion Policy 38.9%，π0.5 28.9%）。
- HF 组织页（核对时）列出 13 个模型、30 个数据集（模型名如 lewam-pusht、lewam-toolhang-tc 等，更新于数天前）；逐文件内容未核对。
- 判定：**已开源**（代码 + 检查点 + 数据集入口）。

## 对 wiki 的映射

- [LeWAM 详情节点](../../wiki/entities/paper-lewam.md)
