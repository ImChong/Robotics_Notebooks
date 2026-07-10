# HumanoidGen: Data Generation for Bimanual Dexterous Manipulation via LLM Reasoning

> 来源归档（ingest · Humanoid Paper Notebooks 深读笔记）

- **标题：** HumanoidGen: Data Generation for Bimanual Dexterous Manipulation via LLM Reasoning
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/HumanoidGen__Data_Generation_for_Bimanual_Dexterous_Manipulation_via_LLM_Reasoning/HumanoidGen__Data_Generation_for_Bimanual_Dexterous_Manipulation_via_LLM_Reasoning.html>
- **分类：** 11_Simulation_Benchmark
- **arXiv：** <https://arxiv.org/abs/2507.00833>
- **入库日期：** 2026-07-10
- **一句话说明：** 现有机器人数据集与仿真基准多面向机械臂平台；对装备双臂 + 灵巧手的人形，仿真任务与高质量演示明显匮乏。双手灵巧操作更复杂——需协调臂运动与手操作，自主采集难。HumanoidGen 是一个自动化任务创建与演示采集框架，利用原子灵巧操作与 LLM 推理生成关系约束。具体：基于原子操作为资产与灵巧手提供空间标注，再用 LLM 规划器依据物体可供性（affordance）与场景生成一串可执行的臂运动空间约束；并用蒙特卡洛树搜索（MCTS）变体增强 LLM 在长时程任务与标注不足下的推理。实验里新建一个含增强场景的基准评估数据质量，结果显示 2D 与 3D 扩散策略的性能可随生成数据规模提升。

## 核心摘录（策展，非全文）

- 本文件为 **Paper Notebooks → 本库 wiki** 的溯源锚点；方法细节请读笔记页与论文 PDF。
- 知识归纳见 wiki 实体页：[paper-notebook-humanoidgen-data-generation-for-bimanual-dextero](../../wiki/entities/paper-notebook-humanoidgen-data-generation-for-bimanual-dextero.md).

## 对 wiki 的映射

- [paper-notebook-humanoidgen-data-generation-for-bimanual-dextero](../../wiki/entities/paper-notebook-humanoidgen-data-generation-for-bimanual-dextero.md)
- 分类父节点：[paper-notebook-category-11-simulation-benchmark](../../wiki/overview/paper-notebook-category-11-simulation-benchmark.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Humanoid_Robot_Learning_Paper_Notebooks/papers/11_Simulation_Benchmark/HumanoidGen__Data_Generation_for_Bimanual_Dexterous_Manipulation_via_LLM_Reasoning/HumanoidGen__Data_Generation_for_Bimanual_Dexterous_Manipulation_via_LLM_Reasoning.html>
- 论文：<https://arxiv.org/abs/2507.00833>
