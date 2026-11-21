# Genie Sim 3.0: A High-Fidelity Comprehensive Simulation Platform for Humanoid Robot

> 来源归档（论文原文）

- **标题：** Genie Sim 3.0 : A High-Fidelity Comprehensive Simulation Platform for Humanoid Robot
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2601.02078>（HTML：<https://arxiv.org/html/2601.02078>）
- **代码：** <https://github.com/AgibotTech/genie_sim>
- **项目页：** <https://agibot-world.com/genie-sim>
- **作者：** Chenghao Yin\*, Da Huang\*, Di Yang\*, Jichao Wang\*, Nanshu Zhao\*, Chen Xu\*, ..., Maoqing Yao
- **入库日期：** 2026-11-21
- **核对范围：** 通读 arXiv HTML 全文（<https://arxiv.org/html/2601.02078>，§I–§VI，含 Genie Sim Generator 四阶段场景生成、3DGS/PGSR 环境重建、PICO 遥操作与 cuRobo 自动采集、五大闭环评测套件与 Sim2Real 实验）；未运行仿真或训练。
- **通读：** 已完成
- **一句话说明：** 智元开源的高保真综合人形仿真平台论文：集成 LLM 驱动自然语言场景生成（5,140 个 RAG 检索资产）、3DGS+PGSR 真实环境重建、PICO 遥操作与 cuRobo 自动数据生成（10,000+ 小时 / 200+ 任务）及 100,000+ 场景闭环评测套件。

## 核心论文要点（通读提炼）

1. **Genie Sim Generator（§III-A）：** 四阶段对话式生成（Intention Interpreter → 基于 Qwen embedding-v4 + ChromaDB 的 5,140 资产检索 → DSL 代码生成 → OpenUSD/Isaac Sim 场景图组装），分钟级产出多样化布局。
2. **环境重建与评测生成（§III-B–C）：** 使用 MetaCam 手持激光扫描 + SuperPoint/LightGlue 改进 COLMAP-PCD + gsplat 3DGS + Difix3D+ 视角外推 + PGSR 表面网格重建；评测侧结合 LLM + ADER（Action Domain Evaluation Rule）与 VLM 判定成功。
3. **双路数据采集（§III-D）：** PICO VR 遥操作处理长程精细任务；cuRobo + GraspNet 候选抓取点 + 失败回滚实现带环境完整碰撞的自动化采集。
4. **实验与发现（§V）：** 覆盖 `GenieSim-Sim2Real`、`Instruction`、`Robust`、`Manipulation`、`Spatial` 五大闭环套件；500/1500 条仿真数据可有效提升真机 $\pi_{0.5}$ 成功率；长程连续分拣（`Sorting Packages Continuous`）和空间构造类任务仍是所有 VLA 基线的瓶颈（≤0.30）。

## 对 wiki 的映射

- [Genie Sim 3.0](../../wiki/entities/genie-sim-3.md)
- [Genie Sim 官方源码](../repos/genie_sim.md)
