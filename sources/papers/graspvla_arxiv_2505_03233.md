# GraspVLA: a Grasping Foundation Model Pre-trained on Billion-scale Synthetic Action Data（arXiv:2505.03233）

> 来源归档（ingest · 原文通读）

- **标题：** GraspVLA: a Grasping Foundation Model Pre-trained on Billion-scale Synthetic Action Data
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2505.03233>（v3，2025-08-27）
- **PDF：** <https://arxiv.org/pdf/2505.03233>
- **项目页：** <https://pku-epic.github.io/GraspVLA-web/>
- **代码：** <https://github.com/PKU-EPIC/GraspVLA>
- **权重与数据：** <https://huggingface.co/vegebirrd/GraspVLA>；<https://huggingface.co/datasets/vegebirrd/SynGrasp-1B>
- **机构：** Galbot；Peking University；The University of Hong Kong；Beijing Academy of Artificial Intelligence
- **作者：** Shengliang Deng*, Mi Yan*, Songlin Wei, Haixin Ma, Yuxin Yang, Jiayi Chen, Zhiqi Zhang, Taoyu Yang, Xuheng Zhang, Wenhao Zhang, Heming Cui, Zhizheng Zhang†, He Wang†
- **核查日期：** 2026-11-21
- **核对范围：** 通读 arXiv:2505.03233v3 全文正文（§1 Introduction、§2 Related Work、§3 SynGrasp-1B Dataset Generation、§4 Model / Progressive Action Generation / Joint Training Loss、§5 Experiments 5.1–5.6 与 Tables 1–5、§6 Conclusion、§7 Limitations and Future Work；未逐页展开外部补充材料页）。
- **通读：** 已完成

## 核心技术与实验事实（原文核对）

- **SynGrasp-1B 数据管线（§3）：** 基于 Objaverse LVIS 子集过滤出 **240 类、10,680 个物体实例**，随机缩放跌落桌面构建布局；先生成稳定对跖抓取，再用 **CuRobo** 做**单步无碰撞规划**（避免传统 pregrasp→grasp 两段式停顿引发策略犹豫），在 **MuJoCo** 中验证抬起成功，并用 **Isaac Sim** 光线追踪渲染双视角（前视 + 腕部/侧视）与照明/材质/背景/相机外参域随机化，结合随机初始机械臂姿态；共消耗 160 张 RTX 4090 运行 10 天生成十亿帧。
- **GraspVLA 架构与 PAG（§4）：** 视觉编码器融合冻结 **DINOv2 + SigLIP** 经可训练投影接入 **InternLM2 1.8B** VLM 主干，并配条件 **Flow Matching Action Expert**。核心 **Progressive Action Generation（PAG）** 把“目标 2D 包围盒预测 → 3D 抓取位姿预测（引入最近 2 步本体状态 token）→ 基于 VLM KV cache 的动作块生成”串联为统一 CoT；联合训练时对 **GRIT** 互联网指代数据仅监督 2D bbox（部分 CoT），对 SynGrasp-1B 监督 bbox + grasp pose + flow-matching 动作块（$L_{S2}+L_{S1}$），使几何动作能力迁移至开放词汇类别。
- **关键实验（§5）：**
  - 真机 Franka Panda（300 trials/方法，5 组分布：basic/light/b.g./dis./height）：合成类与 Web 类平均分别达 **87.2% / 84.7%**，显著高于同微调的 $\pi_0$（51.8% / 36.9%）、OpenVLA（8.8% / 17.8%）、Octo（3.2% / 4.1%）。
  - LIBERO 零样本抓取子集：Long **82.0%**、Goal **91.2%**、Object **94.1%**，超过在 LIBERO 微调的 $\pi_0$（62.7 / 79.4 / 93.8）与 OpenVLA（33.7 / 56.6 / 65.4）。
  - 对照 AnyGrasp（§5.3）：常规物体相当（93.3% vs 100.0%），透明物体大幅领先（**86.6% vs 10.0%**），但推理频率较慢（**5 Hz vs 37 Hz**）。
  - 少样本后训练（§5.5）：100 条演示即可适应罕见工业件、避开杯内侧抓取及密集瓶阵按序抓取，且新类别仅用 bbox 标注即可达 **90%** 任务完成率。
  - 消融（Table 5）：vanilla 共训（66.6/53.3%）→ +PAG-2D（80.0/76.7%）→ +PAG-3D（93.3/93.3%，SPL 90.2/91.7）。
- **原文自陈局限（§7）：** 仅在 Franka Panda 与前/侧双视角上生成和验证；无法消解模糊指令（如“pick up food”）；力封闭标签未显式建模柔性体变形；L40s 上经 Torch Compile 延迟约 **200 ms**，尚不足以应对高速动态目标。

## 对 wiki 的映射

- [GraspVLA](../../wiki/entities/cn-os-graspvla.md)
