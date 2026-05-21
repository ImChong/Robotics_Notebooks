# EWMBench（具身世界模型生成评测）

> 来源归档（ingest）

- **标题：** EWMBench: Evaluating Scene, Motion, and Semantic Quality in Embodied World Models
- **类型：** paper
- **来源：** arXiv abs / arXiv HTML v2
- **原始链接：**
  - <https://arxiv.org/abs/2505.09694>
  - <https://arxiv.org/html/2505.09694v2>
- **作者：** Yue Hu, Siyuan Huang, Yue Liao, Shengcong Chen, Pengfei Zhou, Liliang Chen, Maoqing Yao, Guanghui Ren（* 等贡献；机构含 AgiBot、SJTU、MMLab-CUHK、HIT 等，见论文脚注）
- **入库日期：** 2026-05-16
- **一句话说明：** 首个面向 **具身世界模型（EWM）** 的公开基准：在统一输入（至多四帧初始图像 + 语言指令 + 可选 6D 末端轨迹）下让候选 **文生视频 / 图生视频** 模型自回归续写操作视频，并从 **视觉场景一致性、末端运动正确性、语义对齐与多样性** 三维度系统打分，配套精选子集与开源评测工具；数据构建于 **Agibot-World** 开源操纵数据之上。

## 核心论文摘录（MVP）

### 1) 问题动机：通用视频评测不足以刻画 EWM

- **链接：** <https://arxiv.org/html/2505.09694v2#S1>
- **摘录要点：** 文本到视频扩散模型正被用作 **EWM**：根据语言或动作策略序列生成 **物理上可执行、与任务一致** 的未来场景。通用视频基准偏重感知质量、审美与语言对齐，缺乏对 ** embodiment 运动连贯性、交互逻辑、场景静态结构保持** 等具身任务特质的考核；操纵场景中常要求背景、物体配置与本体形态基本不变，仅末端位姿与接触按指令演化。
- **对 wiki 的映射：**
  - [EWMBench（具身世界模型生成评测）](../../wiki/entities/ewmbench.md) — 「为什么需要专门基准」的论据段。
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 在「挑战 / 评测缺口」语境下引用。

### 2) 任务形式化与管线总览

- **链接：** <https://arxiv.org/html/2505.09694v2#S3>
- **摘录要点：** 记输入图像上下文 $\mathcal{I}$、语言 $\mathcal{L}$、（可选）轨迹 $\mathcal{T}$，生成视频经模型相关预处理 $f_{\text{proc}}$ 与归一化 $v_{\text{norm}}$ 后得到评测用序列 $\mathbf{V}$（论文式 (1)）。管线三组件：**统一世界初始化**、基于 Agibot-World 精选的 **十类操作任务 × 多 episode** 子集、以及覆盖 **Scene / Motion / Semantics** 的 **多维工具链**（含视频 MLLM 提示套件、轨迹检测器与视觉基础模型特征）。
- **对 wiki 的映射：**
  - [EWMBench（具身世界模型生成评测）](../../wiki/entities/ewmbench.md) — 流程图与「评测任务定义」表格骨架。

### 3) 数据集构造要点

- **链接：** <https://arxiv.org/html/2505.09694v2#S3.SS2>
- **摘录要点：** 从 **Agibot-World** 选取 **10** 类具有清晰操作目标与 **动作顺序依赖** 的任务（家庭与工业场景）；每类通过 **体素网格编码轨迹 + 贪心多样性采样** 覆盖多种合法动作模式；每段高层任务拆成 **4–10** 个原子子动作并配 **逐步字幕**，保证视频片段、子动作标签与语言描述 **一一对齐**；初始静态帧裁剪使后续帧严格服务标注指令、减少冗余运动。
- **对 wiki 的映射：**
  - [EWMBench](../../wiki/entities/ewmbench.md) — 「数据子集与任务设计」节。
  - [Agibot-World 站点归档](../sites/agibot-world.md) — 上游数据来源索引。

### 4) 三类指标（实现级摘要）

- **链接：** <https://arxiv.org/html/2505.09694v2#S3.SS3>–**S3.SS4**
- **摘录要点：**
  - **Scene：** 在 embodiment 数据上微调的 **DINOv2** 提取 patch 嵌入，用相邻帧及与首帧的 **余弦相似度** 度量布局、物体恒常性与视角连贯性。
  - **Motion（末端轨迹）：** 微调检测器得到 **末端执行器（EEF）** 轨迹；与真值比较 **对称 Hausdorff（HSD）**、**归一化 DTW（nDTW）** 与基于 **Wasserstein** 的速度/加速度 **动态一致性（DYN）**；每任务要求模型输出 **三条** 候选轨迹，按 Hausdorff **取最优** 再汇总，以降低随机性带来的不公平。
  - **Semantics：** 对齐侧用视频 MLLM 生成 **全局字幕 / 关键步骤描述**，与真值做 **BLEU** 与 **CLIP** 等比对；**逻辑错误惩罚** 显式打击幻觉操作与违背空间常识的叙述；多样性用 **CLIP** 全局视频特征，以 $1-\text{similarity}$ 形式报告。
- **对 wiki 的映射：**
  - [EWMBench](../../wiki/entities/ewmbench.md) — 「三轴指标」主表与与 VBench 类指标的差异说明。

### 5) 实验结论（公开 v2 表述）

- **链接：** <https://arxiv.org/html/2505.09694v2#S4>
- **摘录要点：** 在 **I2V / 文本驱动视频** 设定下评测多款开源、商用与 **领域微调** 模型；**EnerVerse_FT、LTX_FT** 等域适配模型在 **运动与语义** 维整体领先 **Kling、Hailuo、COSMOS、OpenSora、LTX** 等通用模型，显示 **面向机器人场景的微调** 对动力学与任务语义对齐的价值。论文同时报告：**场景一致性与轨迹一致性互补**（高场景分但几乎静止的视频仍可能任务失败），以及 **EWMBench 排名与人类排序** 的一致性优于 **VBench** 等通用视频基准的部分对比实验。局限：当前主要评 **单臂末端轨迹**、**固定视角** 场景，未来扩展 **全臂状态、动相机、导航与移动操作**。
- **对 wiki 的映射：**
  - [EWMBench](../../wiki/entities/ewmbench.md) — 「主要发现与局限」节。
  - [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md) — 在讨论长程幻觉与缺乏力反馈之外，补充「已有公开多维基准可量化部分 embodied 一致性」。

## 当前提炼状态

- [x] arXiv v2 摘要、§3 管线、数据集与三轴指标、§4 结论与局限已摘录
- [x] 与 `sources/repos/ewmbench.md`（环境、HF、目录约定）分工明确
- [x] wiki 映射：`wiki/entities/ewmbench.md` 新建，并与生成式世界模型 / 视频即仿真交叉引用
