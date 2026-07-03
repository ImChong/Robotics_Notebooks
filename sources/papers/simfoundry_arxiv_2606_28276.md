# simfoundry_arxiv_2606_28276

> 来源归档（ingest）

- **标题：** SimFoundry: Modular and Automated Scene Generation for Policy Learning and Evaluation
- **类型：** paper
- **来源：** arXiv:2606.28276v1（2026-06-25）；项目页 <https://research.nvidia.com/labs/gear/simfoundry/>
- **入库日期：** 2026-07-03
- **一句话说明：** 从**单段真机 RGB 视频**全自动构建 **sim-ready 数字孪生**，并沿 **物体 / 场景 / 任务** 三轴生成 **digital cousins**；统一支撑 **real-to-sim 策略评测**（均值 Pearson **0.911**、MMRV **0.018**）与 **sim-to-real 策略训练**（DROID / YAM，多步、铰接、双手任务）。

## 核心论文摘录（MVP）

### 1) 问题设定与统一系统定位（Abstract / Introduction / Table 1）

- **链接：** <https://arxiv.org/abs/2606.28276>
- **核心贡献：** 现有 Real2Sim 工作常只解决「场景重建」或只解决「仿真评测 / 训练」之一；SimFoundry 把 **数字孪生重建**、**多样化训练环境扩展**、**策略 benchmark + 训练数据生成** 收进**模块化**管线。相对 PolaRiS、SAGE、GenieSim 等，论文 Table 1 强调其同时具备 **自动场景构建、铰接物体、多 embodiment、背景重建、object/scene/task cousins、sim2real 训练与 real2sim 评测**。
- **对 wiki 的映射：**
  - [Sim2Real](../../wiki/concepts/sim2real.md)
  - [仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md)
  - [SimFoundry 论文实体](../../wiki/entities/paper-simfoundry-real2sim-scene-generation.md)

### 2) 三阶段管线：Extraction → Generation → Augmentation（Sec.4）

- **链接：** <https://arxiv.org/html/2606.28276v1#S4>
- **核心贡献：**
  - **Extraction：** 代表帧 RGB-D（`V_im2depth`）→ 点云对齐；VLM 检物 + 迭代分割/抠图/inpaint 直至前景清空；输出 per-object RGB-D crop。
  - **Generation：** 2D→3D mesh（`V_mesh`）+ FoundationPose 类对齐；铰接物体走独立 **articulation** 模块；CoACD 碰撞 + VLM 赋质量/摩擦；PyBullet 穿透消解后导出至 **Isaac Lab** 等下游。
  - **Augmentation：** 在孪生上生成 **digital cousins**——**object cousins**（保 affordance 换几何/外观）、**scene cousins**（语义空间谓词如 OnTop/RightOf 改布局）、**task cousins**（VLM 提议可行新任务并转仿真 goal spec）。
  - **背景：** 可选 **3D Gaussian Splat** 背景（自动：同视频前景擦除 + 深度监督 splat；或手动：无物体二遍拍摄）；前景 mesh + splat 混合场景。
- **对 wiki 的映射：**
  - [Manipulation](../../wiki/tasks/manipulation.md)
  - [Isaac Gym / Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md)
  - [PhysX-Omni](../../wiki/entities/physx-omni.md) — 同属 sim-ready 资产生成谱系，侧重不同模态与引擎导出

### 3) Real-to-sim 策略评测（Sec.5.1）

- **链接：** <https://arxiv.org/html/2606.28276v1#S5.SS1>
- **核心贡献：** 7 任务 × 5 策略族（π₀、π₀.₅、GR00T N1.6/N1.7、DreamZero 等）；**零样本 generalist** 与 **50 demo 微调** 两组；指标 **Pearson r** 与 **MMRV**（沿用 PolaRiS / SIMPLER 语境）。SimFoundry 均值 **r=0.911、MMRV=0.018**，相对 PolaRiS **Pearson 高约 0.59**；子任务级评测可把多步任务相关从约 **0.90 提到 0.95**。
- **对 wiki 的映射：**
  - [仿真评测基础设施](../../wiki/concepts/simulation-evaluation-infrastructure.md)
  - [VLA](../../wiki/methods/vla.md)

### 4) Sim-to-real 训练与 cousins 消融（Sec.5.2）

- **链接：** <https://arxiv.org/html/2606.28276v1#S5.SS2>
- **核心贡献：**
  - **零样本 sim→real：** YAM 上 Pot on Stove **99%**、DROID Stack Dishware **100%** 等；DROID 上可 finetune **π₀.₅-DROID** 于仿真演示。
  - **sim+real co-train：** 少量真机 demo 可进一步提升（如 Store Marker **60%→92%**）。
  - **Cousins 增益（平均任务成功率）：** object **+17%**、scene **+21%**、task **+40%**；held-out 物体 Pot on Stove **+50pt**；scene cousins 可把 cousin 布局 Store Marker 从 **0%→16%**。
  - **多任务泛化：** 13 仿真任务 + 7 held-out；π₀.₅-DROID-FT 真机 held-out **29%**（无 task-specific demo）；全库 VLA 微调 **28%→46%**（13 任务）、held-out sim **0%→29%**（7 任务，π₀.₅-base）。
- **对 wiki 的映射：**
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)
  - [NVIDIA GEAR Lab](../../wiki/entities/nvidia-gear-lab.md)

### 5) 重建精度与可扩展性（Sec.5.3）

- **链接：** <https://arxiv.org/html/2606.28276v1#S5.SS3>
- **核心贡献：** 全自动 F1 **0.81–0.92**（12 场景），每物体约 **3 分钟** 人工微调可达 **0.93–0.99**；相对 **SAM3D** 在 Chamfer / 位姿误差上更优，尤其在遮挡杂乱场景；端到端约 **5 min/object** 重建速率。
- **对 wiki 的映射：**
  - [CRISP](../../wiki/methods/crisp-real2sim.md) — 人形视频 Real2Sim 对照轴不同（平面原语 vs 操作场景 mesh+splat）

## 其他公开资料（非 PDF 正文）

- **项目页（交互 demo、Real2Sim Eval、Sim2Real 训练曲线）：** <https://research.nvidia.com/labs/gear/simfoundry/> — 归档见 [sources/sites/nvidia-research-simfoundry.md](../sites/nvidia-research-simfoundry.md)

## 当前提炼状态

- [x] 论文摘要与核心方法摘录（≥3 条）
- [x] wiki 页面映射与项目页互链
