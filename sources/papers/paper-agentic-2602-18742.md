# RoboCurate: Harnessing Diversity with Action-Verified Synthetic Data for Robot Learning

- **作者：** Seungku Kim*，Suhyeok Jang*，Byungjun Yoon，Dongyoung Kim，John Won，Jinwoo Shin（*Equal contribution）
- **机构：** KAIST（韩国科学技术院）、RLWRLD
- **版本 / 载体：** arXiv:2602.18742v2 [cs.RO]（2026-02-21 提交 v1，2026-11-05 更新 v2）；CoRL 2026 G2A Workshop Oral（项目页标注：CoRL 2026 Workshop on The Action Gap: Large Generative Models Learn Rich Knowledge）；Survey 索引别名：`RoboCurate: Harnessing Diversity with Action-Verified Neural Trajectory for Robot Learning`（[`survey_ax_2602_18742`](../repos/awesome-agentic-robots-survey-index.md)）
- **原文：** [arXiv HTML v2](https://arxiv.org/html/2602.18742)、[arXiv Abstract](https://arxiv.org/abs/2602.18742)、[官方项目页](https://seungkukim.github.io/robocurate/)
- **项目页 / 代码：** 项目页可正常访问（仅提供 Paper / arXiv 链接及真机演示视频）；**未确认开源代码**（截至 2026-11-27 未发布 RoboCurate 本身官方 GitHub 代码仓库；论文使用并引用了外部开源依赖：`nvidia-cosmos/cosmos-predict2`、`isaac-sim/IsaacSim`、`FLUX.2-dev`、`Cosmos-Transfer2.5-2B`、`V-JEPA 2`、`GR00T N1.5` 及开源预训练 IDM 检查点 `https://huggingface.co/seonghyeonye/IDM_gr1`）
- **核对范围：** 通读 arXiv HTML v2 全部正文（Abstract、§1 Introduction、§2 Preliminaries、§3 Method、§4 Experiments、§5 Related Work、§6 Conclusion，含 Table 1–9）及全部附录（Appendix A 讨论与局限、Appendix B 训练与仿真回放及 Attentive Probe 超参表 Table 10、Appendix C 生成管线细节、Appendix D 定性示例、Appendix E Probe 在视觉/视角分布偏移下的 AUPRC/P@R 评测 Table 11、Appendix F–H 及 Appendix I.1–I.5 完整 VLM 提示词），并核对官方项目页全文
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [RoboCurate: Harnessing Diversity with Action-Verified Synthetic Data for Robot Learning](../../wiki/entities/paper-agentic-2602-18742.md)

### 1. 研究动机与核心痛点
- 基于视频生成模型 + 逆动力学模型（IDM）的合成数据管线（如 DreamGen）存在两类质量问题：
  1. **视频侧物理失真**：模型未遵循指令或生成穿模、瞬移抓取（tele-grab）等物理不合理轨迹；
  2. **动作侧跨模态不一致（video-action misalignment）**：即使生成视频本身物理合理，IDM 预测的伪动作标签 $\mathbf{A}_{\text{IDM}}$ 仍可能与视频中机械臂/灵巧手运动不一致。既有 VLM 过滤器（DreamGenBench、VideoCon-Physics）只能评估视频本身，无法直接验证动作标签。

### 2. 方法核对要点
1. **VLM 物理合理性预过滤（VLM Pre-filtering）**：
   - 使用 Gemini 2.5 Flash（输入 8 帧），按 1–5 分李克特量表从「接近目标深度/距离、抓取接触精度、搬运位移匹配」三阶段评分，默认保留得分 $\ge 3$（OpenArm 实验提高至 $\ge 4$）的视频。
2. **仿真器回放与 Attentive Probe 动作一致性验证（Action-Verified Filtering）**：
   - 将 $(\mathbf{x}_{\text{gen}}, \mathbf{A}_{\text{IDM}})$ 中的预测动作序列在物理仿真器 $f$ 中以 20 Hz 回放，渲染无物体干扰的机器人运动视频 $\hat{\mathbf{x}}_{\text{sim}} = f(\mathbf{A}_{\text{IDM}}, \tilde{\mathbf{o}})$（GR-1 用基于 robosuite 的 RoboCasa `GR1ArmsAndWaistFourierHands`；ALLEX 用 NVIDIA Isaac Sim 48-DoF USD 虚拟 ZED MINI 左目；OpenArm 用 NVIDIA Isaac Lab 28-DoF 自我中心视角），把跨模态的「视频-动作一致性检验」转化为同模态的「视频-视频运动一致性比较」。
   - **自监督正负样本构造**：仅使用真实示教 $\mathcal{T}=\{(\mathbf{x}_i^{\text{real}}, \mathbf{A}_i^{\text{real}})\}$ 与对应真实动作的仿真回放 $\mathbf{x}_i^{\text{sim}} = f(\mathbf{A}_i^{\text{real}}, \tilde{\mathbf{o}})$，无需人工标注且避免合成噪声标签干扰。正样本 $\mathcal{P}^+$ 为同片段同起始时刻对齐；负样本 $\mathcal{P}^-$ 按 45% / 45% / 10% 由同轨迹时间偏移 $\mathcal{P}^-_{\text{shift}}$（起点相差 $\ge 5$ 帧、终点相同）、同技能跨轨迹 $\mathcal{P}^-_{\text{cross}}$ 及两者组合 $\mathcal{P}^-_{\text{shift+cross}}$ 构成（正负比例 80% / 20%）。
   - **网络结构与超参（Table 10）**：冻结 V-JEPA 2 Large（300M）编码各 16 帧（$256\times 256$），拼接两个视频 token 序列后输入单层交叉注意力探针（1 层、8 头、1 个可学习 query token + 分类头），BCE 损失训练 50 epoch（AdamW, lr=1e-4, wd=0.05, batch size=72）。推理默认决策阈值 $p > 0.5$。
   - **生成时 Best-of-N Critic**：在真机微调等数据稀缺场景下，用不同高斯噪声种子采样 $N=4$ 条候选视频及 IDM 动作，经 VLM 预过滤后选择 $p$ 最高的一对。
3. **结构化场景多样化与语言重标注（Scene Diversification & Re-captioning）**：
   - 使用 Gemini 3 Pro 提取初始帧与参考指令，围绕**桌面外观（Table）、目标物体身份与外观（Target Object）、光照（Lighting）、背景与非侵入干扰物（Background）**四条轴生成反事实组合，并结合 **Canny 边缘图**结构约束：
     - **I2I 编辑**（FLUX.2-dev）：编辑初始帧生成新场景，再用 I2V 模型（Cosmos-Predict2-14B / Cosmos-Predict2.5-2B）生成新轨迹（其中 I2I 与 V2V 比例为 2:1）；
     - **V2V 迁移**（Cosmos-Transfer2.5-2B）：严格保持物体类别与形状不变、仅变换桌面/颜色材质/光照/背景以复用原 IDM 动作标签。
   - 最后使用 Gemini 3 Pro（8 帧）执行两阶段语言重标注（Language Re-captioning）：先判别是否遵循原指令，未遵循则重新生成匹配视频内容的自然语言指令。

### 3. 实验配置与关键数据核对
- **预训练实验（§4.1, Appendix B.1）**：真实数据为 ActionNet 3K 去重子集（GR1-T1, 44 维状态/动作），与 10K 过滤前合成数据按 1:1 混合预训练 GR00T N1.5（动作头从零初始化，60K 步，batch 512，前 50K 步全量合成数据、后 10K 步 WSD 衰减期仅用 RoboCurate 过滤后合成数据，真机与合成数据采用独立 embodiment tag）。
  - **GR-1 Tabletop（24 任务，每任务 50 次试验）**：300 demos 微调时，Real 15.4% → DreamGen 19.5% → RoboCurate(无过滤) 22.7% → RoboCurate(含过滤) **29.3%**（相对 Real +90.3%）；1000 demos 时 Real 27.9% → DreamGen 32.1% → RoboCurate(无过滤) 34.0% → RoboCurate(含过滤) **34.6%**。
  - **DexMimicGen（6 任务，3 个随机种子 × 50 次试验，每任务 100 demos）**：Real 44.6%（GR-1 56.9% / Panda 32.2%）→ DreamGen 46.4% → RoboCurate(无过滤) 49.3% → RoboCurate(含过滤) **57.6%**（GR-1 **72.2%** / Panda **42.9%**，相对 Real +29.1%）。
- **真机联合微调（Co-finetuning, §4.2, Appendix B.2）**：仅对分布内（ID）任务采集人工遥操作数据（ALLEX 48 条、OpenArm 44 条），未见物体/未见行为（OOD）任务完全无真机数据，仅靠合成数据与 Best-of-4 筛选（1:1:1 采样，10K 步）。
  - **ALLEX（48-DoF，GR00T N1.5，24 次试验/任务）**：Real 13.9%（PnP Can 25.0% / PnP Cup 16.7% / Pour Can 0.0%）→ DreamGen 27.8% → RoboCurate* **38.9%**（PnP Can 47.9% / PnP Cup 43.8% / Pour Can **25.0%**，相对 Real +179.9%）。
  - **OpenArm（28-DoF，RLDX-1-PT-IMG，32 次试验/任务）**：Real 21.9%（Lift Bottle 37.5% / Lift Box 6.3%）→ DreamGen 26.6% → RoboCurate* **31.3%**（Lift Bottle 46.9% / Lift Box **15.6%**，相对 Real +42.9%）。
- **消融与诊断（§4.3, Appendix E）**：
  - 过滤对比（GR-1 Tabletop 300 demos）：无过滤 22.7% → DreamGenBench 22.8% → VideoCon-Physics 23.0% → RoboCurate **29.3%**；
  - 组件拆解：+VLM pre-filter 24.2% → +Cosine Top-K 26.0% → +Attentive Probe 29.3%；
  - 人工标注 400 条合成样本中的 27 例「视频合理但动作不一致」样本：DreamGenBench 仅剔除 **3.7%**，RoboCurate 剔除 **70.4%**；
  - 编码器替换：VideoMAEv2 (83M) 为 25.4%，V-JEPA 2 (300M) 为 29.3%；阈值在 0.10 / 0.25 / 0.50 分别为 25.2% / 26.6% / 29.3%；在无场景多样化的 0.3K 真机 + 1K DreamGen 设定下，过滤仍将成功率从 15.8% 提升至 20.0%；
  - Probe 分布外测试（Table 11）：ActionNet 验证集 AUPRC 0.907，仿真相机偏置 +0.1 m & FOV +10° 为 0.904，跨数据集视觉变化（GR00T-Teleop-GR1）为 0.865。
