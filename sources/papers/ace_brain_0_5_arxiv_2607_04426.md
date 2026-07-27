# ACE-Brain-0.5：A Unified Embodied Foundational Model for Physical Agentic AI

> 来源归档（ingest）

- **标题：** ACE-Brain-0.5: A Unified Embodied Foundational Model for Physical Agentic AI
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2607.04426>
- **PDF：** <https://arxiv.org/pdf/2607.04426>
- **机构：** ACE-Brain Team / 大晓机器人（Ace Robotics）；作者含 Ziyang Gong、Haoming Gu、Zehang Luo、Tianyi Zhang、Tao Tao、Yixiao Chi、Zhe Liu、Lingsi Zhu、Jingyuan Liu、Anke Tang、Zhi Hou、Xue Yang、Dacheng Tao、Xiaogang Wang 等
- **项目页：** <https://ace-brain-team.github.io/ACE-Brain-0.5/>
- **代码：** <https://github.com/ACE-BRAIN-Team/ACE-Brain-0.5>（镜像 / 组织页亦可指向 `DAXIAORobotics/ACE-Brain-0.5`）
- **权重：** <https://huggingface.co/ACE-Brain/ACE-Brain-0.5-8B>（`Qwen3VLForConditionalGeneration`；亦见 `ACERobotics/ACE-Brain-0.5-8B`）
- **骨干：** ACE-Brain-0 / **Qwen3-VL-8B**；DINOv3 Fast Vision + flow-matching Action Expert
- **训练范式：** **SSR+**（Scaffold–Specialize–Reconcile + **Reactivate**）
- **入库日期：** 2026-07-27
- **一句话说明：** 在 ACE-Brain-0 空间智能脚手架上，用单一 **8B** 骨干统一 **感知–规划–动作–评估** 闭环（导航/操作 + 进度估计），并以外部执行状态实现部署期自改进；**权重与 HF 推理已开放，训练栈未见公开。**

## 核心摘录（面向 wiki 编译）

### 1) 统一具身脑：五认知功能 + 单一 8B 骨干

- **链接：** arXiv §1 Introduction；§3.1 Model Architecture；Figure 1–2；Table 1
- **摘录要点：**
  - 相对模块化 Sense–Plan–Act、端到端 VLA/WAM、以及多模型 robot-agent，论文主张 **Unified Embodied Foundation Model**：在单一共享表征上耦合五功能——**Spatial Perception / Decision Making / Embodied Interaction / Self Monitoring / Self Improvement**。
  - ACE-Brain-0 建立跨平台空间脚手架，但缺统一动作接口与执行自评估；0.5 把前四功能收进同一 **mixture-of-transformer**，第五功能由伴随框架更新外部执行状态 \(\mathcal{H}\)（任务图式、空间记忆、失败恢复案例）。
  - Table 1：相对 RynnBrain、Cosmos3、π 系、QwenVLA、ABot、QwenRobot 等，宣称同时覆盖 **物体感知、空间理解、任务规划、端到端导航、端到端操作、执行监控、自改进**。
- **对 wiki 的映射：**
  - [ACE-Brain-0.5](../../wiki/entities/paper-ace-brain-0-5.md) — 范式定位与能力对照
  - [Foundation Policy](../../wiki/concepts/foundation-policy.md) — 具身基础策略族谱
  - [RynnBrain 1.1](../../wiki/entities/paper-rynnbrain-1-1.md) — 强空间理解但缺动作/进度闭环的对照

### 2) 架构：共享 LLM 状态 + Fast Vision + flow Action Expert

- **链接：** arXiv §3.1；Figure 2；式 (1)–(3)
- **摘录要点：**
  - **Omni-Vision Encoder** 统一单视角 / 多视角 / 视频；语言与视觉在 **LLM Decoder** 融合为共享具身状态 \(s_t=F_\theta(\ell,o_t,q_t)\)。
  - **Action Expert** 为 π₀ 式 **flow-matching** 头；**DINOv3 Fast Vision** \(z_t=E_{\mathrm{fast}}(o_t)\) 以控制频率注入，与缓存的高阶 \(s_t\) 组成双时间尺度控制，避免每步都过重 LLM。
  - 解码路径：框/掩码/点（感知）→ 自然语言子目标（决策）→ 离散导航动作或连续操作 chunk（交互）→ 帧级进度 \({\hat p}_t\in[0,1]\) 与轨迹偏好（自监控）。
  - 操作学习时 **冻结 VLM 骨干**，只训 Fast Vision + Action Expert；另有 **ACE-Brain-0.5-VLA** 变体：不训 FastVision，全量微调 VLM + 轻量 flow 头（SimplerEnv-Bridge）。
- **对 wiki 的映射：**
  - [ACE-Brain-0.5](../../wiki/entities/paper-ace-brain-0-5.md) — 方法栈与流程图
  - [VLA](../../wiki/methods/vla.md) — flow-matching 动作专家对照
  - [Qwen-VLA](../../wiki/entities/qwen-vla.md) — 同 Qwen3-VL 族通才对照

### 3) SSR+：Scaffold → Specialize → Reconcile → Reactivate

- **链接：** arXiv §3.2；式 (4)–(7)；Appendix A.5
- **摘录要点：**
  - 异构监督（文本 QA/规划、结构化 grounding、导航离散动作、连续操作、进度序列）直接混训易 **交叉接口干扰**；完全隔离则得不到统一模型。
  - **Scaffold**：ACE-Brain-0 / Qwen3-VL-8B-Instruct 空间脚手架 \(\theta_0\)。
  - **Specialize**：独立训 \(\theta_{\mathrm{qa}},\theta_{\mathrm{grd}},\theta_{\mathrm{nav}},\theta_{\mathrm{prog}}\)。
  - **Reconcile**：优化式 task-vector merging（FusionBench，约 1000 次 data-free Adam），层内最小化与各专家输出残差。
  - **Reactivate（新）**：合并后语义知识已在，但输出格式约定失同步；用紧凑混合 SFT \(\mathcal{D}^{\mathrm{mix}}\) 轻量校准，远少于 Specialize 步数即可恢复多接口切换。
- **对 wiki 的映射：**
  - [ACE-Brain-0.5](../../wiki/entities/paper-ace-brain-0-5.md) — SSR+ 四阶段表
  - [Foundation Policy](../../wiki/concepts/foundation-policy.md) — 多能力统一训练工程读法

### 4) 评测：空间 / 导航 / 操作 / 进度

- **链接：** arXiv §4；Tables 2–6
- **摘录要点：**
  - **空间：** 相对 ACE-Brain-0，**18** 项空间感知/grounding 中 **14** 项提升；例 MindCube **86.3%**（0: 82.1）、RefSpatial **55.6%**（0: 26.0）、RoboAfford **75.1%**（0: 56.5）、ShareRobot-Traj 误差 **0.32**（0: 0.46）。驾驶子榜相对 0 代部分回落（统一脑非驾驶专精）。
  - **导航 VLN-CE：** 统一模型 R2R Val-Unseen SR **57.4%** / NE **4.8**；RxR SR **63.8%** / NE **4.3**（统一模型多项领先开源基线）；Specialist 进一步抬 R2R SR **62.2%**。
  - **操作：** LIBERO 均 **98.2%**（Spatial/Object **100%**，Long **97.0%**）；ACE-Brain-0.5-VLA 在 SimplerEnv-Bridge 均 **82.3%**（SOTA，Eggplant **100%**）。
  - **进度：** RBM-EVAL VOC — Standard ID/OOD **0.94 / 0.96**；Refined（含反转成功轨迹负控）**0.80 / 0.88**，优于 Robometer 与若干专用奖励模型。
- **对 wiki 的映射：**
  - [ACE-Brain-0.5](../../wiki/entities/paper-ace-brain-0-5.md) — 结果表
  - [过程奖励建模](../../wiki/concepts/progress-reward-modeling.md) — 进度估计作为自监控接口
  - [Vision-Language Navigation](../../wiki/tasks/vision-language-navigation.md) — VLN-CE 语境

### 5) 自改进与开源边界

- **链接：** arXiv §3.3、§4.5；项目页 / GitHub / HF（2026-07-27 核查）
- **摘录要点：**
  - 自改进优先更新外部执行状态 \(\mathcal{H}\)，不必每轮重训；导航上用闭环节失败 + oracle 接管构造纠正经验 \(\mathcal{D}_{\mathrm{evo}}\)。
  - **开源：** 项目页链 **Report / Code / Model**；HF 提供完整 **safetensors** 与 `transformers` 推理样例；GitHub 仓主体为 **README + assets**，**未见** 训练脚本 / Action Expert 训练入口 → 记为 **部分开源**。
- **对 wiki 的映射：**
  - [ACE-Brain-0.5](../../wiki/entities/paper-ace-brain-0-5.md) — 工程实践与源码时序图
  - [sources/sites](../sites/ace-brain-0-5-github-io.md)、[sources/repos](../repos/ace-brain-0-5.md)

## 当前提炼状态

- [x] arXiv PDF / HTML 摘录对齐（2607.04426）
- [x] 项目页 / GitHub / HF 开源边界核查（2026-07-27）
- [x] wiki 映射：`wiki/entities/paper-ace-brain-0-5.md` 新建
- [ ] 待官方发布训练代码 / Action Expert 权重后补全复现时序
