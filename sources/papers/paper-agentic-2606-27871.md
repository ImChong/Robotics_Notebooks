# LocalNav: Distilling Frontier VLMs and Embodied RL for On-Device Object Goal Navigation

- **作者：** Nicolas Baumann, Liam Boyle, Pu Deng, Edoardo Ghignone, Boyang Sun, Marc Pollefeys, Luca Benini, Michele Magno
- **机构：** ETH Zürich（Center for Project-Based Learning / Integrated Systems Laboratory / Computer Vision and Geometry）
- **版本 / 载体：** arXiv:2606.27871v2 [cs.RO]（2026-06-26 提交 v1，2026-10-31 修订 v2）
- **原文：** [arXiv HTML v2](https://arxiv.org/html/2606.27871) / [arXiv Abs](https://arxiv.org/abs/2606.27871)
- **项目页 / 代码：** 部分开源 / 未确认官方源码（论文 §III-C 脚注 1 与 Appendix -A 脚注 3 声明在 Hugging Face 公开 Gemini 3.1 Pro、GPT-5.4、Claude Sonnet 4.6 与 Combined SFT 蒸馏数据集；截至 2026-11-27，arXiv HTML 原文及元数据未提供独立项目页或官方训练/推理代码仓库 URL，文中所引 `JuliusBrussee/caveman` 与 `ggml-org/llama.cpp` 分别为外部提示词技能与推理后端依赖）
- **核对范围：** arXiv:2606.27871v2 全文（Abstract、§I–§VII、Table I–VI、Fig. 2–10 文字说明及 Appendix -A 至 -E，含提示词样例、动作空间定义、奖励权重与 Hydra/Habitat/LoRA/GRPO 超参数表）
- **通读：** 已完成
- **入库日期：** 2026-11-27

## 原文核对笔记

- **核心动机与系统定位（§I–§III-A）：** 开放词汇物体目标导航（ObjectNav）现有两条主流路线各有瓶颈：低层 VLA 策略需在 5–10 Hz 持续推理，难以在嵌入式机器人 GPU 上实时运行；高层场景图（SG）推理仅在稀疏决策点调用 VLM，但通常依赖云端大模型（约 10.86 s 往返延迟，均值测量自 475 次 Gemini 3.1 Pro 查询）。LocalNav 采用「模块化高层 SG 推理 + 底层 PointGoal 导航解耦」架构，目标是在 Jetson Orin AGX 上本地运行 4B 级 VLM（Qwen3.5-4B），同时匹配云端精度与端侧延迟预算（目标约 200 输出 token/次，为并发感知与建图留出算力余量）。
- **双通道 Scene Graph 接口与动作空间（§III-A、§III-B、Appendix -B）：**
  - 由 **Hydra** 在线构建 3D 场景图（仿真使用 Habitat 语义真值，真机使用 Mask2Former + ZED2i RGB-D + VIO）；每步将房间/物体 ID 列表写入文本提示词，并将视野内 SG 物体 ID 按 Set-of-Mark 风格投影到 RGB / 360° 全景图像上。
  - VLM 在结构化输出 `<HISTORY>`、`<REASONING>`、`<ACTION>` 中预测高层动作 $\mathcal{A}_{\text{SG}} = \{\texttt{navigate(object\_id)}, \texttt{explore(room\_id)}, \texttt{find\_new\_rooms()}, \texttt{done()}\}$。
  - 仿真底层采用 **DD-PPO** PointGoal 策略输出 Habitat 离散动作 $\mathcal{A}_{\text{LL}}$；真机（**Unitree A2-W** 四足机器人或手持 **LocalNav Module**）采用基于 VAE 与 **SRU-RNN** 的 PointGoal 策略，直接输出连续速度 $\{v_x, v_y, v_\omega\}$。
- **教师轨迹蒸馏与 E-RLVR 闭环强化学习（§III-C、§III-D、Appendix -E）：**
  - **SFT 数据收集：** 仅在 HM3D OVON `train` 集使用特权动作空间 $\mathcal{A}_{\text{priv}}$（Habitat 最短路径按 0.5 m 采样传送，排除 PointGoal 跟踪误差）采集成功轨迹；每个教师模型仅需 481–532 对图像-输出样本，用 rank-16 LoRA（冻结视觉层）微调 Qwen3.5-4B。在单教师蒸馏中，**Claude Sonnet 4.6** 教师效果最佳（35.23% SR），高于 Gemini 3.1 Pro（30.63% SR，尽管 Gemini 自身零样本达 41.61% SR）与 GPT-5.4（28.63% SR）。
  - **E-RLVR（GRPO）：** 在 Habitat 闭环环境中每步采样 $N=4$ 个动作并分叉仿真分支执行，以环境可验证回报 $R = \sum \lambda_i R_i$（$\lambda_{\text{done}}=2.0, \lambda_{\text{nav}}=1.0, \lambda_{\text{exp}}=1.0, \lambda_{\text{brevity}}=0.25$）计算组内相对优势更新 LoRA（$r=16, \alpha=32, \beta=0.04$）。不加长度惩罚时 SR 升至 **37.36%**（SPL 18.01%）；加入线性长度惩罚 $R_{\text{brevity}}$ 后，平均输出从 424.20 token 降至 **165.41 token**（每轮查询降至 5.95 次、重复率 0.04），SR 保持 **35.53%**。
- **端侧性能剖析与量化加速（§IV-B–§IV-D）：**
  - 在 Jetson Orin AGX（BF16）上剖析 SFT-Claude 推理耗时：Prefill 平均 2746.5 token（1271.4 tok/s，16.01 s），自回归 Token Generation（TG）平均 486.71 token（18.04 tok/s，197.88 s），TG 占单回合端到端延迟（213.89 s）约 92.5%。
  - 结合 **llama.cpp IQ4-XS** 量化（在 KLD 与 TG 吞吐量 Pareto 前沿上选取），TG 吞吐从 18.04 tok/s 提升至 **40.31 tok/s**，单回合总推理延迟从 213.89 s 降至 **42.32 s**（累计降低 **80.21%**），SR 保持 **34.51%**。
- **核查边界与局限（§IV-E、§IV-F、§VI）：** HM3D OVON `val_unseen`（3000 集，0.25 m VIEW_POINTS）评测中，LocalNav 依赖语义传感器构建 SG，与仅用 RGB / RGB-D 的基线存在传感器输入差异；真机实验为 4 个连续室内查询（手持模块）及 Unitree A2-W 单次属性目标演示，尚无大规模统计真机评测；静态 SG 无法原生绑定瞬态组合语义（如「有人坐着的椅子」），仍需 VLM 抽样推理确认。
- **对 wiki 的映射：** [LocalNav: Distilling Frontier VLMs and Embodied RL for On-Device Object Goal Navigation](../../wiki/entities/paper-agentic-2606-27871.md)
