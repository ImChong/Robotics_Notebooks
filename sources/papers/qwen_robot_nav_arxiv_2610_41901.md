# Qwen-RobotNav Technical Report: A Scalable Navigation Model Designed for an Agentic Navigation System

- **作者：** Qwen Team（核心贡献者：Jiazhao Zhang, Gengze Zhou, Hale Yin, Yiyang Huang, Zixing Lei, Qihang Peng 等；项目指导：Chenxu Lv, Chenfei Wu, Xiong-Hui Chen）
- **链接：** https://arxiv.org/abs/2606.18112 （注：本归档沿用集合期索引文件名 `2610_41901`，正文与仓库 BibTeX 核验正式 arXiv 编号为 `2606.18112`，v1 发布于 2026 年 6 月 17 日；官方页 https://qwenlm.github.io/Qwen-RobotNav/ ）
- **状态：** primary_full_text_checked
- **核对范围：** 已通读 `arXiv:2606.18112` 技术报告全文（1–3999 行：模型架构 `Qwen3-VL` + MLP 路径点动作头、任务自适应观测编码 Algorithm 1、两层智能体导航系统与 Evidence Notebook、15.6M 多源共训配方、VLN-CE / ObjNav / EVT-Bench / EQA / NAVSIM / AlpaSim 及 Unitree Go2 实机延迟与导航实验）
- **通读：** 已完成

## 核心要点提取

1. **轻量耦合与任务自适应观测编码（Task-Adaptive Observation Encoding）**
   - 基座采用 **Qwen3-VL**（SigLIP-2 动态分辨率 ViT + 2D-RoPE + `s_m^2` 两层 MLP patch merger + DeepStack 多尺度视觉特征），末端仅接 **4 层 MLP 动作头**（隐藏维 512，GELU）直接输出局部坐标下 **8 步连续路点 `(x_k, y_k, theta_k)`**（共 24 维，按各数据集 p99 分位归一化）；总损失 `L = L_traj + 1.0 * L_VL`。提供 **4B** 与 **8B** 两个规模版本（8B 模型全局 batch size=256，约消耗 2,816 H100 GPU 小时，骨干 lr `2e-5` / 动作头 lr `1e-4`）。
   - **Algorithm 1 动态预算分配**：围绕计算预算 `B in [2048, 4096]` tokens、时间衰减系数 `gamma in [1, 3]`、多相机权重 `{w_c}` 及历史帧采样策略 `m in {random, latest}`，训练时随机化每帧每相机 token 上下界 `b_min in [1, 8]`、`b_max in [128, 256]`，由 `ConstrainedAlloc` 迭代截断分配并按比例重缩放输入图像，再附加自然语言时空标签（如 `Time step t Front View <image>`）与载体前导提示（`robot` vs `car`）。推理时无需重新训练即可按任务需求调节延迟-视野权衡。

2. **两层 Agentic 导航架构与 Trajectory-to-Evidence 闭环**
   - 上层多模态编排智能体（技术报告评测中使用 `Qwen3.6-Plus`，早期社区预览曾提及其演进系列）结合全局状态、检索视图工具与可持久化 **Evidence Notebook** 进行长程任务拆解、记忆检索与错误重规划；通过统一工具调用 `W_i = nav_qwennav(L_i, tau_i, Phi_i)` 驱动底层 `Qwen-RobotNav` 执行 `tau_i in {VLN, PointNav, ObjNav, Tracking}` 子任务。
   - **Trajectory-to-Evidence 封装器**：底层滚动执行后返回结构化执行摘要（`subgoal`、`task_mode`、`config`、`progress`、`salient`、`outcome`、`key_frames`），交由上层判断已完成、受阻换路还是触发跨集语义召回。

3. **15.6M 多任务混合数据与跨基准实测**
   - 85% 导航轨迹（约 13.3M 样本）：VLN 指令跟随 5.63M（R2R 1.491M + RxR 4.140M，含 Qwen-Image-Edit 视效增强）、PointNav 984K、ObjNav 2.0M（骨架中轴探索 + VLM 目标标注 + HM3D-OVON）、人员跟踪 EVT-Bench 1.486M、自动驾驶 ~3.2M（nuScenes 78K + OpenScene 3.138M）及 40K T2V 合成探索。15% 图文/具身理解语料（~2.23M）：一般 VL 1.0M、4 视角结构化导航推理 873K、MP3D 多轮对话离散轨迹 362K。
   - **核心指标**：VLN-CE 全景 R2R / RxR 下 8B 达 **72.1% SR / 66.6% SPL** 与 **76.5% SR / 65.7% SPL**（超越 OmniNav `69.5/73.6`、ABot-N0 `66.4/69.3`）；单目 R2R 4B 达 `66.9% SR`；HM3Dv2 ObjNav（纯 RGB）4B 达 **75.6% SR / 30.6 SPL**；EVT-Bench 跟踪率 `90.0%`（4B）/ `89.7%`（8B），碰撞率低至 `5.70%`（但纯跟踪 SR `78.6%` 略逊于专精跟踪模型 TrackVLA++ `86.0%`）；搭配 `Qwen3.6-Plus` 在 HM-EQA 达 **76.7%**（仅需 0.15 相对路径步长，FAST-EQA 为 `69.2% / 0.65`）、EXPRESS-Bench `79.27`；NAVSIM 4B 达 **91.4 PDMS**（剥离自车历史骤降至 `79.5`）；AlpaSim 零样本泛化得分 `0.15 / 0.17`（低于 Alpamayo-R1-10B `0.72`）。
   - **真机实测（Unitree Go2）**：远程服务约 **196 ms（5.1 Hz）**，边缘端 **Jetson Thor** 经 FP8 + TensorRT 量化部署达 **204 ms（4.9 Hz）** 且时延抖动更低；实测完成 21.78 m 折返指令导航与 Cotti Coffee 绿伞目标自主探索。

## 局限性与边界条件

- 模型权重与推理代码目前尚未开源。
- 在高频动态追踪（EVT-Bench SR）与极端长尾闭环驾驶仿真（AlpaSim）中仍落后专精单任务模型，上层多模态 LLM 推理延迟也制约高频反应式重规划。
