# OnFly: Onboard Zero-Shot Aerial Vision-Language Navigation toward Safety and Efficiency

- **作者：** Guiyong Zheng, Yueting Ban, Mingjie Zhang, Juepeng Zheng, Boyu Zhou（通讯作者）
- **机构：** 中山大学人工智能学院（SYSU）、南方科技大学（SUSTech）、香港科技大学（广州）（HKUST-GZ）
- **版本 / 载体：** arXiv:2603.10682v1 [cs.RO]（2026-03-11 提交，HTML 版）；官方 GitHub 仓库简介标注 `[IROS'26]`
- **原文：** [arXiv HTML 全文](https://arxiv.org/html/2603.10682) / [arXiv 摘要页](https://arxiv.org/abs/2603.10682)
- **项目页 / 代码：** 宣称将开源或未列链接 — 官方仓库 [Robotics-STAR-Lab/OnFly](https://github.com/Robotics-STAR-Lab/OnFly)（约 105 stars）当前仅含 `README.md` 与 `assets/bg.mp4`，明确标注 `Code coming soon.`；项目页入口为 [robotics-star.com/OnFly/](https://robotics-star.com/OnFly/)，截至 2026-11-27 尚未发布可运行源码。
- **核对范围：** 完整通读 arXiv HTML 全文（Abstract、§I Introduction、§II Related Work、§III Problem Formulation、§IV Methodology §IV-A–§IV-D、§V Experiments §V-A–§V-E、§VI Conclusion、Table I–IV、Fig. 1–6 图注与 References [1]–[21]）以及官方仓库页面 `Robotics-STAR-Lab/OnFly` 的 `README.md` 与目录样本（`README.md`、`assets/bg.mp4`）。
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [OnFly：面向安全与效率的机载零样本空中视觉语言导航](../../wiki/entities/paper-agentic-2603-10682.md)
- **核心问题（§I–§III）：** 现有零样本空中视觉语言导航（AVLN）面临三类耦合瓶颈：(1) 单流 VLM 推理同时承担高频目标生成与低频进度监控，造成频率失配与目标冲突；(2) 滑窗记忆会覆写早期观测，丢失全局轨迹上下文并破坏 KV-cache 前缀稳定性；(3) 短时域保守动作导致严重的停顿-前进（stop-and-go）低效，而激进长距离目标预测缺乏系统性几何安全校验易引发碰撞。
- **双智能体与混合记忆（§IV-A, §IV-C）：**
  - **任务管理（Task Management）：** 收到长程指令 $\mathcal{L}$ 后由机载 VLM 一次性拆分为子任务队列 $\{\mathcal{L}_0, \dots, \mathcal{L}_n\}$，在监控智能体输出稳定 `STOP` 后顺序推进。
  - **决策智能体（Decision Agent）：** 高频输出图像平面目标像素 $(p_x, p_y)$，在提示词 $\mathcal{P}_{\text{dec}}$ 中重投影上一帧 3D 目标为 2D 历史点线索以缓解单帧遗忘；输出候选目标 $G_c = [p_x, p_y, \mathrm{depth}_t, \mathbf{f}_t, \mathrm{pose}_t]$。
  - **监控智能体（Monitoring Agent）：** 低频（每 2 s）输出强约束状态 $s_t \in \{\texttt{CONTINUE}, \texttt{STOP}, \texttt{LOST}\}$；触发 `LOST` 时无人机悬停并转向至上一正常点航向进行恢复。
  - **共享感知、独立 KV-cache：** 两个智能体共享同一 ViT 编码器与缓存视觉特征 $\mathbf{f}_t$，但各自维护独立的提示词、上下文与解码器 KV-cache。
  - **混合记忆（Hybrid Memory）：** 构造 $\mathcal{M}_t = [o_1, \mathcal{Q}_t, o_t]$（首帧 + 去重关键帧 + 最新帧）。关键帧经过池维护（ViT 特征余弦距离阈值 $\epsilon$ 去重）、轨迹等距分段（$S=4$ 段，粘滞选择策略）及前缀稳定序列化（优先保留 $\mathcal{Q}_{t-1}$ 最长有效前缀并按时间排序），兼顾长程覆盖与 KV-cache 前缀复用。
- **语义-几何验证与局部规划（§IV-B）：**
  - **语义引导目标修正：** 当深度 $d \le D_{\max}$（$7\,\mathrm{m}$）时，在局部 ROI 内基于缓存 ViT 特征相似度 $\tau=0.5$ 提取与原始像素 $\mathbf{p}$ 连通的区域 $\mathcal{C}_{\mathbf{p}}$，并在膨胀半径 $r=0.2$ 的深度安全可行域 $\mathcal{S}_t$ 内选取最近邻可行像素 $\mathbf{p}'$。
  - **方位角感知几何门控：** 根据水平方位角 $\theta$ 对有效深度施加高斯衰减 $d_f = d'\exp(-\frac{1}{2}(\frac{\theta}{\sigma_\theta \theta_{\max}})^2)$，避免目标靠近图像边缘时无人机在航向未对准前过度前冲。
  - **局部规划器：** 基于 Fast Planner 与局部 ESDF 地图生成满足动力学约束（$v_m=0.6\,\mathrm{m/s}, a_m=0.6\,\mathrm{m/s}^2, \omega_m=0.4\,\mathrm{rad/s}$）的无碰撞轨迹，并加入目标方位角偏航正则项 $J_\psi = \lambda_\psi \|\mathrm{wrap}(\psi_t - \psi_g)\|^2$。
- **机载部署与关键实验数据（§IV-D, §V）：**
  - **仿真基准（UE 4.27 + AirSim）：** 10 个场景、150 个任务（40% 目标导航、40% 精细导航、20% 长程导航），统一使用 `Qwen3-VL-4B-AWQ`（RTX 4090）。OnFly 达到 SR `67.8%`、OSR `78.1%`、CR `2.7%`、FT `27.1 s`，显著优于 TypeFly（SR `5.7%` / CR `66.5%`）、PIVOT（`10.3%` / `64.1%`）、SPF（`26.4%` / `42.7%`）及加装避障规划器的 `SPF w/ planner`（SR `43.2%` / CR `4.7%` / FT `50.2 s`）。
  - **消融实验（Table III）：** 去除 Dual Agent 后 SR 降至 `48.6%`（FT 升至 `44.1 s`）；去除 Verification 后 SR 降至 `63.7%`（OSR 降至 `72.6%`）；去除 Planner 后 CR 飙升至 `37.5%`（SR 降至 `35.8%`）；Hybrid Memory（SR `67.8%`）优于 Time Sampling（`64.2%`）与 Sliding Windows（`60.4%`）；VLM 随 `2B` / `4B` / `30B-A3B` 扩展，SR 从 `43.4%` 提升至 `67.8%`、进一步达 `71.8%`。
  - **机载算力实测（Jetson Orin NX 16GB, Table IV）：** 结合 TensorRT、LLM AWQ 量化、ViT FP16、特征缓存、KV-cache 复用与 CUDA Graphs，决策延迟由 Transformers 基线的 `3.83 s` 降至 `0.81 s`（`4.73×`），监控延迟由 `7.47 s` 降至 `1.98 s`（Edge-Deploy）并经 Hybrid Memory 进一步降至 `1.15 s`（`6.50×`）。
  - **真机验证（§V-D）：** 自研四旋翼搭载 Jetson Orin NX 16GB、Mid360 LiDAR、D435 RGB-D 与 PX4，完成环形行人跟踪、长距走廊导航、户外台阶草坪边缘精细导航及跨护栏上二楼平台避障找人 4 项定性飞行演示。
