# WAM-TTT：Steering World-Action Models by Watching Human Play at Test Time

> 来源归档（ingest）

- **标题：** WAM-TTT: Steering World-Action Models by Watching Human Play at Test Time
- **类型：** paper
- **arXiv abs：** <https://arxiv.org/abs/2607.06988>
- **arXiv HTML：** <https://arxiv.org/html/2607.06988v2>
- **机构：** 北京大学（PKU）· 银河通用（Galbot）· 中科院自动化所（CASIA）· 清华大学（Tsinghua）
- **骨干：** **LDA-1B**（latent dynamics action WAM，arXiv:2602.12215）
- **入库日期：** 2026-07-16
- **一句话说明：** 冻结 **LDA WAM** 上外挂 **Spatial-TTT 式 fast-weight 记忆分支**：meta-training 用 **2286 对** 人–机相位同步示教 + **KV 记忆重建损失** 对齐人视频与机器人 Query；部署时仅用 **无标注 egocentric 人视频** 做 **自监督视频预测 TTT** 更新 fast weights，即可 **steer** 新任务变体，无需机器人动作、人手姿态或全模型微调；**9 项真机**（G1 人形 + Galbot 双臂）**New 家庭场景** 平均 **46.2%** progress，显著优于 **WAM-ICL（7.1%）** 与同骨干 **LDA（32.5%）**。

## 核心摘录（面向 wiki 编译）

### 1) 问题：RFM 部署后难以用人类示教快速转向

- **链接：** §1 Introduction
- **摘录要点：** 现有 RFM 知识固化在预训练权重与有限条件接口（语言、目标图、短历史）；转向新任务变体、物体交互或用户偏好策略通常需 **额外机器人示教** 或 **全模型微调**，限制开放部署中的灵活复用。人类视频是自然可扩展接口，但既有路线多依赖 **手姿/3D 运动/重定向轨迹** 等昂贵监督，或 **in-context 人视频条件** 导致上下文长度快速增长。
- **对 wiki 的映射：**
  - [WAM-TTT](../../wiki/entities/paper-wam-ttt-human-video-test-time-steering.md) — 部署时 steering 问题定义
  - [World Action Models](../../wiki/concepts/world-action-models.md) — WAM 联合视频–动作表征语境

### 2) 方法：人视频作 TTT 记忆，而非模仿轨迹

- **链接：** §3 Method；Figure 2
- **摘录要点：**
  - **底座：** LDA 每个 diffusion transformer block 含 **video expert** 与 **action expert** 联合注意力；仅在 **video expert** 加 **TTT 残差分支** \(\Delta z_{\mathrm{TTT}}\)，action 流不变。
  - **Meta-training（人–机配对）：** 对 action-free 人视频 clip \(u_h\) 与同步机器人轨迹，内环 SGD 最小化 **人侧视频预测** \(\mathcal{L}_{\mathrm{vg}}^{\mathrm{human}}\) + **逐层 KV 记忆重建** \(\mathcal{L}_{\mathrm{KVM}}^{(\ell)}=\|f_W(K_h)-V_h\|^2\)；外环在机器人侧标准 **LDA 多任务扩散损失** \(\mathcal{L}_{\mathrm{WAM}}^{\mathrm{robot}}\) 反传穿过内环更新。
  - **Test-time：** 冻结 WAM、TTT 慢投影与 \(W_{\mathrm{init}}\)；仅用目标域 **无标注人视频** 批次 \(\mathcal{B}_h\) 做与 meta-training 同形的内环 TTT；固定 fast weights \(W_N\) 后 rollout 采样动作 chunk。
  - **相位同步：** 机器人时刻 \(t/T_r\) 映射到配对人视频最近相位帧，无需显式 retargeting。
- **对 wiki 的映射：**
  - [WAM-TTT](../../wiki/entities/paper-wam-ttt-human-video-test-time-steering.md) — 两阶段 Mermaid 与损失表
  - [RoboTTT](../../wiki/entities/paper-robottt-test-time-training-vla-context.md) — 同名 TTT、不同信息源与层设计对照

### 3) 实验设置与主结果

- **链接：** §4 Experiments；Table 1
- **摘录要点：**
  - **平台：** Unitree **G1**（人形）、Galbot **gripper**（双臂二指）、Galbot **sharpa**（双臂灵巧手）；**9 项** 操作任务（Transfer Bottle、Table Bussing、Deliver Drink、Swap Place、Pour Water、Stamp Paper、Flip Steak、Pyramid Stacking、Multi-step Steak）。
  - **设定：** **Orig.** 训练同 cubicle；**New** 未见家庭环境（光照/桌高/物体联合 OOD）。指标 **progress (%)**，每格 **25** 次试验。
  - **Meta 数据：** **2286** 对 egocentric 人–机 episode（人侧 GoPro，无姿态估计）；覆盖 9 任务。
  - **New 环境平均：** WAM-TTT **46.2%** > LDA **32.5%** > WAM-Cotrain **25.3%** > EgoScale **15.0%** ≈ π₀.₅ **14.8%** > WAM-ICL **7.1%**。
  - **关键对照：** 同人视频下 **WAM-ICL**（上下文条件、无 fast-weight 适应）几乎失效，支持「吸收为记忆」而非「堆上下文」假设。
- **对 wiki 的映射：**
  - [WAM-TTT](../../wiki/entities/paper-wam-ttt-human-video-test-time-steering.md) — 结果表与平台矩阵
  - [Manipulation](../../wiki/tasks/manipulation.md) — 多具身真机操作 benchmark 锚点

### 4) 消融与泛化保持

- **链接：** §4.3–4.4；Table 2–3
- **摘录要点：**
  - **消融（Table Bussing / Swap Place）：** 完整 WAM-TTT **100% / 88.9%**；去 meta-training **9% / 0%**；去 memory recon. **66.7% / 72%**；去 TTT **40% / 74.1%**；**WAM-LoRA** **30% / 0%** — TTT 记忆结构优于通用低秩适应。
  - **扰动泛化（Deliver Drink）：** 光照扰动 WAM-TTT **66%** vs WAM-ICL **12%**；空间扰动 **56%** vs **20%** — 适应后仍保留预训练 WAM 鲁棒性。
- **对 wiki 的映射：**
  - [WAM-TTT](../../wiki/entities/paper-wam-ttt-human-video-test-time-steering.md) — 消融与局限
  - [EgoWAM](../../wiki/entities/paper-egowam-egocentric-human-wam-co-training.md) — 训练期人–机共训对照

### 5) 局限与展望

- **链接：** §5 Conclusion
- **摘录要点：** (1) meta-training 假设配对人 episode 与机器人 **技能相位分布对齐**，错位会无声劣化内环信号；(2) fast-weight 表达力与 meta 分布外任务漂移边界未充分刻画；(3) 部署接口仅 **egocentric RGB**，未用手姿/接触/3D 场景线索。
- **对 wiki 的映射：**
  - [WAM-TTT](../../wiki/entities/paper-wam-ttt-human-video-test-time-steering.md) — 局限小节

## 当前提炼状态

- [x] arXiv HTML v2 摘要 + 方法/实验主文已对齐摘录
- [ ] 项目页 / 代码公开后补链
