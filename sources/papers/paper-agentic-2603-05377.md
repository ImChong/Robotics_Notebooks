# OpenFrontier: General Navigation with Visual-Language Grounded Frontiers

- **作者：** Esteban Padilla-Cerdio\*、Boyang Sun\*、Marc Pollefeys、Hermann Blum
- **机构：** ETH Zurich、Microsoft Spatial AI Lab、University of Bonn
- **版本 / 载体：** arXiv:2603.05377v3 [cs.RO]（2026-06-28；v1 提交于 2026-03-05；仓库标注 RSS 2026）
- **原文：** <https://arxiv.org/html/2603.05377>（摘要页 <https://arxiv.org/abs/2603.05377>）
- **项目页 / 代码：** 已开源 — 官方代码仓库 <https://github.com/cvg/OpenFrontier>（项目主页 <https://boysun045.github.io/OpenFrontier-Project/>）
- **核对范围：** 已通读 arXiv HTML v3 全文（Abstract、§I–§VI、Table I–II、Algorithm 1–4、Appendix S1–S2 与 Table S1–S2），并核对官方 GitHub 仓库 README 与文件列表采样
- **通读：** 已完成

## 原文核对笔记

1. **核心问题与设定：** 面向开放世界目标导向导航（ObjectNav / Open-Vocabulary Navigation），避免依赖稠密 3D 语义建图、任务专属 RL/IL 训练或 VLN/VLA 端到端微调；将导航建模为「稀疏子目标识别与到达」问题。
2. **视觉前沿语义锚定（§III-A）：**
   - 沿用 FrontierNet 从带位姿的关键帧 RGB 图像 $\mathbf{I}_t$ 中直接检测 2D 视觉前沿簇 $\mathbf{F}_i$，预测纯探索信息增益 $\hat{g}_i$，并反投影至 3D 空间得到位姿 $\mathbf{X}=\{\mathbf{p},\mathbf{q}\}$。
   - 使用 Set-of-Marks（SoM）在图像平面标注前沿质心标记，与自然语言目标 $\mathbf{L}$ 一起单次前向输入 VLM，获得语义相关性概率 $p_i = \text{VLM}(\mathbf{I}_t, \mathbf{F}_i, \mathbf{L}) \in [0,1]$。
   - 前沿效用融合探索与利用：$g_i = p_i \cdot \hat{g}_i$；全局排序结合机器人当前位置 $\mathbf{p}_r$ 计算 $u_i = \frac{g_i}{\|\mathbf{p}_r - \mathbf{p}_i\|}$。
3. **全局前沿管理与目标核验（§III-B，Algorithm 1–4）：**
   - 维护全局活跃前沿集合 $\mathcal{F}$ 与已清除集合 $\mathcal{C}$，通过 `MergeUpdate`（$\tau_{\text{merge}}=1.8\text{ m}$）与 `Prune`（$r_{\text{clear}}=0.5$）避免重复探索，并用 `HandleStall`（$\epsilon_{\text{stall}}=0.1\text{ m}, T_{\text{stall}}=15$）处理卡滞前沿。
   - 导航期间用开放词汇分割模型 SAM3 检测目标掩码；命中后结合深度图与相机位姿投影并做 DBSCAN 聚类得到目标质心 $\mathbf{x}_i$，在其前方 $r_{\text{sep}}$ 处插入高优先级的视角前沿（viewpoint frontier）。
   - 到达视角前沿（$d < r_{\text{near}}$）后，再次调用 VLM（`VLMDetect`）确认目标是否在视野内、5 m 以内且可达（$p > p_{\text{presence}}=0.7$）；确认则置 `hasTarget=1` 并逼近至 $r_{\text{goal}}=1.0\text{ m}$ 内终止。
   - 若无可见前沿，触发应急旋转机制（Appendix S1-B），插入左/右/后 3 个旋转前沿，必要时将 $g^{\min}$ 减半最多重试 $n_{\text{emergency}}=3$ 轮。
4. **实验结果（§IV，Table I–II，Appendix S1-D / S2）：**
   - 仿真（Habitat 0.2.4，Gemini-2.5-flash + FrontierNet + SAM3 + VLFM 预训练 DD-PPO PointNav，每 6 步检测与推理一次）：
     - HM3D Val：77.3% SR，35.6% SPL
     - MP3D Val：40.7% SR，17.8% SPL
     - OVON Val Unseen：39.0% SR，20.1% SPL
   - VLM 敏感性（HM3D Val，Table II）：替换为 LLaVA 1.6（74.2% SR / 35.3% SPL）、InternVL3_5-8B（74.5% SR / 34.5% SPL）、Gemma-3-4b-it（76.9% SR / 33.7% SPL），SR 波动 $\le 3\%$；同 VLM（LLaVA 1.6）下显著高于 UniGoal（54.5% SR）与 History-Aug. VLM（46.0% SR）。
   - 真机部署（Boston Dynamics Spot 机械臂末端 RGB 相机，~3 Hz）：结合 Metric3Dv2（`metric_depth_vit_large_800k`）单目深度估计、Wavemap 体素建图（0.1 m 体素）与 OMPL $RRT^\star$ 规划器（16 Hz 管理循环）；5 类目标共 10 次试验取得 70% 成功率。
5. **局限与失败模式（§V-B，Appendix S2-D）：** 主要失败来源于目标误检（False Positive，含数据集标注遗漏或识别偏差）、步数耗尽（Max Steps，未找到目标）以及底层 PointNav 陷入局部极小值卡死；系统在执行卡顿时缺乏快速语义级重规划/恢复机制。
6. **对 wiki 的映射：** [wiki/entities/paper-agentic-2603-05377.md](../../wiki/entities/paper-agentic-2603-05377.md)
