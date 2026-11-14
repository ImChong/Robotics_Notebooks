# Topology-VLA（arXiv:2609.23944）

> 来源归档（paper）

- **标题：** Topology-Informed Visual Prompting For Vision Language Action Policies
- **作者：** Haoyang Wu、Abhinav Kumar、Dmitry Berenson
- **机构：** 密歇根大学（University of Michigan）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.23944>
- **PDF：** <https://arxiv.org/pdf/2609.23944>
- **项目页：** <https://topology-vla.github.io/>
- **代码：** 未开源（项目页仅提供 Paper PDF 与展示视频，无 GitHub 仓库或模型权重发布）
- **入库日期：** 2026-10-18
- **线索来源：** [机器人会写控制程序之后，能否自己造训练数据？](../blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)（具身智能小站，2026-10-18）
- **核对范围：** 全文通读 arXiv `2609.23944v1`（§I–§VII，含 Algorithm 1、Eq. 1–4、Tables I–III、Figs. 1–6 及 Appendix/References），并核对项目页 `topology_vla.html` 消融补充表
- **通读：** 已完成
- **一句话说明：** 通过虚拟环路拓扑不变量（virtual h-signature / Gauss Linking Integral）在仿真中自动收集绕障纠偏与续跑示范，并微调 VLM 按当前观测预测拓扑状态及 2D 路径箭头提示来引导下游 VLA（π0 / π0.5）。

## 开源状态

- **未开源（项目页未列代码）**（2026-11-14 核查）：arXiv 与项目页 `https://topology-vla.github.io/` 均未给出 GitHub 代码库、数据生成器或微调权重。

## 核心摘录

### 1) 拓扑状态抽象：Virtual h-Signature（§III–§V-A）

- **摘录要点：** 在有线缆、绳索、穿插框（如货架立柱与箱体把手）的任务中，从视觉表面看极其接近的两帧可能位于不同同伦/缠绕类别，后续要求朝相反方向绕行避障。本文以高斯环绕积分（Gauss Linking Integral）定义 **virtual h-signature** \(h^v = Lk(L_t, O)\)：把开放的机器人末端–物体路径通过虚拟直线段闭合成环 \(L_t\)，对预先标注的障碍骨架环 \(O\) 计算拓扑绕数（如 \(0\) vs \(1\)）。关键工程分界：**障碍环 \(O\) 仅在训练期仿真器内作为特权状态使用；部署时完全不依赖特权障碍几何。**

### 2) 仿真纠偏数据扩增 + 两步 VLM 视觉提示（§V-B & §V-C, Algorithm 1）

- **摘录要点：**
  1. **纠偏与续跑数据收集：** 先用 `D_nominal`（每任务 100 条示范）微调基础策略 `π_base`；在仿真中闭环执行并在末端注入高斯状态扰动 `PerturbState`，当真实特权 `h^v != h^{v*}` 时调用工作空间目标导向 RRT（`CorrectSignature`）规划最短无碰撞回到目标拓扑的轨迹 `D_corr`，随后由 `π_base`（失败时用末端 IK 控制器兜底）完成后续动作 `D_cont`。最终从**原始预训练 VLA 权重**出发在 `D_train = D_nominal ∪ D_corr ∪ D_cont` 上重新微调 `π_VLA`。
  2. **单模型两调用视觉提示：** 用同一个 `Qwen3-VL-8B-Instruct`（LoRA `r=8, α=16`）推理期按固定频率分两步调用：`f_hv(o_t) -> \hat{h}^v_t` 估计当前拓扑签名；再由 `f_way(o_t, l, \hat{h}^v_t, h^{v*}) -> W_t` 输出图像空间 2D 路径点序列并绘成绿色/青色箭头叠加到观测上（若拓扑已对齐或无须绕行则输出空序列保持原图），送入 π0 / π0.5（LoRA `r=16/32`, `H_a=10`, 20k 步）。

### 3) 仿真、消融与真机验证结果（§VI, Tables I–III）

- **摘录要点：**
  - **仿真成功率（Table I）：** Rope Pulling（50 trials，`h^v=0`）：π0 Base `70%` / PEEK `54%` / Ours `66%`；π0.5 Base `52%` / PEEK `50%` / Ours `64%`。Cable Threading（100 trials，`0→1`）：π0 Base `49%` / PEEK `13%` / Ours **`71%`**；π0.5 Base `47%` / PEEK `20%` / Ours **`77%`**。Box Pickup（100 trials，`1→0`）：π0 Base `31%` / PEEK `57%` / Ours **`84%`**；π0.5 Base `45%` / PEEK `66%` / Ours **`82%`**。
  - **VLM 拓扑预测准确率（Table II，100 unseen）：** Cable Threading `74%`（π0）/ `76%`（π0.5）；Box Pickup `92%`（二者同）。
  - **项目页补充消融（π0.5）：** 完整系统（64% / 77% / 82%）> 去掉视觉提示箭头仅靠纠偏数据训练（52% / 69% / 75%）> 仅保留 `D_nominal ∪ D_cont`（N/A / 24% / 54%）。
  - **真机 Box Pickup（Table III，30 trials on π0）：** 基于仿真模型加域随机化、仅 15 条真机到达示范 + 3 条纠偏轨迹（约为仿真帧数 3%）+ 1,000 张真机人工标定 `h^v` 图（约为 VLM 仿真数据 5%），整体任务成功率 Base `20%` / PEEK `3%` / Ours **`60%`**，拓扑纠偏成功率 Base `30%` / PEEK `20%` / Ours **`83%`**。PEEK 在真机暴跌是因为图像遮罩把货架立柱或绳环这类几何避障参考全遮掉了。

## 对 wiki 的映射

- [paper-topology-informed-visual-prompting-vla](../../wiki/entities/paper-topology-informed-visual-prompting-vla.md)
- [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-18-26-papers-technology-map.md)
