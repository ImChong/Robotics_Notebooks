# FALCON：Learning Force-Adaptive Humanoid Loco-Manipulation

> 来源归档（ingest · 人形 Loco-Manip 161 篇长文 第 109/161 · 原文通读补全）

- **标题：** FALCON: Learning Force-Adaptive Humanoid Loco-Manipulation
- **机构：** Carnegie Mellon University、Field AI、Nissan USA
- **类型：** paper
- **发表：** L4DC 2026 Oral / arXiv:2505.06776v2（2025-11-16）
- **Loco-Manip 161 分类：** 05 动捕、人类视频与交互动作规划
- **原始链接：**
  - <https://arxiv.org/abs/2505.06776>
  - 项目页：<https://lecar-lab.github.io/falcon-humanoid>
  - 官方代码：<https://github.com/LeCAR-Lab/FALCON>
- **入库日期：** 2026-06-26（原文通读补全：2026-11-14）
- **核对范围：** 通读 `arXiv:2505.06776v2` PDF 全文抽取文本（第 1–1247 行）：覆盖 Abstract、§1–§2 引言与相关工作、§3 双代理学习框架与力矩约束感知 3D 外力课程（Eq. 1–5）、§4 仿真与真机实验（Table 1–4 跟踪误差、Fig. 3 课程消融、§4.5–4.6 遥操作与自主物流部署）、§5–§6 结论与局限，以及 Appendix A.1–A.6（奖励表 Table 5、域随机化 Table 6、外力估计对比、500N 拉力计实测峰值、FoundationPose + MoCap 物流系统与 MuJoCo vs 真机热过载极限分析）。未逐像素复核训练曲线图。
- **通读：** 已完成
- **一句话说明：** 面向载荷搬运、拉车与开门等持续 3D 末端受力任务，FALCON 将全身控制解耦为共享 5 步本体历史的「下肢稳定 Agent + 上肢末端跟踪与隐式力补偿 Agent」，结合尊重关节力矩极限的 3D 外力课程联合训练，并在无需按本体重调的前提下部署至 Unitree G1 与 Booster T1。

## 核心摘录

### 1) 原有策展索引定位
- **在 161 篇地图中的位置：** 05 动捕、人类视频与交互动作规划，编号 **109/161**；原始汇总归档见 [wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md](../blogs/wechat_embodied_ai_lab_humanoid_loco_manip_161_survey.md) 与 [wechat_humanoid_loco_manip_161_2026-06-26.md](../raw/wechat_humanoid_loco_manip_161_2026-06-26.md)。

### 2) 双代理分解与力矩边界感知 3D 外力课程（§3.1–§3.2）
- **双代理架构：** 下肢策略 \(\pi^l(s_t^p, G_t^l)\to a_t^l\)（目标为根速度、步态相位、根高与腰部偏航角）与上肢策略 \(\pi^u(s_t^p, G_t^u)\to a_t^u\)（目标为上肢期望关节角 \(q_t^{upper*}\)，训练期从 AMASS 随机采样，部署期通过 IK 计算）共享 5 步全身本体历史 \(s_t^p\)，但各自优化专属奖励与非对称 Critic（特权读取根线速度与末端外力 \(F_t^{ee}\)）。
- **力矩边界感知 3D 外力课程：** 依据重力补偿力矩 \(\tau_g\)、末端 Jacobian \(J_{EE}\) 与关节力矩上限 \(\tau^{lim}\)，按 \(-\tau^{lim} \le \tau_g + J_{EE}^T f^{ee} \le \tau^{lim}\) 计算各轴可行力界 \([f_i^{min}, f_i^{max}]\)；再经 Dirichlet 分布分配三轴权重、乘以全局进度系数 \(\alpha_g\in(0,1)\) 施力，并行走期将平面外力反向投影至速度方向、随机沿末端远端连杆变换施力点，实现无需显式力传感器的隐式力补偿。

### 3) 关键定量评测与硬件局限（§4, Table 1–4, Appendix A.4–A.6）
- **IsaacGym 252 个 ACCAD 目标评估（Table 1）：** 在大外力（`L-Force`, \(\alpha_g=1.0\)）下，`FALCON` 上肢追踪误差 `E_tracking^{upper} = 0.37±0.04`，显著优于 `PID-Force-Curr.`（`0.60±0.04`）、`PD-ID-Force-Curr.`（`0.53±0.04`）与单体全身 RL `M-WB-RL-with-Force-Curr.`（`0.73±0.05`），下肢根速度误差保持 `0.45±0.04`；若去掉力矩边界约束直接大范围截断采样 `[-100, 100] N`（Table 2 & Fig. 3），课程进度因频繁超力矩停滞在 \(\alpha_g=0.6\) 且 `L-Force` 上肢误差恶化至 `0.61±0.10`（对比完整版 `0.36±0.04`）。
- **真机双臂负重定量追踪（Table 4，Unitree G1 每手挂重 `1.2 kg` 以 `0.5 m/s` 行走）：** `FALCON` 取得 `E_tracking^{upper} = 0.39±0.08`、`E_tracking^{root} = 0.42±0.03`，明显低于 `Upper-PD-Force-Curr.`（`1.81±0.13`）和 `M-WB-RL-Force-Curr.`（`0.81±0.11`）。
- **真机持续交互与力表实测峰值（§4.5 & Fig. 8）：** 搬运载荷 `0–20 N`、拉车 `0–100 N`（500N 手持拉力计实测峰值 `107.9 N`）、开门 `0–40 N`（峰值 `47.3 N`）；静态拉扯抗衡峰值 Unitree G1 达 `57.4 N`、质心更低的 Booster T1 达 `66.3 N`；配合 FoundationPose 6D 位姿与 MoCap 全局定位在仓库料箱物流任务中完成自主取放（§A.5）。
- **真机电机热极限（§A.6）：** 在默认伸臂姿势下，MuJoCo 仅做力矩截断时每臂可带 `>3 kg` 行走（`-1 m/s`），但真机 Unitree G1 手腕电机因持续发热很快过热，限制长时稳态持重不超过约 `2 kg`/臂（短时峰值拉力如拉车不受此限制）。

## 对 wiki 的映射

- [paper-loco-manip-161-109-falcon](../../wiki/entities/paper-loco-manip-161-109-falcon.md)
