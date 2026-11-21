# ALERT：用轻量人类避障管线扩展人形动作跟踪器（arXiv:2611.22258）

> 来源归档（ingest）；提炼论文原文，独立结构化知识见 wiki 实体页。

- **英文标题：** ALERT: Extending Humanoid Motion Trackers with a General and Lightweight Human Avoidance Pipeline
- **中文标题：** ALERT：用通用且轻量的人体避障管线扩展人形动作跟踪器
- **类型：** paper / humanoid motion tracking / human avoidance / safe interaction / policy adaptation
- **论文：** <https://arxiv.org/abs/2611.22258>
- **HTML 全文：** <https://arxiv.org/html/2611.22258>
- **PDF：** <https://arxiv.org/pdf/2611.22258>
- **版本：** arXiv:2611.22258v1（2026-11-17，cs.RO，预印本，10 页）
- **作者：** Zhaoyang Li、Chiyu Zhang、Chengxin Xu、Peng Yang、Zhonghao Guo、Wanting Li、Ziyu Chen（前三位同等贡献；后两位通讯作者）
- **机构：** 中国科学院自动化研究所（CASIA）；中国科学院大学（UCAS）；北京理工大学（BIT）
- **入库日期：** 2026-11-21
- **项目及代码核查：** arXiv 摘要页与 v1 HTML/PDF 未提供可确认的 ALERT 官方项目主页、训练/推理源码或权重地址；2026-11-21 检索也未验证到作者发布的官方仓库。故按「**官方实现未核实开放**」记录，不臆造项目页、源码运行入口或复现命令。
- **核对范围：** 通读 arXiv HTML v1 §1 引言、§2 相关工作、§3.1–3.4 算法及式 (1)–(12)、§4 实验（Table 1–3、Fig. 1–8，含仿真、G1 真机、对照及消融）、§5 结论与局限；对照 PDF 第 4 页 Figure 2 方法总览。
- **通读：** 已完成

## 原文要点与 wiki 映射

### 1. 问题定义与基座抽象（§1、§3.1）

- 既有 SONIC、TWIST2、BFM-Zero 全身动作跟踪策略主要依赖本体状态和参考运动，不感知靠近的人；挥臂、踢腿及遥操作存在碰撞风险。
- 基座表示为 $\xi_t=E(s_t,\mathcal M_t)$、$a_t^0=D_\omega(s_t,\xi_t)$；ALERT **保留各基座原生 reference-conditioning**，只在其 action network 加入人体势场条件化的低秩修正。
- 冻结 $W_\ell,b_\ell$，训练 $A_\ell,B_\ell,G_\ell$；层输出 $y_\ell=[W_\ell+B_\ell\,\mathrm{diag}(g_\ell)A_\ell]h_\ell+b_\ell$，$g_\ell=\sigma(G_\ell O_U+b_\ell^g)$；$B_\ell$ 零初始化。
- **映射：** [ALERT 独立论文实体](../../wiki/entities/paper-alert-human-avoidance.md)、[Whole-Body Tracking Pipeline](../../wiki/concepts/whole-body-tracking-pipeline.md)。

### 2. LiDAR 人体包络（§3.2）

- 全向 LiDAR 点云 → **Person-MinkUNet** 人体检测 → 粒子滤波点云配准 → $\alpha$–$\beta$ 位置速度滤波，获取连续的人体中心轨迹。
- 在线凸包络 $\mathcal K_t=b_t+S_t\mathcal K_0$，三轴缩放 $S_t$ 适配人体体型、姿态与局部点云，满足点云覆盖比例 $\ge 1-\varepsilon$。
- 从机器人采样点到人体包络边界求距离 $d$ 和梯度，避免直接对噪声点云计算不连续斥力。
- **映射：** [ALERT](../../wiki/entities/paper-alert-human-avoidance.md)。

### 3. 势场 → 目标速度流 → 条件 LoRA 策略（§3.3–3.4）

- 近人排斥势能采用平滑函数 $U=f_{\max}\delta\log[1+\exp((d_0-d)/\delta)]$；多人体势能**叠加**为固定大小的逐身体采样点特征。
- 在第 $i$ 个采样点，沿势能下降方向 $e_i$ 增加速度 $\sqrt{2U_i/m_i}e_i$，与参考速度 $v_i^0$ 叠加得到目标 $v_i^\star$；势能降低后回到参考流，尽量保留等势面切向运动。
- $O_U=\{U_i,e_i,v_i^d,v_i^t\}_{i=1}^m$ 作为 LoRA gate 条件；PPO 训练中用基座 tracking reward、$-\sum_i w_i\|v_{i,t+1}-v_i^\star\|^2$ 流跟踪奖励以及人体碰撞时 $-10$ 惩罚；更新 LoRA 与 value network。
- **映射：** [ALERT](../../wiki/entities/paper-alert-human-avoidance.md)、[人形 Motion Tracking 选型](../../wiki/queries/humanoid-motion-tracking-method-selection.md)。

## 实验与数字（§4，按条件解读）

| 实验 | 论文报告 | 边界 |
|---|---|---|
| 微调配置 | SONIC、TWIST2、BFM-Zero；BONES-SEED 选取 500 条易伸入周边空间的动作；单张 RTX 4090，在 Isaac Sim / Isaac Gym 中微调 | 并非从零训练三个基座；每次微调使用一张 4090 |
| 避障观测与适配 | LoRA rank=16；21 个身体采样点（骨盆 1，双臂与双腿各合计 10） | 保留各模型原生 motion condition |
| Table 1 无人场景 | SONIC 原始全局位置误差 52.32±4.92 mm，适配后 54.70±4.80 mm | 仅「接近」原跟踪精度，并非完全无损 |
| Table 1 有人场景 | SONIC / TWIST2 / BFM-Zero 归一化安全增益 M 分别 1.21±0.14 / 2.18±0.38 / 2.16±0.86 | M 是相对原穿透深度的归一化净空改善量，不是成功率 |
| Table 3 SONIC 对照 | 100 次试验的测试设置中 SR：原 SONIC 0、ConstrainedMimic 83、CSF 95、ALERT **100**（%） | 论文指定的人体进入参考动作扫掠区测试，不能推出普适零碰撞 |
| G1 真机（§4.2） | Ninja Dance、Disco Dance、Kick、Teleop，每类 20 次试验 SR **100%**；测得最小离 SMPL 表面净空 **3 cm** | 限于本文实验条件与观察人数，不构成工业安全认证 |
| LiDAR 感知 | 约 30% 遮挡与人速 ≤2 m/s 下，人体中心误差均值 8 cm、包络表面覆盖 93%，避障响应估计延迟 100 ms | 高速、更强遮挡与传感器异常工况仍需独立验证 |
| 消融 Table 3 | 不用条件 LoRA / 不用势场 / 不用速度流，SR 分别 51/60/79（%），完整方法 100（%） | 对同一 SONIC 设置的消融，不可跨任务直接外推 |
| 未见动作验证 | 2048 条不参与微调的动作；条件 LoRA 相比全参数微调多数跟踪误差比 <1 | 体现跟踪保留趋势，不能当成任意新动作零失效保证 |

## 对照与局限

- **CSF：** 在安全风险下替换后续参考运动，可能截断原动作内容。
- **ConstrainedMimic：** 对参考/底层输出作当前步约束修正，可能忽略随后接触转换与平衡的动态影响。
- **ALERT：** 将避障行为通过策略适配纳入长期协调，不改动各 tracker 原生参考编码；不是严格约束下的形式化安全证明。
- **作者明示局限：** 凸人体包络不能表达双臂等肢体之间的空隙，近距离避让可能保守，未来拟采用关节化人体几何。
- **工程边界：** 无公开官方可运行代码时，不提供未经核实的源码运行时序图或安装命令；真机复现仍需安全隔离、急停与传感器故障处理。

## 参考来源

- [ALERT 论文 HTML（arXiv:2611.22258v1）](https://arxiv.org/html/2611.22258)
- [ALERT 论文 PDF](https://arxiv.org/pdf/2611.22258)
