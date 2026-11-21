# DR-APO: Decoupled-Representation Advantage-Guided Policy Optimization（arXiv:2610.16618）

> 来源归档（paper / arXiv v1）

- **标题：** DR-APO: Decoupled-Representation Advantage-Guided Policy Optimization from Continuous Control to Embodied Navigation
- **arXiv：** <https://arxiv.org/abs/2610.16618>（v1，2026-10-12；PDF：<https://arxiv.org/pdf/2610.16618>）
- **全文：** <https://arxiv.org/html/2610.16618>
- **项目页：** <https://developer.deeprobotics.cn/research/articles/dr-apo/index.html>
- **作者：** Runhua Zhang, Yingjie Niu†*, Chao Li†
- **机构：** 云深处科技（DEEP Robotics）；浙江大学
- **入库日期：** 2026-11-21
- **核对范围：** 通读 `https://arxiv.org/html/2610.16618`（v1，2026-10-12）全文：Abstract、§I 引言、§II 相关工作、§III 方法（III-A 表示解耦策略优化、III-B 优势引导目标分布改进：Wasserstein-2 + KL 近端局部细化与指数重加权、III-C 表示兼容直接动作拟合）、§IV 实验（MuJoCo-v5 五任务、消融、InternScenes 仿真与 DEEP Robotics M20 真机 NavDP 后训练，Table I–V）、§V 结论。边界：未在本地复现实验；论文与项目页均未公开代码与权重。
- **通读：** 已完成
- **一句话说明：** 通过“表示解耦的优势引导目标改进（局部动作梯度细化 + 指数优势重加权）”与“表示兼容的直接动作拟合（行为锚定读取 + 原生去噪/流匹配辅助损失）”，为 Gaussian、Diffusion 和 Flow-Matching 策略提供统一 RL 优化与后训练接口，免除似然比回传与推理期多候选 critic 选择。

## 核心方法与实验要点（原文核对）

- **两阶段解耦推导（§III-A–III-B）：** 在局部策略改进代理目标下引入 $\beta D_{\mathrm{KL}}(q\|\pi_k)+\frac{1}{2\eta}W_2^2(q,p)$ 分布一致性正则，得到最优改进目标 $p_k^+$ 的封闭形式——**先移动动作位置** $a_i^+ = \Phi_{k,\eta}(s_i, a_i)$（用可微优势估计器 $A_\phi$ 的动作梯度在半径 $\rho$ 内做局部细化），**再按近端优势赋权** $\omega_i \propto \exp(A_{k,\eta}^+(s_i, a_i)/\beta)$。
- **表示兼容拟合（§III-C）：** 高斯策略直接对策略均值做加权回归；扩散/流策略从行为锚定潜状态读出并取多中间步平均做直接动作回归，同时叠加原生去噪或流匹配损失作辅助正则（Table III 显示 Full 优于单用 Native Only 或 Direct Only，且 `w/o Refinement` 与 `w/o Weighting` 均掉点）。
- **连续控制基准（MuJoCo-v5 1M 步、5 seeds，Table I）：** 仅用单次策略执行（不做推理期 Best-of-N critic 选择）时，Gaussian 版 DR-APO 平均 AUC@1M 排名 1.40（SAC 1.80、PPO 2.80）；Diffusion 版在 9/10 项（5 项 AUC@1M + 4 项 Final Return）领跑，avg rank 1.20/1.00；Flow 版在全部 10/10 项居首（avg rank 1.00/1.00，超越 FPO、Flow-SDE、ReinFlow、RFM）。
- **具身导航后训练（§IV-D，Table IV–V）：** 对预训练扩散导航策略 NavDP 做 DR-APO 后训练——InternScenes 轮式仿真整体 SR 由 70.33% 升至 **89.55%**（超 X-NavDP 的 88.63%），但 SPL 由 77.67 略降至 74.55（高成功率与路径长度间存在权衡）；在 **DEEP Robotics M20** 四足真机 20 个场景测试中，SR 由 25% 升至 **65%**。

## 对 wiki 的映射

- [DR-APO（arXiv:2610.16618）](../../wiki/entities/paper-dr-apo.md)
