# CompliantWBC（arXiv:2609.33310）

> 来源归档（ingest）

- **标题：** CompliantWBC: Whole-Body Compliance for Heavy Humanoids via Force Latent Estimation and Residual Impedance Targets
- **arXiv：** <https://arxiv.org/abs/2609.33310> · <https://arxiv.org/pdf/2609.33310>
- **项目页：** <https://dotandung.github.io/compliantwbc/>
- **作者/单位：** Tan-Dzung Do 等；VinRobotics、National University of Singapore、VinUniversity、TU Darmstadt
- **入库日期：** 2026-10-05
- **摘要：** 冻结基础策略上叠加外力 latent 与阻抗目标残差；论文报告仿真和约 70 kg 人形真机任务。
- **详情：** [CompliantWBC](../../wiki/entities/paper-compliantwbc-heavy-humanoid.md)

## 核心摘录

（2026-10-05 核对 arXiv HTML 版）

- 两阶段 RL：Stage 1 联合训练力编码器与基础策略（compliance-fidelity reward 来自多点全身阻抗参考控制器）；Stage 2 在冻结基础策略上训练只修改阻抗平衡点的有界残差。
- Table II（100 组 paired rollouts）：Ours (full) E_imp = 2.58、S = 0.98、R_LB = 0.31；TWIST2 S = 0.59、E_imp = 6.79；GentleHumanoid S = 0.91、R_LB = 0.14。
- Table III：估计 wrench 下残差使 E_imp 3.81 → 2.58 cm，回收 oracle（2.44 cm）差距约 90%。
- 真机（约 70 kg 自研人形）：静态/动态受力响应、协作搬运（100 N 横杆）、擦板、负重下蹲，仅定性报告。
