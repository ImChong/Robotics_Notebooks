# Adaptive Whole-Body MPC with Optimal Posture Planning for Humanoid Robots under Unknown Payloads

> 来源归档（ingest）

- **类型：** paper / humanoid / model predictive control / payload
- **论文：** <https://arxiv.org/abs/2610.12972>（2026-10-09 提交）
- **作者：** Manh Cuong Nguyen、Duc Cuong Vu、Viet Anh Vu、Duc Tuan Pham、Toan Nguyen、Minh Hiep Phung、Ngo Anh Vien、Minh Nhat Vu
- **机构：** VinRobotics；VinUniversity 人工智能研究中心（CAIR）
- **状态：** arXiv 预印本
- **入库日期：** 2026-10-28
- **核查范围：** 本次可访问摘要，论文全文未能获取；实验条件、指标和源码状态待核实。

## 摘录 1：问题

未知或变化载荷影响人形机器人全身动力学，可能使搬运过程中的平衡控制退化。

**对 wiki 的映射：** [MPC 与 WBC 集成](../../wiki/concepts/mpc-wbc-integration.md)。

## 摘录 2：方法

论文摘要描述三部分：质心 MPC 生成动力学可行参考；自适应模糊平衡控制器针对载荷引入的模型失配生成补偿力矩并分配到支撑腿关节；基于三重积分器与航点的最优姿态规划器生成满足质心稳定、关节限制和全身可行性约束的起坐轨迹。

**对 wiki 的映射：** [论文实体](../../wiki/entities/paper-adaptive-whole-body-mpc-payload.md)、[质心动力学](../../wiki/concepts/centroidal-dynamics.md)。

## 摘录 3：评测边界

摘要称仿真与硬件实验相对非自适应基线改善平衡鲁棒性并减小跟踪退化；摘要没有给出具体平台、负载范围、数值和试验次数，本归档不推断这些细节。

**对 wiki 的映射：** [论文实体](../../wiki/entities/paper-adaptive-whole-body-mpc-payload.md)的评测与局限。
