# HumanoidNDM（arXiv:2610.34126）

- **论文：** [HumanoidNDM: Closing the Training-Deployment Gap of Motion Tracking via Humanoid Neural Dynamics Model](https://arxiv.org/abs/2610.34126)
- **版本：** [arXiv v1，2026-10-25](https://arxiv.org/html/2610.34126v1)，CoRL 2026
- **作者：** Chao Chen、Zhenrong Wang、Jilong Wang、Yusen Feng、Jiangran Lyu、Li Yi、He Wang
- **机构：** 北京大学计算机学院前沿计算研究中心、中国科学院自动化研究所、Galbot、北京智源人工智能研究院、清华大学
- **项目页：** [论文给出的地址](https://pku-epic.github.io/HumanoidNDM)；2026-10-30 访问返回 404，代码与数据开放状态未能核实。
- **核对范围：** 已核对摘要、方法、主要实验、结论和部分附录；全文及其余附录尚待完整复核。
- **入库日期：** 2026-10-30

## 摘录 1：探索（§4.1）
FENCE 在参考关节轨迹的低频分量上加噪，再重新计算速度；根据当前策略邻域数据覆盖率，用多臂老虎机在相邻扰动强度中自适应选择。原文表 6 的最高扰动频率为 1 Hz。

**对 wiki 的映射：** [项目实体](../../wiki/entities/paper-humanoidndm.md)的探索与流程图。

## 摘录 2：模型与策略（§4.2–4.3，附录 A）
历史本体状态与动作输入神经动力学模型；可微正向运动学把预测和观测转换到局部重力对齐坐标下的连杆位姿，以配置空间与欧氏空间双重损失训练。策略从真实数据缓冲区的历史片段初始化，在模型中自回归展开并反传跟踪损失。模型历史长度 10、集成头数 5，模型和策略展开长度分别为 8 与 128。

**对 wiki 的映射：** [项目实体](../../wiki/entities/paper-humanoidndm.md)的核心机制。

## 摘录 3：验证与限制（§5–6，附录 D、G）
仿真覆盖 5 个动作与质量、关节增益、摩擦三类差异；真机 G1 六项任务中套笔采用 50 条轨迹，其余 200 条，每项重复 3 次。论文指出方法限于时不变扰动；飞行阶段的高度估计需特殊处理，附录 G 是跳跃仿真实验。

**对 wiki 的映射：** [项目实体](../../wiki/entities/paper-humanoidndm.md)的评测与限制。
