# BRACE 论文通读（地形与力感知的人形全身跟踪）

> 来源归档（原文通读 · 支撑 `wiki/overview/frontier-humanoid-quadruped-2026-10-05-10-09.md`）

- **标题：** BRACE: Adapting Whole-Body References for Force and Terrain Aware Humanoid Motion Tracking
- **作者：** Sudarshan Harithas, Chen Yu, Juan Borbon, Shubhankar Mondal, Winston Zha, Srinath Sridhar, Dingqi Zhang, Jiuguang Wang（Multiply Labs、布朗大学、新墨西哥大学）
- **类型：** paper / humanoid / whole-body-tracking
- **原文：** <https://arxiv.org/abs/2610.07052>（arXiv PDF v1，2026-10-05 版，2026-11 获取全文）
- **一句话说明：** 不改控制器、改参考：把平地参考按地形与指令手部力矩解析地改写，再把“施力”与“抗力”两个特权教师经 DAgger 蒸馏成只用本体感知的流匹配学生。
- **核对范围：** 通读 §I–§V 全文（引言、相关工作、§III-A 参考合成、§III-B 教师、§III-C 学生蒸馏、§III-D 实现细节、§IV 全部实验与图 3–5、结论）及参考文献列表；该版本无独立附录与“局限”小节，故局限处只能依据文中自述的评测边界。项目页与代码未核对（本归档不涉及开源状态）。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题（§I）：** 全身跟踪的参考多在平地、无负载下录制，缺少力与地形信息；力可控策略不给全身位姿，地形自适应跟踪器又把负载当扰动抑制。
- **思路（§III 开头）：** 在策略学习前先用解析变换改写参考，避免用辅助奖励去“发现”修正而与跟踪奖励冲突。
- **地形调整 T（§III-A）：** 保持足端水平位置，抬高到 z=H(x,y)+z_ref；根高度跟随基座下地形；脚朝向按地面法向旋转；全身 IK 与上身对齐。
- **力矩调整 F（§III-A，式 2–4）：** ①可行限幅：用雅可比把手部力/力矩映射到关节力矩，约束 |(JpᵀF+JrᵀM)_i| ≤ γ·τ_budget；②手部前导：弹簧阻尼顺应模型，由 k_F⁻¹ = n̂ᵀJpKq⁻¹Jpᵀn̂ 求出位移；③质心支撑（COM brace）：Δc = ẑ×Σ[(p_h−c_cop)×(−F)−M]/(mg)，其中一部分由压力中心在支撑多边形内移动吸收，余量作为躯干倾斜。
- **教师（§III-B）：** Teacher-E（施力，虚拟力矩场提供反作用）、Teacher-C（抗力，外部力矩下跟踪 x_ref^T；负载受残余力矩预算式 6 约束，含课程）；均用 PPO + 非对称 critic，输出 29 个关节位置目标，并有重建 x* 的辅助头。
- **学生（§III-C）：** DAgger 蒸馏，模式标志 μ∈{E,C} 选教师；流匹配基座以 25 帧本体感知历史建模多模态动作分布，任务编码器接平地目标、模式与力矩指令；辅助估计头回归地形与外力，部署时丢弃。
- **实现（§III-D）：** Unitree G1 29 自由度，50 Hz；训练用 IsaacLab + ProtoMotions，8 张 RTX Pro 6000；参考数据取自 Bones-Seed 中按文本筛出的约 6.5 万条移动与操作轨迹；抗力教师训到 20 N、施力教师到 9 N，力矩到 4 N·m。
- **评测（§IV）：** 基线 SONIC、TWIST2、Any2Track、HEFT；测试集 PHUMA 与 Bones-Seed 留出集；坡度名义 4.6–17.7°；成功判据为躯干/腕/踝高度偏差<0.25 m、躯干姿态误差<1 rad、位置漂移条件。最高负载下全局误差平均降低约 40%，成功率平均提升 18.25 个百分点（抗力平地 +10.7、斜坡 +22.9；施力平地 +5.7、斜坡 +33.7）。力矩误差在 0.5–2.5 N·m 指令下比最近基线低约 33%（平地）/32%（斜坡）。
- **应用（§IV-E）：** 362 条遥操作香蕉搬运示范微调 GR00T N1.7 3B；约 1000 英里跨站点遥操作；PICO VR 遥操作；手柄驱动 MotionBricks。
- **文中自述边界（§IV-C）：** 与“直接指令末端力”的控制器（UNIFP、FACET）的相对力精度“尚未测试”，这些控制器因接口不同未纳入对比。
