# QuietWalk: Physics-Informed Reinforcement Learning for Ground Reaction Force-Aware Humanoid Locomotion Under Diverse Footwear（arXiv:2604.23702）

> 来源归档（ingest）

- **标题：** QuietWalk: Physics-Informed Reinforcement Learning for Ground Reaction Force-Aware Humanoid Locomotion Under Diverse Footwear
- **类型：** paper / humanoid locomotion / physics-informed RL / GRF estimation / low-noise walking / cross-footwear
- **arXiv abs：** <https://arxiv.org/abs/2604.23702>
- **arXiv HTML：** <https://arxiv.org/html/2604.23702v1>
- **PDF：** <https://arxiv.org/pdf/2604.23702>
- **机构：** 中科院宁波材料技术与工程研究所（NIMTE, CAS）、中国科学院大学（UCAS）、浙江省精密驱动与智能机器人重点实验室、西湖大学工学院（Westlake University）
- **硬件：** Unitree G1；足底 **八单元植埋式力传感鞋垫**（训练 GRF 预测器）；机载 **仅本体感知** 部署
- **仿真：** NVIDIA Isaac Sim；500 Hz 物理 / 50 Hz 控制；PPO actor–critic
- **入库日期：** 2026-07-07
- **一句话说明：** 用 **逆动力学约束 PINN** 从 6 帧本体感知估计双足 **竖直 GRF**，冻结后作为 RL **冲击惩罚奖励**（无需部署力传感器）；**分阶段课程**（$\alpha$ 渐增 + 多鞋型随机切换）在 G1 真机 **1.2 m/s** 下相对基线 RL 平均降噪 **7.17 dB（MNL）/ 4.98 dB（PNL）**，并验证 **赤脚 / 滑板鞋 / 运动鞋 / 高跟鞋** 与多地面材质交叉泛化。

## 摘要级要点

- **问题：** 人形机器人在家庭/医院等 **人机共处环境** 需抑制足地冲击引起的 **振动与噪声**；现有低噪行走多依赖 **足端速度运动学代理** 或 **脆弱力传感器**；**鞋型改变接触动力学** 带来分布偏移，缺乏系统跨鞋鲁棒验证。
- **GRF 预测器（PINN）：** 输入 6 帧（0.12 s）$[q;\dot{q};\ddot{q}]$（$n=29$）；**DynamicsKAN** 学 $M(q)$（对称正定因子化）、**PotentialKAN** 学 $V(q)$ 得 $G(q)=\nabla V$、由 $M$ 算 $C(q,\dot{q})$；**Sequence-Net** 学有效广义力矩 $\tau_{\mathrm{eff}}$；经阻尼最小二乘伪逆从逆动力学残差重构 $f_z$，再 $\max(\cdot,0)$ 投影；损失 $\mathcal{L}=\lambda_{\mathrm{grf}}\mathcal{L}_{\mathrm{grf}}+\lambda_{\mathrm{dyn}}\mathcal{L}_{\mathrm{dyn}}+\lambda_{\mathrm{swing}}\mathcal{L}_{\mathrm{swing}}+\lambda_{\mathrm{smooth}}\mathcal{L}_{\mathrm{smooth}}$。
- **消融（held-out 真机数据）：** 去掉 $\mathcal{L}_{\mathrm{dyn}}$ 使左右足 RMSE 从 **14.5/14.0 N** 升至 **106.3/79.4 N**，$R^2$ 从 **0.99/0.99** 跌至 **0.39/0.67**；动力学一致性是可靠力反馈与奖励设计的关键。
- **低冲击 RL：** 总奖励 $r_t=r_{\mathrm{task}}+r_{\mathrm{bonus}}+r_{\mathrm{impact}}$，其中 $r_{\mathrm{impact}}=-\alpha\big((f_z^{(L)})^2+(f_z^{(R)})^2\big)$；**冻结** GRF 预测器仅用于奖励；关节 **位置增量** $\Delta q$ + 固定 PD；课程：先小 $\alpha$ 学稳态走，再增大 $\alpha$ 并引入地形随机化与 **多鞋型联合采样** 得单一策略。
- **鞋模管线：** Blender 将鞋 mesh 与 G1 足端 **布尔并集** 为「足+鞋」统一几何，同步 URDF/MJCF/USD 视觉与碰撞，按统一密度重算惯量；三类鞋：**滑板鞋、运动鞋、高跟鞋**。
- **声学评估：** 腿侧约 15 cm 处电影级无线麦，48 kHz；报告 **A 加权 SPL** 的 **MNL（均值）** 与 **PNL（峰值）**；四种地面：混凝土、地毯、木地板、瑜伽垫。
- **主结果（赤脚，四地面均值，1.2 m/s）：** D2（QuietWalk）相对 D1（基线 RL）MNL **89.47→82.30 dBA（−7.17 dB）**，PNL **104.28→99.30 dBA（−4.98 dB）**；相对宇树内置控制器 D3，MNL/PNL 差距约 **4.17/2.34 dB**（地毯/木/瑜伽垫 PNL 差缩至 0.47–1.86 dB）。
- **跨鞋×地面：** 缓冲好、接触面积大的鞋型（滑板鞋/运动鞋）噪声更低；**高跟鞋×木地板** 与 **运动鞋×瑜伽垫** 组合差近 **20 dB**；户外草地/碎石/鹅卵石/石板/沥青上多鞋型仍稳定行走。

## 核心摘录（面向 wiki 编译）

### 与运动学代理低噪路线的对比

| 维度 | QuietWalk（本文） | 足端速度惩罚（Aibo / Olaf 等） | QuietPaw |
|------|-------------------|-------------------------------|----------|
| 冲击信号 | **PINN 预测竖直 GRF** | 接触速度/加速度代理 | 可调「安静度」约束 |
| 物理一致性 | **逆动力学残差约束** | 无显式力模型 | 未强调力估计 |
| 部署传感 | **仅本体感知** | 仅本体感知 | 环境相关 |
| 额外维度 | **跨鞋型接触变化** | 未系统验证 | 未强调鞋型 |

### 与 MPC-RL / 特权力反馈路线的关系

- [MPC-RL（arXiv:2606.05687）](../../wiki/entities/paper-mpc-rl-humanoid-locomotion-manipulation.md) 在 **critic 特权信息** 中使用仿真 **真实 GRF** 做 `mpc_grf` 软对齐；QuietWalk 在 **无部署力传感器** 前提下，用 **预训练 PINN** 提供 **训练–部署一致的力反馈代理**。
- 与 [OpenCap Monocular](../../wiki/entities/paper-opencap-monocular.md) 的 GRF 估计目标不同：后者服务 **人体临床动力学**；QuietWalk 服务 **人形 RL 奖励塑形与室内低噪行走**。

## 对 wiki 的映射

- 沉淀实体页：[QuietWalk 物理感知低噪人形行走（arXiv:2604.23702）](../../wiki/entities/paper-quietwalk-humanoid-locomotion.md)
- 交叉补强：[Humanoid Locomotion](../../wiki/tasks/humanoid-locomotion.md)、[Locomotion](../../wiki/tasks/locomotion.md)、[Contact Dynamics](../../wiki/concepts/contact-dynamics.md)、[Locomotion 奖励设计指南](../../wiki/queries/locomotion-reward-design-guide.md)、[Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)、[MPC-RL](../../wiki/entities/paper-mpc-rl-humanoid-locomotion-manipulation.md)

## 当前提炼状态

- [x] 摘要、PINN 结构、RL 奖励、鞋模管线、声学/GRF 实验要点摘录
- [x] wiki 实体页与任务/奖励指南交叉链接规划
- [ ] 待作者公开代码后补 `sources/repos/`
