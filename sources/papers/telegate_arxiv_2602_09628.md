# TeleGate: Whole-Body Humanoid Teleoperation via Gated Expert Selection with Motion Prior（arXiv:2602.09628）

> 来源归档（ingest）

- **标题：** TeleGate: Whole-Body Humanoid Teleoperation via Gated Expert Selection with Motion Prior
- **类型：** paper / humanoid / teleoperation / whole-body tracking / gated experts / motion prior
- **arXiv abs：** <https://arxiv.org/abs/2602.09628>
- **arXiv HTML：** <https://arxiv.org/html/2602.09628>
- **PDF：** <https://arxiv.org/pdf/2602.09628>
- **项目页：** <https://anywitresearch.github.io/TeleGate/>
- **会议：** Robotics: Science and Systems（**RSS 2026**）
- **机构：** 中国科学技术大学（USTC）、芜湖哈特机器人技术研究院；通讯 Feng Wu（USTC）
- **硬件：** Unitree G1（29 DoF）；**惯性动捕** 在线重定向参考轨迹
- **仿真：** MuJoCo；策略 50 Hz，PD 500 Hz
- **入库日期：** 2026-07-03
- **一句话说明：** **冻结多域专家 + 轻量门控 Top-1 路由** 替代蒸馏统一策略；**VAE 非对称编解码**（历史 $M_t^-$ → 未来 $M_t^+$ 潜变量）补偿实时遥操作无未来参考；**2.5 h** 六类惯性动捕自采数据；仿真 SR **97.3%**，G1 真机跑跳/跌倒恢复/踢球等。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | [TeleGate](https://anywitresearch.github.io/TeleGate/) | 演示视频、数据集申请表 |
| 蒸馏对照 | TWIST [61]、Any2track [63] | 统一策略或 DAgger 蒸馏专家；论文 Table II 基线 |
| MoE 对照 | GMT [6] | 端到端联合训练 MoE；跟踪精度劣于门控选专家 |
| 大规模 tracking | SONIC [40] | 700 h 数据规模化；高训练成本 |
| 运动先验 | VMP [49] | VAE 轨迹段编码启发；TeleGate 用历史→未来非对称 VAE |
| 重定向 | Retargeting Matters [1] | 在线 skeleton retarget 至 G1 参考轨迹 |

> 截至入库时，**未见公开代码仓库**；惯性动捕数据集需邮件申请。后续若发布可补 `sources/repos/`。

## 摘要级要点

- **问题 1：** 实时全身遥操作 **无法访问未来参考** $M_t^+$，跳跃、起身等需 **预判** 的动作难跟踪。
- **问题 2：** 多类人体运动动态差异大；单策略多任务 **灾难性遗忘**；蒸馏多专家到单网络 **有损压缩**，高动态精度下降。
- **TeleGate 回答：** 按动态相似性将六类数据划为 **4 组专家**（走跑 / 舞武 / 跌倒恢复 / 跳跃），PPO 训练专家时 **联合优化 VAE 运动先验**；专家冻结后训 **门控网络 Top-1 选专家**；全身 **残差 PD** 跟踪。
- **数据：** **2.5 h** 惯性动捕（与部署设备一致）：走 40 min、跑 24 min、舞 24 min、武 20 min、跌倒恢复 26 min、跳 16 min；clip <10 s；**失败率课程采样**。
- **仿真（Table II，同数据集）：** SR 97.3% vs TWIST 68.9% / Any2track 91.2% / GMT 92.0%；$E_{mpjpe}$ 17.22 mm。
- **消融：** Top-1 门控（96.7%）接近 Oracle（96.3%），优于 DAgger 蒸馏（91.2%）；+VAE 再提升至 97.3%。
- **真机：** G1 跑、跳、跌倒起身、踢球、抓放、侧滑、单脚跳等；策略 50 Hz + 37.5 Hz 低通。

## 核心摘录（面向 wiki 编译）

### 1) 三阶段框架

| 阶段 | 内容 |
|------|------|
| I 采集 | 惯性动捕 → 在线 retarget → 记录基座位姿 + 关节参考 |
| II 专家 | 4 专家 × PPO + **联合 VAE**（encoder $M_t^-$，decoder 重建 $M_t^+$，$d=32$） |
| III 门控 | 冻结专家；$G_\theta(o_t,m_t)\rightarrow\mathbb{R}^K$；**argmax Top-1** 执行 $\pi_{i^*}$ |

### 2) 观测与动作

- $s_t=\{o_t,m_t,z_t\}$；$o_t$ 含重力投影、基座速度、$q,\dot q$、$q_{t-1}^{target}$。
- $M_t^-=\{m_{t-20},m_{t-10},m_{t-5},m_{t-1},m_t\}$ 非均匀稀疏历史窗。
- $a_t$ 为全身关节 **残差**：$q_t^{target}=\hat q_t + a_t\cdot\alpha$ → PD 力矩。

### 3) 专家划分（IV-C1）

| 专家组 | 包含运动类型 |
|--------|----------------|
| 1 | walking + running |
| 2 | dancing + martial arts |
| 3 | fall + recovery |
| 4 | jumping |

### 4) 与仓库内路线的关系

| 维度 | TeleGate | TWIST2 / TWIST | SONIC | BFM 掩码蒸馏 | PILOT MoE |
|------|----------|----------------|-------|--------------|-----------|
| 任务 | **实时全身遥操作 LLC** | 便携采集 + tracking | 大规模 tracking | 单策略多接口 | 感知 loco-manip LLC |
| 多技能 | **门控选冻结专家** | 统一 tracking 策略 | 数据规模化 | 在线蒸馏 | 联合训练 MoE |
| 数据 | **2.5 h 惯性动捕** | 42 h 光学 / 便携流 | 700 h | 大规模 MoCap | 无 MoCap 课程 |
| 未来信息 | **VAE 从历史推断** | 离线 tracking 可用未来 | — | CVAE 接口 | 高程图前瞻 |

## 对 wiki 的映射

- 沉淀实体页：[TeleGate（论文实体）](../../wiki/entities/paper-telegate.md)
- 交叉补强：[Teleoperation](../../wiki/tasks/teleoperation.md)、[Motion Retargeting](../../wiki/concepts/motion-retargeting.md)、[Whole-Body Control](../../wiki/concepts/whole-body-control.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)、[TWIST2](../../wiki/entities/paper-twist2.md)、[SONIC 方法页](../../wiki/methods/sonic-motion-tracking.md)

## 当前提炼状态

- [x] 摘要、三阶段方法与实验要点摘录
- [x] wiki 实体页与任务页交叉链接规划
- [ ] 待公开代码 / 数据集下载页后补 `sources/repos/`
