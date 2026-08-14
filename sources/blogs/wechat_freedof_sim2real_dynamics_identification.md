# Sim2Real 动力学辨识：动力学模型、参数辨识与实验设计

> 来源归档（blog / 微信公众号）

- **标题：** Sim2Real 动力学辨识：动力学模型、参数辨识与实验设计
- **类型：** blog
- **作者：** 自由度FreeDof（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/B_sH9VNRxB6GCTJwnx6esQ
- **发表日期：** 2026-08-12
- **入库日期：** 2026-08-14
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox；`--no-images`）；直连一次成功。Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **原文归档：** [`sources/raw/wechat_freedof_sim2real_dynamics_identification_2026-08-12.md`](../raw/wechat_freedof_sim2real_dynamics_identification_2026-08-12.md)
- **一句话说明：** 单关节 PD 闭环下，惯量/延迟/摩擦/阻尼在同一条阶跃曲线上纠缠；先建模、再判断结构性不可辨识，再用分级实验把它们拆开，而不是把全部参数丢给优化器。
- **开源状态（步骤 2.5）：** 工程方法文，无项目页、无代码仓 → **步骤 2.5 不适用**（确认无开源代码）。

## 核心摘录（归纳，非全文）

### 问题不是「差一点」，而是「分不开」

整机一次失败只给一个标量（摔或不摔），却要反推几十个参数。单关节实验把未知量降一个数量级，把观测量从标量变成连续曲线。即便如此，惯量偏大与延迟未建模都让响应变晚；摩擦与阻尼都让曲线衰减更快——叠在同一条位置曲线上，光看贴合分不出谁是谁。

### 单关节 PD 闭环：两个数

无机械弹簧时刚度全由 $K_p$；总阻尼是 $K_d$ 与被动黏性 $b$ 之和。闭环自然频率 $\omega_n=\sqrt{K_p/J_{\mathrm{eff}}}$ 决定时间尺度，阻尼比 $\zeta$ 决定形状。按幅值与 $\omega_n$ 归一化后，线性阶跃只携带这两个数。不同幅值归一化曲线不重合 → 非线性（摩擦、死区、饱和）的证据。

未进二阶模型、但指纹不同因而可分开的项：

| 项 | 指纹 |
|----|------|
| 纯延迟 $\Delta$ | 时域整段平移；频域只改相位、不改幅值（合成等效量，不是某一段物理时间） |
| 摩擦 | 小阶跃占主导（$\tau_c$ 与幅值无关，驱动力矩正比于幅值） |
| 饱和 | 有阈值；软件扭矩与限幅后扭矩分岔后不能混拟合 |
| 链路增益 $k_t$ | 不改形状、只缩放力矩，与惯量完全等价；位置曲线给出的是 $J/k_t$ |

有效惯量 $J_{\mathrm{eff}}=J_{\mathrm{link}}+J_r G^2+\cdots$。高减速比时转子折算项常是主项；仿真里的 `armature` 是这个物理量，不是数值旋钮。

### 三条辨识原则

1. **实验约束的是输入–输出映射，不是参数。** 位置 PD 下 $(J,K_p,K_d)$ 同时乘 $\alpha$ 轨迹不变；必须引入扭矩通道才能打破尺度不变性。
2. **纠缠参数要换工况，不要换更强优化器。** 延迟↔惯量、库仑↔黏性、静摩擦↔低 $K_p$、控制器 $K_d$↔机械阻尼、柔性↔大阻尼、惯量耦合↔两独立关节，各有让它们分道扬镳的实验。
3. **按可分离程度排序，逐级固定。** 延迟（时间戳）→ 摩擦（准静态恒速）→ 惯量与总阻尼（中等幅值动态）→ 饱和与柔性（极端幅值/高频）。顺序反了，未建模误差会被后面的参数吸收。

### 实验流水线

前提：刚性固定上游、锁其他关节、正反方向、多档幅值/速度、统一时间戳、重复 ≥3 次。自由基座会把有效惯量退化成约化惯量，套固定基座模型会把「响应变快」误判为惯量偏小；用基座 IMU 判耦合，吊带只作保护。

| 顺序 | 实验 | 交付 |
|------|------|------|
| ① | 双向多幅值阶跃（含扭矩） | $\Delta$、饱和拐点，再 $J_{\mathrm{eff}}$、总阻尼 |
| ② | 低速往返 + 恒扭矩 | 库仑/黏性/静摩擦/死区（分方向）；半差给摩擦、半和给偏置 |
| ③ | 扫频 / Chirp | 传动刚度、共振、结构阻尼 |
| ④ | 自由衰减 | 被动阻尼 $b$、恢复刚度；扫多个 $K_d$ 外推到 $K_d=0$ |
| ⑤ | 交叉轴激励（并联） | 随姿态变化的有效惯量矩阵；`armature` 装不下非对角元 |

有可信扭矩时，选明显加速且未饱和窗对 $\tau-J\alpha$ 做线性回归，不必从阶跃曲线绕 $J$。只有位置时，多幅值截距法把 $\Delta$ 从「贴地二次曲线」里拆出来，再拟合整段归一化曲线，而不是只拟合超调/峰值时间两个点。

### 并联机构

两电机经并联机构驱动两自由度时，$J_{\mathrm{eff}}=J^\top I_m J$（合同变换），耦合项是结构常态。MuJoCo `armature` 只加对角，无法表达非对角；耦合显著就要显式并联模型，不能靠两个标量 `armature` 凑。

## 对 wiki 的映射

- 升格 [`wiki/methods/sim2real-joint-sysid-experiment-design.md`](../../wiki/methods/sim2real-joint-sysid-experiment-design.md)（实验设计 / 可辨识性；与算法页分工）。
- 交叉：[`wiki/methods/joint-actuator-parameter-identification.md`](../../wiki/methods/joint-actuator-parameter-identification.md)、[`wiki/concepts/system-identification.md`](../../wiki/concepts/system-identification.md)、[`wiki/concepts/armature-modeling.md`](../../wiki/concepts/armature-modeling.md)、[`wiki/concepts/joint-friction-models.md`](../../wiki/concepts/joint-friction-models.md)、[`wiki/concepts/robot-link-and-rotor-inertia.md`](../../wiki/concepts/robot-link-and-rotor-inertia.md)、[`wiki/concepts/humanoid-parallel-joint-kinematics.md`](../../wiki/concepts/humanoid-parallel-joint-kinematics.md)、[`wiki/queries/sim2real-closed-loop-engineering.md`](../../wiki/queries/sim2real-closed-loop-engineering.md)。
