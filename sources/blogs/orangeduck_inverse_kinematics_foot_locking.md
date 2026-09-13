# Inverse Kinematics and Foot Locking

> 来源归档（blog / theorangeduck.com）

- **标题：** Inverse Kinematics and Foot Locking
- **类型：** blog
- **作者：** Andrew McDonald（The Orange Duck）
- **原始链接：** https://theorangeduck.com/page/inverse-kinematics-foot-locking
- **发表日期：** 2026-07-30
- **入库日期：** 2026-09-13
- **配套代码：** https://github.com/orangeduck/GenoView-InverseKinematics — 归档见 [`sources/repos/genoview-inverse-kinematics.md`](../repos/genoview-inverse-kinematics.md)
- **项目页：** https://theorangeduck.com/page/inverse-kinematics-foot-locking — 归档见 [`sources/sites/theorangeduck-ik-foot-locking.md`](../sites/theorangeduck-ik-foot-locking.md)
- **一句话说明：** 动画/游戏管线中消除脚滑的五段「配方」：两骨 IK 腿链求解 → 惯性化运行时足锁 → 趾速/高度启发式标注接触 → 离线 PBD 式约束迭代 → 哲学：脚滑本质是速度误差、锁趾不锁跟、IK 是姿态修正而非替换。

## 核心摘录（归纳，非全文）

### 1. 腿链 IK（SolveLegChain）

目标：在尽量保留输入姿态的前提下，把 **趾（toe）** 放到目标位置。

1. 由趾目标反推 **跟（heel）** 目标：保持输入 pose 中 heel–toe 向量。
2. **两骨 IK**（hip–knee–heel）：余弦定理求髋/膝旋转；`maxExtension` + 指数 soft clamp 防超伸；用 knee **side vector** 定旋转轴，免 pole vector。
3. **Heel look-at**：`QuaternionBetween` 让 heel→toe 指向趾目标。
4. **可选 toe-end look-at + 地面高度 clamp**：bind pose 最小高度约束，防穿地。

### 2. 运行时足锁（Foot Locking + Inertialization）

- 无接触：趾目标跟随输入动画。
- 有接触：锁定接触开始时地板上的静态位置。
- **三次惯性化（cubic inertialization）** 在 lock/unlock 间平滑切换；`lockDistance` / `unlockDistance` 迟滞防抖。
- 每帧：`UpdateFootLockingState` → `SolveLegChain`。

### 3. 接触时间标注（Contact Times）

- **金标准**：人工标注。
- **启发式（约 90%）**：趾关节 **全局速度模长** 阈值（典型 0.1–0.5 m/s @60Hz）+ 趾高 sanity check（~0.1 m）。
- 后处理：**majority vote / median filter**（5 帧 @60Hz）；可选 Gaussian 平滑得连续接触强度，运行时再阈值。
- 跑步等短接触（1–2 帧 @30Hz）易漏标；60Hz + 三次插值算速度更稳。

### 4. 离线脚滑移除（Offline Foot Locking）

- 有整段 clip 时：把 pelvis + 左右趾当 **粒子链**，软约束保帧间/帧内相对几何，硬约束让 **接触帧趾粒子零相对位移**。
- 迭代 Jacobi 式投影（`softFactor≈0.05`, `hardFactor≈0.9`, 上万次迭代）；输出修正后的 pelvis/toe 轨迹再喂 `SolveLegChain`。
- 比运行时惯性化更能 **全局分配** 修正量，保源动画形状。

### 5. 结论 / 哲学

| 要点 | 含义 |
|------|------|
| 脚滑 = **速度误差** | 不是单纯违反地面摩擦；根运动缩放、混合局部旋转都会让视觉足速 ≠ 源数据足速 |
| 锁 **趾** 不锁跟 | 落地多为前掌；跟锁死反而丑；IK 重点约束趾 |
| IK 是 **修正** 非替换 | 在输入 pose 上叠加最小旋转，而非 pole-vector 式完全重算 |

## 对 wiki 的映射

- 升格 [`wiki/methods/foot-locking-ik-orangeduck.md`](../../wiki/methods/foot-locking-ik-orangeduck.md)
- 实体 [`wiki/entities/genoview-inverse-kinematics.md`](../../wiki/entities/genoview-inverse-kinematics.md)
- 交叉：[`wiki/formalizations/inverse-kinematics.md`](../../wiki/formalizations/inverse-kinematics.md)、[`wiki/concepts/motion-retargeting.md`](../../wiki/concepts/motion-retargeting.md)、[`wiki/methods/motion-retargeting-gmr.md`](../../wiki/methods/motion-retargeting-gmr.md)
