# Animating Rotation with Quaternion Curves（Shoemake, SIGGRAPH 1985）

> 来源归档（ingest）

- **标题:** Animating Rotation with Quaternion Curves
- **作者:** Ken Shoemake
- **机构:** The Singer Company, Link Flight Simulation Division
- **Venue:** SIGGRAPH 1985（Proc. 12th annual conference on Computer graphics and interactive techniques）
- **DOI:** https://doi.org/10.1145/325334.325242
- **PDF 镜像:** http://graphics.cs.cmu.edu/nsp/course/15-464/Fall05/assignments/p245-shoemake.pdf
- **入库日期:** 2026-09-09
- **一句话说明:** 提出在单位四元数球面 $S^3$ 上做 **SLERP**（球面线性插值）及球面 Bézier 样条，解决欧拉角插值的万向锁与不自然路径；并阐明 **对径点 $q \equiv -q$ 同一旋转** 的拓扑与选号策略。

## 开源状态（步骤 2.5）

- **无官方代码**；SLERP 已写入 virtually 所有图形/机器人库（MimicKit `torch_util.slerp`、Unity、Blender 等）。
- **结论:** 算法一手资料为 SIGGRAPH 论文 PDF。

## 摘录 1：为何用四元数而非欧拉角插值

> Solid bodies roll and tumble through space … rotations are best described using … quaternions … unit quaternions … suitable for smoothly in-betweening sequences of arbitrary rotations.

- 欧拉角 **分轴独立插值** 不保持刚体旋转的群结构，路径 unnatural。
- 四元数乘法非交换，恰能表达「先绕 y 再绕 z ≠ 先绕 z 再绕 y」的复合（文内书本翻转示例）。

**对 wiki 的映射:** [`wiki/formalizations/unit-quaternion-so3.md`](../../wiki/formalizations/unit-quaternion-so3.md) — SLERP vs 欧拉 slerp 误区。

## 摘录 2：单位四元数 ↔ 轴角（Sec. 3.2）

单位四元数 $q=[w, \mathbf{v}]$ 对应旋转角 $\theta$、轴 $\hat u$：

$$
w = \cos\frac{\theta}{2}, \quad \|\mathbf{v}\| = \sin\frac{\theta}{2}, \quad \hat u = \mathbf{v}/\|\mathbf{v}\|
$$

矩阵对角和 $1+2\cos\theta = 4w^2-1$ 给出 $w$ 与旋转角关系。

**对 wiki 的映射:** 与 Diebel 轴角四元数（[`diebel_2006_representing_attitude_quaternions.md`](diebel_2006_representing_attitude_quaternions.md) Sec. 6.12）一致。

## 摘录 3：SLERP 公式（Sec. 3.3 Great arc in-betweening）

设 $q_1, q_2$ 为单位四元数，$\cos\theta = q_1 \cdot q_2$：

$$
\mathrm{Slerp}(q_1, q_2; u) = \frac{\sin((1-u)\theta)}{\sin\theta}\, q_1 + \frac{\sin(u\theta)}{\sin\theta}\, q_2
$$

- 插值沿 $S^3$ 大圆弧，角速度恒定。
- **选短弧:** 若 $q_1\cdot q_2 < 0$，取 $q_2 \leftarrow -q_2$（MimicKit `slerp` 中 `neg_mask` 同理）。

**对 wiki 的映射:** 工程实践节链到 MimicKit `quat_pos` / `slerp` 实现。

## 摘录 4：双覆盖与拓扑（Sec. 3.4）

> … north and south poles are the same! … each pair of opposite points represents the same rotation.

- SO(3) 几何像球面，但拓扑上 **对径等同**（转 360° 绳子打结 vs 720° 还原——物理上需 spinor 视角）。
- 从矩阵 lift 到四元数时，应选与 **插值链相邻** 的半球分支，避免路径跳变。

**对 wiki 的映射:** 与 [SE(3) 位姿表示](../../wiki/formalizations/se3-representation.md) 中四元数不连续 / 6D 连续族论述互补。

## 摘录 5：坐标无关性

> … motion is independent of coordinate axes. Euler interpolants … will do wildly different things.

**对 wiki 的映射:** 动画/MoCap 重定向管线应用 SLERP 于 **本体或世界系一致的四元数序列**，而非 RPY 分通道 lerp。

## 建议 wiki 动作

- 新建 **`wiki/formalizations/unit-quaternion-so3.md`** — SLERP 专节
- 交叉 [DeepMimic](../papers/deepmimic.md) / MoCap 姿态插值场景
