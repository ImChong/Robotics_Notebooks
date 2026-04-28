---
type: formalization
tags: [dynamics, contact, physics, control, manipulation, math]
status: complete
updated: 2026-04-23
related:
  - ./friction-cone.md
  - ../concepts/contact-dynamics.md
  - ../concepts/whole-body-control.md
  - ../concepts/contact-rich-manipulation.md
  - ../concepts/tactile-sensing.md
  - ../concepts/visuo-tactile-fusion.md
sources:
  - ../../sources/papers/contact_dynamics.md
  - ../../sources/papers/whole_body_control.md
summary: "接触力旋量锥（Contact Wrench Cone, CWC）把摩擦锥从单点三维力推广到包含力与力矩的六维力旋量，是多点面接触下判断支撑可行性、CoM 可行区域与抓取稳定性的核心凸约束。"
---

# 接触力旋量锥 (Contact Wrench Cone)

**接触力旋量锥 (Contact Wrench Cone, CWC)** 把 [摩擦锥 (Friction Cone)](./friction-cone.md) 从“单点三维力”推广到“面接触的六维力旋量”。它描述了一个刚性接触面在不脱离、不滑移的前提下能够向被控物体（足或被抓物体）施加的**所有合力 + 合力矩（wrench）**组成的凸锥，是人形机器人多点支撑平衡与抓取稳定性分析的统一数学工具。

## 从点接触到面接触

单点接触时，接触只能传递 3 维力 $\mathbf{f} \in \mathbb{R}^3$，其可行集是摩擦锥 $\mathcal{F}$。

对一个有限接触面（例如脚底矩形、指尖弹性垫），接触被等效为一个 **接触坐标系** $\{C\}$ 原点上的 6 维 **力旋量 (wrench)**：

$$
\mathbf{w} = \begin{bmatrix} \mathbf{f} \\ \boldsymbol{\tau} \end{bmatrix} \in \mathbb{R}^6
$$

其中 $\mathbf{f}$ 是合力，$\boldsymbol{\tau}$ 是相对 $\{C\}$ 原点的合力矩。CWC 给出了 $\mathbf{w}$ 必须满足的所有物理约束的凸包。

## 两种等价表示

### 1. V-representation（顶点表示）

把接触面离散化为 $N$ 个顶点 $\{\mathbf{p}_i\}_{i=1}^{N}$。每个顶点处的接触力 $\mathbf{f}_i$ 位于该顶点的线性化摩擦锥中，可进一步展开为 $K$ 条边缘方向 $\{\mathbf{s}_{i,k}\}$ 的非负线性组合：

$$
\mathbf{f}_i = \sum_{k=1}^{K} \lambda_{i,k}\, \mathbf{s}_{i,k}, \quad \lambda_{i,k} \geq 0
$$

该顶点力对 $\{C\}$ 原点的贡献为 $\begin{bmatrix} \mathbf{f}_i \\ \mathbf{p}_i \times \mathbf{f}_i \end{bmatrix}$。整个接触面在 $\{C\}$ 下的可行 wrench 就是

$$
\mathcal{W} = \left\{ \sum_{i=1}^{N}\sum_{k=1}^{K} \lambda_{i,k} \begin{bmatrix} \mathbf{s}_{i,k} \\ \mathbf{p}_i \times \mathbf{s}_{i,k} \end{bmatrix} \;\middle|\; \lambda_{i,k} \geq 0 \right\}
$$

这正是由有限个“基础 wrench”张成的**有限凸锥**。

### 2. H-representation（面表示）

对 V-representation 做一次 double-description / cdd 变换，得到等价的线性不等式形式：

$$
\mathbf{A}\,\mathbf{w} \leq \mathbf{0}
$$

矩阵 $\mathbf{A}$ 的每一行是 CWC 的一个支撑超平面，直接可以塞进 QP / LP 作为线性约束。对标准矩形脚底，常见的显式分量约束（在接触坐标系下）是：

- 法向单向：$f_z \geq 0$
- 切向摩擦：$|f_x| \leq \mu f_z,\; |f_y| \leq \mu f_z$
- ZMP 不越出支撑矩形：$|\tau_x| \leq Y\, f_z,\; |\tau_y| \leq X\, f_z$（$X, Y$ 为脚底半长/半宽）
- 绕法向扭矩受摩擦限制：$|\tau_z| \leq \tau_{z,\max}(\mu, X, Y, f_z)$

最后一项是 CWC 相对摩擦锥真正新增的“偏航摩擦耦合”，反映了面接触抗自旋的能力。

## 与 ZMP / CoP 的关系

对于单接触面，ZMP / CoP 的 2D 位置可以直接由 wrench 的前两项力矩给出：

$$
p_x = -\tau_y / f_z, \qquad p_y = \tau_x / f_z
$$

“ZMP 不出支撑多边形”就是 CWC 中 $|\tau_x| \leq Y f_z,\; |\tau_y| \leq X f_z$ 的几何解读。也就是说：

- **点接触 + 摩擦锥** → 只保证不滑；
- **面接触 + CWC** → 同时保证不滑、不翻转、不自旋。

## 多接触与质心动力学

多接触时，每个接触面生成自己的 CWC $\mathcal{W}_i$，经过接触系到世界系的伴随变换 $\mathrm{Ad}_{T_i}^{\top}$ 映射到世界原点后求和：

$$
\mathbf{w}_{\text{net}} = \sum_{i} \mathrm{Ad}_{T_i}^{\top}\, \mathbf{w}_i, \qquad \mathbf{w}_i \in \mathcal{W}_i
$$

由 [Centroidal Dynamics](../concepts/centroidal-dynamics.md)：

$$
\mathbf{w}_{\text{net}} = \begin{bmatrix} m(\ddot{\mathbf{c}} - \mathbf{g}) \\ \dot{\mathbf{L}}_G \end{bmatrix}
$$

因此 CWC 的 Minkowski 和（Contact Wrench Sum, CWS）就是 CoM 可行的“加速度 + 角动量变化”集合。这是 **CWS 判据**：只要所需 $\mathbf{w}_{\text{net}}$ 落在 CWS 内，就存在不滑不翻的接触力分配。

## 在抓取与操作中的版本

在灵巧抓取分析中，CWC 换一个名字叫 **Grasp Wrench Space (GWS)**：

- 每根手指的指尖摩擦锥 → 手指的接触 wrench 锥
- 所有手指 wrench 锥的 Minkowski 和 → GWS
- **Force closure** ⇔ GWS 包含原点的某邻域（可以抵抗任意外扰 wrench）
- **抓取质量**常取 GWS 内切球半径（Ferrari-Canny 度量）

这把“抓得牢不牢”变成了一个几何判断：**扰动 wrench 是否落在 GWS 内**。

## 最小实现骨架

```python
# 伪代码：矩形脚底在接触坐标系下的 CWC 线性不等式
def rectangular_foot_cwc(mu, X, Y, fz_min=5.0):
    # 状态：[fx, fy, fz, tx, ty, tz]
    A = []
    A += [[ 0,  0, -1, 0, 0, 0]]              # fz >= fz_min
    A += [[ 1,  0, -mu, 0, 0, 0],
          [-1,  0, -mu, 0, 0, 0],
          [ 0,  1, -mu, 0, 0, 0],
          [ 0, -1, -mu, 0, 0, 0]]              # 摩擦锥（线性化）
    A += [[ 0,  0, -Y, 0,  1, 0],
          [ 0,  0, -Y, 0, -1, 0],
          [ 0,  0, -X, 1,  0, 0],
          [ 0,  0, -X,-1,  0, 0]]              # CoP 在矩形内
    A += [[ 0,  0, -mu*(X+Y), 0, 0,  1],
          [ 0,  0, -mu*(X+Y), 0, 0, -1]]       # 偏航摩擦（近似）
    b = [-fz_min] + [0]*10
    return A, b  # A w <= b
```

## 方法局限性

- **线性化误差**：把真正的二阶锥（SOC）换成多面体，会高估或低估可行集，取决于内切/外接选择。
- **刚性平面假设**：面接触模型假设接触面刚性且平坦，对软垫手指或不平整地面会偏乐观。
- **静态凸锥**：CWC 本身只刻画一瞬间的可行集，真实动态演化需要和 [Contact Dynamics](../concepts/contact-dynamics.md)、[Centroidal Dynamics](../concepts/centroidal-dynamics.md) 一起使用。

## 学这个方法时最该盯住的点

1. “摩擦锥只约束力，CWC 同时约束力矩”——先把这一条记死。
2. ZMP / CoP 判据只是 CWC 在矩形脚底、单点接触简化下的**投影**，不要反过来以为它更一般。
3. Grasp 的 force closure 和 Loco 的 CWS 判据其实是**同一个数学对象** CWC 在不同任务下的换名字。

## 关联页面
- [摩擦锥 (Friction Cone)](./friction-cone.md)
- [Contact Dynamics (接触动力学)](../concepts/contact-dynamics.md)
- [Whole-Body Control](../concepts/whole-body-control.md)
- [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md)
- [Tactile Sensing](../concepts/tactile-sensing.md)

## 参考来源
- [contact_dynamics.md](../../sources/papers/contact_dynamics.md)
- [whole_body_control.md](../../sources/papers/whole_body_control.md)
- Hirukawa, H., et al. (2006). *A Universal Stability Criterion of the Foot Contact of Legged Robots — Adios ZMP*.
- Caron, S., Pham, Q.-C., Nakamura, Y. (2015). *Leveraging Cone Double Description for Multi-contact Stability Computations*.
- Ferrari, C., Canny, J. (1992). *Planning optimal grasps*.
