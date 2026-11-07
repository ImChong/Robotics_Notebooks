---
type: formalization
tags: [dynamics, contact, physics, control, manipulation, math]
status: complete
updated: 2026-11-07
related:
  - ./friction-cone.md
  - ../concepts/contact-dynamics.md
  - ../concepts/whole-body-control.md
  - ../concepts/contact-rich-manipulation.md
  - ../concepts/tactile-sensing.md
  - ../concepts/visuo-tactile-fusion.md
original:
  - ../../sources/papers/caron_2015_contact_wrench_cone_rectangular_arxiv_1501_04719.md
  - ../../sources/papers/modern_robotics_textbook.md
sources:
  - ../../sources/papers/caron_2015_contact_wrench_cone_rectangular_arxiv_1501_04719.md
  - ../../sources/papers/modern_robotics_textbook.md
  - ../../sources/papers/contact_dynamics.md
  - ../../sources/papers/whole_body_control.md
summary: "接触力旋量锥（Contact Wrench Cone, CWC）把摩擦锥从单点三维力推广到包含力与力矩的六维力旋量，是多点面接触下判断支撑可行性、CoM 可行区域与抓取稳定性的核心凸约束。"
---

# 接触力旋量锥 (Contact Wrench Cone)

## 一句话定义

**接触力旋量锥 (Contact Wrench Cone, CWC)** 把 [摩擦锥 (Friction Cone)](./friction-cone.md) 从“单点三维力”推广到“面接触的六维力旋量”。它描述了一个刚性接触面在不脱离、不滑移的前提下能够向被控物体（足或被抓物体）施加的**所有合力 + 合力矩（wrench）**组成的凸锥，是人形机器人多点支撑平衡与抓取稳定性分析的统一数学工具。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| CWC | Contact Wrench Cone | 刚性接触面不脱离、不滑移时可传递的全部合力 + 合力矩（6 维 wrench）构成的凸锥 |
| GWS | Grasp Wrench Space | 抓取分析里对应的 wrench 集合（各指接触 wrench 锥的 Minkowski 和） |
| CoP | Center of Pressure | 接触面上压力合力的作用点；单接触面下与 ZMP 重合 |
| CoM | Center of Mass | 质心，平衡与 locomotion 规划的核心状态量 |
| QP | Quadratic Programming | 将 WBC/控制问题写成二次规划的标准求解形式 |
| ZMP | Zero Moment Point | 足式平衡判据，地面反力合力矩为零的点 |
| Manipulation | Robot Manipulation | 抓取、移动、操作物体的任务总称 |

## 为什么重要

用「每个接触点一个三维力」描述多点 / 面接触会有冗余：三倍点数的力变量对应只有 6 维的合 wrench（Caron et al. 2015），逆动力学与时间最优轨迹参数化（TOPP）都因此变难。ZMP 判据对接触稳定只是必要条件，不管水平滑动与绕竖轴旋转（yaw）。CWC 把约束直接写在 6 维 wrench 上，一次覆盖不滑、不翻、不转，是多接触平衡规划、CoM 可行域和抓取稳定性共用的凸约束。

## 核心原理

### 从点接触到面接触

单点接触时，接触只能传递 3 维力 $\mathbf{f} \in \mathbb{R}^3$，其可行集是摩擦锥 $\mathcal{F}$。

*Modern Robotics* §12.2 的点接触版本：库仑摩擦 $\sqrt{f_x^2+f_y^2}\le\mu f_z$、$f_z\ge0$，半角 $\alpha=\tan^{-1}\mu$，$\mu$ 典型 0.1–1；多接触的可传递 wrench 是各接触 wrench 的正张成（凸锥）。接触力换成 wrench 后教材写作 $F=([p]f,f)$（**力矩在前**），本页写作 $\mathbf w=(\mathbf f,\boldsymbol\tau)$（力在前），对照教材时要换序。

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
\mathcal{W} = \left\{ \sum_{i=1}^{N}\sum_{k=1}^{K} \lambda_{i,k} \begin{bmatrix} \mathbf{s}_{i,k} \\ \mathbf{p}_i \times \mathbf{s}_{i,k} \end{bmatrix} \mid \lambda_{i,k} \geq 0 \right\}
$$

这正是由有限个“基础 wrench”张成的**有限凸锥**。Caron et al. 的命题 1 说明：对凸多边形支撑面，只要连续的压力 / 剪应力场满足单边与库仑约束，就存在仅作用于各顶点、合 wrench 相同且同样满足约束的顶点力——这是只取顶点离散化的理论依据。

### 2. H-representation（面表示）

对 V-representation 做一次 double-description / cdd 变换，得到等价的线性不等式形式：

$$
\mathbf{A}\,\mathbf{w} \leq \mathbf{0}
$$

矩阵 $\mathbf{A}$ 的每一行是 CWC 的一个支撑超平面，直接可以塞进 QP / LP 作为线性约束。对矩形脚底（接触系原点在中心，顶点 $(\pm X,\pm Y)$，顶点摩擦锥取线性化 $|f_{ix}|,|f_{iy}|\le\mu f_{iz}$，$f_{iz}>0$），Caron, Pham & Nakamura（ICRA 2015，arXiv:1501.04719）命题 2 给出闭式解：存在满足约束的顶点力，当且仅当

- 法向单向：$f_z > 0$
- 切向摩擦：$|f_x| \leq \mu f_z,\; |f_y| \leq \mu f_z$
- ZMP 不越出支撑矩形：$|\tau_x| \leq Y\, f_z,\; |\tau_y| \leq X\, f_z$（$X, Y$ 为脚底半长 / 半宽）
- 绕法向扭矩（yaw）有上下界：$\tau_{\min}\le\tau_z\le\tau_{\max}$，其中
  $\tau_{\min}=-\mu(X+Y)f_z+|Yf_x-\mu\tau_x|+|Xf_y-\mu\tau_y|$，
  $\tau_{\max}=+\mu(X+Y)f_z-|Yf_x+\mu\tau_x|-|Xf_y+\mu\tau_y|$

最后一项是 CWC 相对摩擦锥新增的“偏航摩擦耦合”（作者称据其所知为新结果）：切向力或横滚 / 俯仰力矩越大，可用 yaw 力矩范围越窄，所以它不是简单的 $|\tau_z|\le\mu(X+Y)f_z$。最“安全”的 yaw 力矩不是 0，而是 $\tau_{safe}=(\tau_{\min}+\tau_{\max})/2$，论文用它作抑制足底偏转的控制律。

> **线性化方向（本页推导，非论文原话）：** $|f_x|,|f_y|\le\mu f_z$ 是包住圆锥的方棱锥（角点切向合力达 $\sqrt2\mu f_z$），会高估摩擦；*Modern Robotics* 的内接棱锥边为 $(\pm\mu,0,1),(0,\pm\mu,1)$，低估摩擦。要保守，可把上式的 $\mu$ 换成 $\mu/\sqrt2$ 或加摩擦裕度（论文实验即用缩小 45% 接触面与 $\mu=0.4$（仿真真值 0.8）作安全裕度）。

## 与 ZMP / CoP 的关系

对于单接触面，ZMP / CoP 的 2D 位置可以直接由 wrench 的前两项力矩给出：

$$
p_x = -\tau_y / f_z, \qquad p_y = \tau_x / f_z
$$

“ZMP 不出支撑多边形”就是 CWC 中 $|\tau_x| \leq Y f_z,\; |\tau_y| \leq X f_z$ 的几何解读（Caron et al. 称 ZMP 对接触稳定必要不充分）。也就是说：

- **点接触 + 摩擦锥** → 只保证不滑；
- **面接触 + CWC** → 同时保证不滑、不翻转、不自旋。

## 多接触与质心动力学

多接触时，每个接触面生成自己的 CWC $\mathcal{W}_i$，经过接触系到质心系 $\{G\}$ 的伴随变换（*Modern Robotics* 的 wrench 换系规则为 $F_a=[\mathrm{Ad}_{T_{ba}}]^\top F_b$，教材 wrench 力矩在前）映射到质心后求和：

$$
\mathbf{w}_{\text{net}} = \sum_{i} \mathrm{Ad}_{T_i}^{\top}\, \mathbf{w}_i, \qquad \mathbf{w}_i \in \mathcal{W}_i
$$

（此处 $\mathbf w_{\text{net}}$ 是对质心取矩；Herzog et al. 2016 式 21 同样把动量变化率写成「接触力在质心的合力 / 合力矩 + 重力项」。）

由 [Centroidal Dynamics](../concepts/centroidal-dynamics.md)：

$$
\mathbf{w}_{\text{net}} = \begin{bmatrix} m(\ddot{\mathbf{c}} - \mathbf{g}) \\ \dot{\mathbf{L}}_G \end{bmatrix}
$$

因此 CWC 的 Minkowski 和（Contact Wrench Sum, CWS）就是 CoM 可行的“加速度 + 角动量变化”集合。这是 **CWS 判据**：只要所需 $\mathbf{w}_{\text{net}}$ 落在 CWS 内，就存在不滑不翻的接触力分配。

## 在抓取与操作中的版本

在灵巧抓取分析中，CWC 换一个名字叫 **Grasp Wrench Space (GWS)**：

- 每根手指的指尖摩擦锥 → 手指的接触 wrench 锥
- 所有手指 wrench 锥的 Minkowski 和 → GWS
- **Force closure**：*Modern Robotics* §12.2.3 定义为接触 wrench 的凸锥等于整个 wrench 空间（$\mathbb{R}^3$ 或 $\mathbb{R}^6$）；无摩擦时等价于一阶形封闭，一阶形封闭平面体至少 4 个点接触、空间体至少 7 个（教材摘要）。「GWS 包含原点的某邻域」是对该定义的等价说法（本页推导）
- **抓取质量**常取 GWS 内切球半径（Ferrari–Canny 度量，本页未核对原文，推测）

这把“抓得牢不牢”变成了一个几何判断：**扰动 wrench 是否落在 GWS 内**。

## 工程实践

1. **作为线性约束放进 QP / TOPP：** 绝对值项展开成 $2\times2$ 的符号组合后整体仍是线性不等式。下面的实现按 Caron et al. 命题 2 写出，与顶点力采样对照验证过（2 万组随机顶点力无违反，yaw 上界在 $f_x=f_y=\tau_x=\tau_y=0$ 时取到 $\mu(X+Y)f_z$）。
2. **单脚支撑直接解 wrench：** 单接触时 $W=(P J_{wr})^{-1}P\,[M\ddot q+h]$（$P$ 为非驱动行选择矩阵），无需再分配顶点力；论文用它接入 TOPP，在 HRP-4 / OpenHRP 仿真重定时 7.285 s 的动作，加裕度后可再加速约 5%，加速 10% 开始失去面接触，15% 摔倒。
3. **留裕度：** TOPP 的时间最优性会使轨迹至少饱和一个接触约束，所以在名义极限上执行没有余量；缩接触面、降 $\mu$。

```python
# 矩形脚底在接触坐标系下的 CWC：A w <= b，w = [fx, fy, fz, tx, ty, tz]
# 依据 Caron, Pham, Nakamura (ICRA 2015) 命题 2；X, Y 为半长 / 半宽
def rectangular_foot_cwc(mu, X, Y, fz_min=0.0):
    A, b = [[0, 0, -1, 0, 0, 0]], [-fz_min]            # fz >= fz_min
    for s in (1, -1):
        A += [[s, 0, -mu, 0, 0, 0], [0, s, -mu, 0, 0, 0],   # |fx|,|fy| <= mu fz
              [0, 0, -Y, s, 0, 0], [0, 0, -X, 0, s, 0]]     # |tx| <= Y fz, |ty| <= X fz
        b += [0, 0, 0, 0]
    for s1 in (1, -1):
        for s2 in (1, -1):                              # tau_min <= tz <= tau_max
            A += [[s1*Y, s2*X, -mu*(X+Y),  s1*mu,  s2*mu,  1],
                  [s1*Y, s2*X, -mu*(X+Y), -s1*mu, -s2*mu, -1]]
            b += [0, 0]
    return A, b
```

## 方法局限性

- **线性化误差**：把真正的二阶锥（SOC）换成多面体，会高估或低估可行集，取决于内切/外接选择（见上文线性化方向）。
- **只推导了矩形**：闭式解仅对矩形支撑面；一般多边形需另做 double description（本页引用的 Caron 2015 cdd 文献未逐篇核对）。
- **常数 $\mu$、刚体假设**：论文假设摩擦系数处处相同（真实 $\mu(x,y)$ 随位置变化）、环境与连杆为刚体。
- **刚性平面假设**：面接触模型假设接触面刚性且平坦，对软垫手指或不平整地面会偏乐观。
- **静态凸锥**：CWC 本身只刻画一瞬间的可行集，真实动态演化需要和 [Contact Dynamics](../concepts/contact-dynamics.md)、[Centroidal Dynamics](../concepts/centroidal-dynamics.md) 一起使用。

## 学这个方法时最该盯住的点

1. “摩擦锥只约束力，CWC 同时约束力矩”——先把这一条记死。
2. ZMP / CoP 判据只是 CWC 在矩形脚底、单点接触简化下的**投影**，不要反过来以为它更一般。
3. Grasp 的 force closure 和 Loco 的 CWS 判据其实是**同一个数学对象** CWC 在不同任务下的换名字。

## 关联页面
- [Modern Robotics 教材](../entities/modern-robotics-book.md) — Ch 12 摩擦锥、wrench cone 与力封闭
- [Centroidal Dynamics](../concepts/centroidal-dynamics.md) — 合 wrench 与质心动量变化率的关系
- [仿真物理保真度链路选型指南](../queries/simulation-physics-fidelity.md) — 本页所述物理/仿真要素在保真度链路（建模 ① → 数值 ② → 接触 ③ → 随机化 ④）中的定位
- [摩擦锥 (Friction Cone)](./friction-cone.md)
- [Contact Dynamics (接触动力学)](../concepts/contact-dynamics.md)
- [Whole-Body Control](../concepts/whole-body-control.md)
- [Contact-Rich Manipulation](../concepts/contact-rich-manipulation.md)
- [Tactile Sensing](../concepts/tactile-sensing.md)

## 参考来源
- [Caron, Pham & Nakamura (2015)，arXiv:1501.04719 原文归档](../../sources/papers/caron_2015_contact_wrench_cone_rectangular_arxiv_1501_04719.md) — 矩形 CWC 闭式解、yaw 界
- [Modern Robotics 教材归档](../../sources/papers/modern_robotics_textbook.md) — §12.2 摩擦锥与 wrench cone
- [contact_dynamics.md](../../sources/papers/contact_dynamics.md)
- [whole_body_control.md](../../sources/papers/whole_body_control.md)
- Hirukawa, H., et al. (2006). *A Universal Stability Criterion of the Foot Contact of Legged Robots — Adios ZMP*.
- Caron, S., Pham, Q.-C., Nakamura, Y. (2015). *Leveraging Cone Double Description for Multi-contact Stability Computations*.
- Ferrari, C., Canny, J. (1992). *Planning optimal grasps*.

## 推荐继续阅读

- [Caron et al., ICRA 2015 (arXiv:1501.04719)](https://arxiv.org/abs/1501.04719) — 矩形面接触 wrench cone 闭式推导与 OpenHRP 实验
- [Modern Robotics 官方 PDF](https://hades.mech.northwestern.edu/images/7/7f/MR.pdf) — Ch 12 接触运动学、摩擦与力封闭
