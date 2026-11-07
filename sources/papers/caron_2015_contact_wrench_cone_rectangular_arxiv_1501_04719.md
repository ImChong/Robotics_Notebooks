# Stability of Surface Contacts for Humanoid Robots: Closed-Form Formulae of the Contact Wrench Cone for Rectangular Support Areas（Caron, Pham, Nakamura, 2015）

> 来源归档（ingest · 原文通读）

- **标题：** Stability of Surface Contacts for Humanoid Robots: Closed-Form Formulae of the Contact Wrench Cone for Rectangular Support Areas
- **作者：** Stéphane Caron、Quang-Cuong Pham、Yoshihiko Nakamura（东京大学 / 南洋理工大学）
- **Venue：** ICRA 2015
- **arXiv：** <https://arxiv.org/abs/1501.04719>（v1，2015-01-20）
- **PDF：** <https://arxiv.org/pdf/1501.04719>
- **类型：** paper / contact-stability / humanoid
- **入库日期：** 2026-11-07
- **核对版本：** arXiv v1 PDF 全文
- **核对范围：** 读完摘要、§1 引言、§2 接触稳定性定义（运动方程、wrench 与接触雅可比、互补条件、弱稳定性）、§3 面接触物理与命题 1（凸多边形顶点接触力等价）、§4 矩形支撑面 wrench cone 闭式解（命题 2 式 15–20、$\tau_{safe}$）、§5 OpenHRP / HRP-4 的 TOPP 实验与结论、附录 A.1 前半（式 4 证明）；**未读：** 命题 2 的完整附录证明细节、作者脚本网页、参考文献原文。
- **通读：** 已完成
- **开源状态：** 论文无官方仓库；作者在 scaron.info 公开验证脚本（未核对）。

## 关键内容摘录

- **动机：** ZMP 对接触稳定性必要但不充分——不管水平面滑动与绕竖轴（yaw）旋转；用各顶点接触力会引入冗余（3×点数 ≫ 6）。把「每个接触力在摩擦锥内」换成「接触 wrench 在 wrench cone 内」后维度只有 6。
- **运动方程：** $M(q)\ddot q+h=S^\top\tau_a+\sum_iJ(C_i)^\top f_i=S^\top\tau_a+J_{wr}^\top W$，$W=(f,\tau)$，$f=\sum f_i$，$\tau=\sum \overrightarrow{OC_i}\wedge f_i$（式 1–5）。
- **命题 1：** 支撑面为凸多边形时，若存在满足单边 + Coulomb 约束的压力 / 剪应力场，则存在仅作用于各顶点、合 wrench 相同且满足同样约束的接触力。
- **命题 2（矩形，线性化摩擦锥 $|f_{ix}|,|f_{iy}|\le\mu f_{iz}$，$f_{iz}>0$，中心系 wrench $W=(f^x,f^y,f^z,\tau^x,\tau^y,\tau^z)$，矩形半边长 $X,Y$）：** 存在满足约束的顶点力当且仅当
  $|f^x|\le\mu f^z,\ |f^y|\le\mu f^z,\ f^z>0,\ |\tau^x|\le Yf^z,\ |\tau^y|\le Xf^z,\ \tau_{\min}\le\tau^z\le\tau_{\max}$，
  其中 $\tau_{\min}=-\mu(X+Y)f^z+|Yf^x-\mu\tau^x|+|Xf^y-\mu\tau^y|$，$\tau_{\max}=+\mu(X+Y)f^z-|Yf^x+\mu\tau^x|-|Xf^y+\mu\tau^y|$。前两式为 Coulomb 摩擦，三四式等价于 ZMP 在支撑面内，最后一式（yaw 力矩上下界）作者称据其所知为新结果。
- **yaw 耦合：** 切向力或横滚 / 俯仰力矩增大会压缩可用 yaw 力矩范围；「最安全」的 $\tau^z$ 不是 0，而是 $\tau_{safe}=(\tau_{\min}+\tau_{\max})/2$，可作为抑制足底 yaw 的控制律。
- **实验：** HRP-4 在 OpenHRP 中单脚支撑，手工 11 个关键姿态 + Bézier + TOPP 重定时（7.285 s）。加安全裕度（接触面缩 45%、$\mu=0.4$，仿真真值 $\mu=0.8$）后可加速约 5% 成功；加速 10% 触发失去面接触但稳定器可恢复，15% 摔倒。TOPP 的时间最优性使轨迹至少饱和一项接触约束，所以结果恰在极限边缘。

## 局限（作者原文 + 页面核对时的观察）

- 仅推导**矩形**支撑面、线性化（棱锥）摩擦锥；一般多边形需另行计算（本页其他内容不属原文）。
- 假设环境与接触连杆为刚体、摩擦系数 $\mu$ 取常数（作者注明真实 $\mu(x,y)$ 随位置变化）；论文结论为单脚支撑的实验证明，未覆盖多接触。

**对 wiki 的映射：** [`wiki/formalizations/contact-wrench-cone.md`](../../wiki/formalizations/contact-wrench-cone.md)
