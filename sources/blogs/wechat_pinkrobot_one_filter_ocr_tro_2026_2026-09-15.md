# TRO 2026 | 一个滤波器适配所有控制器：未知环境下四足机器人鲁棒安全导航【文献解读】

> 来源归档（blog / 微信公众号）

- **标题：** TRO 2026 | 一个滤波器适配所有控制器：未知环境下四足机器人鲁棒安全导航【文献解读】
- **类型：** blog
- **作者：** PinkRobot（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/sxpyCHEgK3qtAcknpXPgAA
- **发表日期：** 2026-09-15（入库日）
- **入库日期：** 2026-09-15
- **抓取方式：** WebFetch
- **论文：** One Filter to Deploy Them All: Robust Safety for Quadrupedal Navigation in Unknown Environments
- **期刊：** IEEE TRO, Vol. 42, pp. 545–560, 2026
- **项目页：** <https://sia-lab-git.github.io/One_Filter_to_Deploy_Them_All/>
- **代码：** <https://github.com/albertklin/observation-conditioned-reachability>
- **一句话说明：** OCR（Observation-Conditioned Reachability）：离线 HJ 可达性监督训练 OCR-VN（LiDAR + 降阶状态 + 扰动界 → 安全价值与梯度），在线用最近状态–动作历史估计动力学误差，QP 最小修改 nominal twist；**控制器无关**安全盾。

## 步骤 2.5（开源核查）

- **已开源：** <https://github.com/albertklin/observation-conditioned-reachability>
- **依赖：** `hj_reachability` 生成 ground-truth；100 束 LiDAR；3D Dubins 降阶 + 闭环等效扰动界 $\bar{d}$

## 核心摘录（归纳）

### 解决的双重泛化

1. **环境泛化：** 部署前未知障碍布局（LiDAR 观测条件化，非全局地图）
2. **动力学泛化：** 摩擦、负载、低层跟踪误差、降阶模型误差、外扰——统一吸收进扰动界

### 与「专用 safety critic」的区别

学习 **策略无关** 的 HJ 最优安全价值；只要 nominal controller 以 $(v_x, \omega_z)$ twist 接入即可复用——「One Filter to Deploy Them All」。

### 在线安全滤波（Equation 8–9）

- BRT 外：nominal 原样执行
- 近边界：QP 求最小偏离 nominal 的安全 twist
- **Conformal prediction** 校准神经网络高估安全（部署用 $\hat{V} - 0.49$ m 级 calibration level，非固定几何距离）

### 降阶关键

全维四足 HJ 网格不可行 → 把「本体 + 低层 locomotion policy」视为 3D Dubins $(x,y,\theta)$ + 扰动；安全层不需知道 12 关节力矩如何算。

## 对 wiki 的映射

- **新建：** [paper-one-filter-ocr-quadruped-navigation](../../wiki/entities/paper-one-filter-ocr-quadruped-navigation.md)
- **交叉：** [Optimal Control](../../wiki/concepts/optimal-control.md)、[Locomotion](../../wiki/tasks/locomotion.md)、[Sim2Real](../../wiki/concepts/sim2real.md)
