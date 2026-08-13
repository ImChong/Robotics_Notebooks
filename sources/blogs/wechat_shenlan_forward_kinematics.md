# 通俗讲透机器人正向运动学

> 来源归档（blog / 微信公众号）

- **标题：** 通俗讲透机器人正向运动学：从原理到工程实例，一次性讲明白！
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号；《具身智能基础》专栏第 8 篇）
- **原始链接：** https://mp.weixin.qq.com/s?__biz=MzkwMDcyNDUzMQ==&mid=2247506508&idx=1&sn=8c7041c9e509d8a83cf92ca3cb63023c
- **发表日期：** 2026-07-17
- **入库日期：** 2026-08-13
- **抓取方式：** Agent Reach v1.5.0 + wechat-article-for-ai（Camoufox）；专辑页同会话跳转；`--no-images` 解析
- **专栏专辑：** [《具身智能基础》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653)
- **专栏姊妹篇：** [齐次坐标](wechat_shenlan_homogeneous_coordinates_transform.md)；[逆运动学](wechat_shenlan_inverse_kinematics.md)；[雅可比](wechat_shenlan_robot_jacobian.md)
- **一句话说明：** FK 是单行道：$q\mapsto T\in\mathrm{SE}(3)$，输出唯一。工程载体是标准 DH 四参数（公垂线）→ 单连杆 $4\times4$ → 连乘；标准 DH 与 Craig 改进 DH **不可混用**。工程三场景：DH 标定（~1 mm→~0.05 mm）、运动学树实时全身 FK、Sim2Real 运动学一致性校验。

## 核心摘录（归纳，非全文）

### 位姿与连乘

末端 = 位置 $p$ + 姿态 $R\in\mathrm{SO}(3)$；齐次 $T$ 把旋转+平移焊成矩阵乘，故 $T_n^0 = T_1^0\cdots T_n^{n-1}$。

### DH 四参数

任意两空间直线用公垂线只需 4 参数。转动关节变量为 $\theta$，移动关节变量为 $d$。为何不是 6：关节轴约束掉 2 个自由度。

坐标系：$z_i$ 沿关节轴；$x_i$ 沿 $z_i$ 与 $z_{i+1}$ 公垂线；$y_i$ 右手系。

**标准 DH vs 改进 DH**：坐标系固联位置不同，变换矩阵不同；项目内选定一种贯彻到底。文内推导用标准 DH。

### 四步分解

绕 $z$ 转 $\theta$ → 沿 $z$ 移 $d$ → 沿 $x$ 移 $a$ → 绕 $x$ 转 $\alpha$。展开即单连杆 DH 矩阵。

### 平面 3R

$\alpha=d=0$，末端 $x=\sum \ell_i\cos(\theta_{1..i})$，$y=\sum \ell_i\sin(\theta_{1..i})$，朝向 $\sum\theta_i$——3D 臂只是同一套连乘、参数表更满。

### 工程落地

| 场景 | 做法 |
|------|------|
| 工业标定 | 激光跟踪 + LM 拟合 $d,a,\alpha$ |
| 人形 1 kHz | 运动学树按父索引拓扑一次算完全身 |
| Sim2Real | 同 $q$ 序列对比仿真/真机末端；合格约位置 <1 mm、姿态 <0.5° |

文内提醒：URDF 本质是 DH/几何的 XML 化；RL 位姿误差与 IK 都依赖同一套 FK。

## 对 wiki 的映射

- 升格 [`wiki/formalizations/forward-kinematics.md`](../../wiki/formalizations/forward-kinematics.md)。
- 交叉：[`wiki/formalizations/homogeneous-coordinates-transform.md`](../../wiki/formalizations/homogeneous-coordinates-transform.md)、[`wiki/formalizations/inverse-kinematics.md`](../../wiki/formalizations/inverse-kinematics.md)、[`wiki/entities/modern-robotics-book.md`](../../wiki/entities/modern-robotics-book.md)、[`wiki/entities/pinocchio.md`](../../wiki/entities/pinocchio.md)、[`roadmap/motion-control.md`](../../roadmap/motion-control.md)。
