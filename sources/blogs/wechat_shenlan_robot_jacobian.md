# 雅可比矩阵：统一速度映射与力映射

> 来源归档（blog / 微信公众号）

- **标题：** 聊聊雅可比矩阵，如何统一机器人控制的两条主线？
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号；《具身智能基础》专栏第 10 篇）
- **原始链接：** https://mp.weixin.qq.com/s?__biz=MzkwMDcyNDUzMQ==&mid=2247507685&idx=1&sn=f240a287b15dfb1fc7ebb4804f61a359
- **发表日期：** 2026-08-07
- **入库日期：** 2026-08-13
- **抓取方式：** Agent Reach v1.5.0 + wechat-article-for-ai（Camoufox）；专辑页同会话跳转；`--no-images` 解析
- **专栏专辑：** [《具身智能基础》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653)
- **专栏姊妹篇：** [正向运动学](wechat_shenlan_forward_kinematics.md)；[逆运动学](wechat_shenlan_inverse_kinematics.md)
- **一句话说明：** 任务空间目标必须在关节空间执行。FK 是位置地图，雅可比是当前构型的局部比例尺：$v=J\dot q$ 与 $\tau=J^\top F$ 对偶，同一座桥两个方向。IK / WBC / MPC / RL 都在反复调用这层局部运动学接口。奇异与零空间是边界：秩亏则速度/力爆炸，冗余则 $J\dot q=0$ 可做次级任务。

## 核心摘录（归纳，非全文）

### 两个空间

- **关节空间**：维数 = 电机数；指令是各关节位置/速度。
- **任务空间**：通常 6 维（3 线速度 + 3 角速度）；人规划的是末端位姿、接触力。

### 几何列

第 $i$ 列 = 仅第 $i$ 关节单位速度时末端的速度贡献。转动关节：角速度沿轴 $\omega_i$，线速度 $\omega_i\times(p_e-p_i)$；移动关节只贡献线速度。末端速度是各列线性叠加。

### 力对偶

虚功：$\tau=J^\top F$。阻抗/导纳/力位混合都在反复调用 $J$ 与 $J^\top$。打磨「法向恒力 + 切向匀速」= 任务空间力/速度经 $J$ 分配到关节。

### 算法主线

| 方法 | 如何用 $J$ |
|------|------------|
| 数值 IK | $J^+$ 把位姿误差变成关节修正；冗余加零空间 |
| WBC | 多任务速度/力约束投影到关节 |
| MPC | 工作点局部线性化 |
| RL | 提供「哪个关节对末端最敏感」的局部结构 |

### 边界

奇异：秩下降，同样末端速度要极大关节速度。零空间：冗余时 $J\dot q=0$ 不改末端，可避障/节能。$J$ 只是当前构型最佳线性近似，不是全局答案。

## 对 wiki 的映射

- 升格 [`wiki/formalizations/robot-jacobian.md`](../../wiki/formalizations/robot-jacobian.md)。
- 交叉：[`wiki/formalizations/inverse-kinematics.md`](../../wiki/formalizations/inverse-kinematics.md)、[`wiki/concepts/whole-body-control.md`](../../wiki/concepts/whole-body-control.md)、[`wiki/methods/model-predictive-control.md`](../../wiki/methods/model-predictive-control.md)、[`wiki/concepts/tsid.md`](../../wiki/concepts/tsid.md)、[`roadmap/motion-control.md`](../../roadmap/motion-control.md)。
