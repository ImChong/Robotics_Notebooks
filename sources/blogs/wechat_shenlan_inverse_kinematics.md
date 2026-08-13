# 机器人逆运动学（IK）：五个关键点

> 来源归档（blog / 微信公众号）

- **标题：** 机器人逆运动学（IK）到底在算什么？五个关键点从数学本质到工程落地
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号；《具身智能基础》专栏第 9 篇）
- **原始链接：** https://mp.weixin.qq.com/s?__biz=MzkwMDcyNDUzMQ==&mid=2247506764&idx=1&sn=45d102a22d570c268435bca714a5e088
- **发表日期：** 2026-07-23
- **入库日期：** 2026-08-13
- **抓取方式：** Agent Reach v1.5.0 + wechat-article-for-ai（Camoufox）；专辑页同会话跳转；`--no-images` 解析
- **专栏专辑：** [《具身智能基础》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653)
- **专栏姊妹篇：** [正向运动学](wechat_shenlan_forward_kinematics.md)；[雅可比](wechat_shenlan_robot_jacobian.md)；[RL 求解 IK](wechat_shenlan_rl_inverse_kinematics.md)
- **一句话说明：** IK 是 $T^\star\mapsto q$，解可能 0/多/无穷，且在奇异点数值崩溃。五条主线：解析（Pieper/球腕）→ 雅可比迭代 + DLS → 奇异/操纵度 → 冗余零空间任务优先级 → 学习型 IK（IKFlow）。场景：工业分拣用闭式、7 轴避障用零空间、灵巧手用生成式候选池。

## 核心摘录（归纳，非全文）

### 问题变质

FK：$q\mapsto T$ 唯一、$\mathcal{O}(n)$ 矩阵乘。IK：非线性三角、工作空间外无解、6DOF 球腕最多 16 组、冗余 $n>6$ 无穷多。

### 解析解

Pieper（后三轴共点）→ 腕心位置解 $\theta_{1,2,3}$（前/后、肘上/肘下）→ $R_{36}$ 欧拉解姿态关节；过滤限位后按最小关节位移选解。微秒级，换构型要重推。

### 数值雅可比 IK

$v = J(q)\dot q$；几何列：转动关节 $[\omega_i\times(p_e-p_i);\ \omega_i]$。迭代：位姿 6D 误差 → 伪逆或 DLS → 线搜索/限位。奇异附近用自适应 $\lambda$（最小奇异值小时加大阻尼）。

### 奇异与操纵度

$\mathrm{rank}(J)$ 下降则某方向瞬时不可动。Yoshikawa $w=\sqrt{\det(JJ^\top)}=\prod\sigma_i$，路径规划可把 $w$ 当代价绕开奇异。工业常见：球腕 4/6 轴近同轴还走直线。

### 冗余零空间

$\dot q = J^+ \dot x + (I-J^+J)z$；次级 $z$ 可最大化操纵度、关节居中、避障距离梯度。

### 学习型

IKFlow：条件 Normalizing Flow 从 $(z,T^\star)$ 并行出 K 组 $q$，过滤 FK 误差后择优。文内指灵巧手/全身 IK；与扩散策略的「多候选再筛选」同构。

## 对 wiki 的映射

- 升格 [`wiki/formalizations/inverse-kinematics.md`](../../wiki/formalizations/inverse-kinematics.md)。
- 交叉：[`wiki/formalizations/forward-kinematics.md`](../../wiki/formalizations/forward-kinematics.md)、[`wiki/formalizations/robot-jacobian.md`](../../wiki/formalizations/robot-jacobian.md)、[`wiki/entities/mink-ik.md`](../../wiki/entities/mink-ik.md)、[`wiki/entities/pink-ik.md`](../../wiki/entities/pink-ik.md)、[`wiki/entities/ssik.md`](../../wiki/entities/ssik.md)、[`wiki/concepts/tsid.md`](../../wiki/concepts/tsid.md)。
