---
title: 机器人逆运动学（IK）到底在算什么？五个关键点从数学本质到工程落地
author: 深蓝具身智能
date: "2026-07-23 10:56:00"
source: "https://mp.weixin.qq.com/s?__biz=MzkwMDcyNDUzMQ==&mid=2247506764&idx=1&sn=45d102a22d570c268435bca714a5e088"
---

# 机器人逆运动学（IK）到底在算什么？五个关键点从数学本质到工程落地

![Image](https://mmbiz.qpic.cn/sz_mmbiz_gif/kaugqJpv9nuCktylvYoMKHYNAVojoRUpfyf1py08JvUnkfPXArzj4t5bMiaS6RBCXHHGhf8xlyw8icHrJcjEyYoA/640?wx_fmt=gif&from=appmsg#imgIndex=0)

![Image](https://mmbiz.qpic.cn/sz_mmbiz_gif/uwFbeBKoFGdoQIjfWURy7c7azJGMnuUQibr6u1pxdISwVl0rCCmib0a6yh3JIDkbcXCVdYLPAXUzP6c1l33TdkJY8MMKyyf99RZVFUpmGyWn8/640?wx_fmt=gif&from=appmsg#imgIndex=1)

一个"反着问"的问题，为何如此难？

> 大家好，这里是【深蓝具身智能】。
>
> 本文出自《具身智能基础》专栏，是本栏目下的第九篇文章，聚焦于逆运动学。
>
> 全文 5000 余字，建议收藏阅读。

---

[💙](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653&token=1419275101&lang=zh_CN#wechat_redirect)[订阅《具身智能基础》专栏](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653&token=1419275101&lang=zh_CN#wechat_redirect)

你的订阅和收藏，将支持我们把这件事持续做下去✨

问：机械臂需要把一颗螺丝精确拧入一个斜面上的孔位，目标位置和姿态已知，每个关节该转多少度？

这个问题，从直觉上看似乎只是"倒着算一遍"，但真正推导下来就会发现它完全不是这么回事。

- 正向的计算（已知关节角→求末端位姿）是一条单行道：

给定输入，通过确定性的矩阵连乘，输出唯一结果。整个过程没有歧义，数值稳定，计算量可控。

- 逆向的问题（已知末端位姿→求关节角）则像是在迷宫里找出口：

出口可能有一个、八个、无数个，甚至根本不存在。更麻烦的是，就算出口存在，找到它的路径本身就是一个非线性的高维搜索问题，在某些特殊位置，搜索的"地图"会直接失效。

这就是逆运动学（Inverse Kinematics，IK）：机器人控制栈里那个承受着数学压力、工程约束和实时性要求三重夹击的核心模块。

本文尝试从数学本质出发，走完解析解→数值解→现代方法→工程落地这条完整的链路，并在关键节点给出可执行的伪代码和核心公式。

**我们开设此账号，除了想要向各位对【具身智能】感兴趣的人传递前沿权威的知识讯息外，也想和大家一起见证它到底是泡沫还是又一场热浪？****欢迎关注****【深蓝具身智能】**👇

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4YWHX9iazKkdkgh363zh9GFAfZia4RWWoYhutUeS8g43MnicLMfe9kUAZg/640?wx_fmt=jpeg&from=appmsg#imgIndex=2)

位姿的数学空间：为什么 IK 天然就难

### 末端位姿活在 SE(3) 上

机械臂末端的位姿（Pose）由位置和姿态两部分组成，数学上它是特殊欧氏群 SE(3) 的一个元素：

字母含义：

- ：齐次变换矩阵，同时编码旋转和平移
- ： 旋转矩阵，满足
- ：末端位置向量
- ： 零向量，维持齐次坐标结构

SE(3) 是一个李群，不是平坦的欧几里得空间。（在专栏的第一篇文章中有详细介绍）

这意味着你不能直接对两个位姿做线性插值——两个旋转矩阵的平均不是旋转矩阵。

这个事实在后续的误差计算和轨迹插值中会反复造成麻烦。

### 正运动学：确定性的链式计算

正运动学（Forward Kinematics, FK） 通过 DH 参数（Denavit-Hartenberg Convention）将 n 个关节的变换依次连乘：

字母含义：

- ：末端执行器在基坐标系中的位姿
- ：第  个连杆相对于第  个连杆的变换，是关节角  的函数
- ：第  个关节的角度（转动关节）或位移（移动关节）

FK 是一个从关节空间  到位姿空间  的光滑映射 ，有唯一输出，计算量是  的矩阵乘法。

## 逆运动学：反过来问，问题变质

IK 的目标是找到关节角向量 ，使得正运动学的输出等于目标位姿：

字母含义：

- ：关节角向量（配置空间中的一个点）
- ：满足关节限位的可行域
- ：目标位姿
- ： 上的"差"运算（通常分解为位置误差 + 旋转轴角误差）
- ：正运动学映射

难点在于：

- f 是非线性的（含大量三角函数）
- 解可能不存在（目标在工作空间外）
- 解可能不唯一（6DOF 球腕机器人最多有 16 组解析解）
- 解可能无穷多（冗余机器人，n > 6）
- 在奇异点附近，问题的数值条件急剧恶化

![Image](https://mmbiz.qpic.cn/mmbiz_gif/uwFbeBKoFGeybyhV2dcOvQSYURTNEiaks20AyxJSDQH2Ezbf1ob3yMqm1icx4FodxDAtHUx8ED28LIxlofdmGoKQuKiay5D3vCkUq8hO8ia4Ogc/640?wx_fmt=gif&from=appmsg#imgIndex=3)![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nsAicIiaQwb1eFDMZwlNcXLBibqgVaodXH45G6Pdbk9xSEsUtlicqgxKkAiaK0P8QzGwuLiatibYiaIagQoOg/640?wx_fmt=jpeg&from=appmsg#imgIndex=4)

## 解析解（Analytical IK）：能算就别迭代

### DH 参数与连杆变换

满足 Pieper 条件（后三轴共点）的机器人存在封闭形式解析解。每个连杆的 DH 变换矩阵为：

字母含义（DH 四参数）：

- ：关节角，绕轴的旋转量（唯一变量，其余为常量）
- ：连杆扭角（Link Twist），绕轴，与的夹角
- ：连杆长度（Link Length），沿方向，与的公垂线长度
- ：连杆偏距（Link Offset），沿方向的偏移
- ，以此类推

### 解析 IK 的伪代码骨架（以6DOF球腕机器人为例）


```apache
function analytical_IK(T_target, dh_params) -> list[q]:
    solutions = []

    # Step 1: 解腕心位置（Wrist Center Position）
    # 利用球腕结构，末端姿态决定腕心在哪里
    p_wc = T_target[:3, 3] - d6 * T_target[:3, 2]
    # p_wc: 腕心在基坐标系中的位置
    # d6:   第6连杆偏距（末端法兰到腕心距离）
    # T_target[:3,2]: 目标姿态的z轴方向（末端接近方向）

    # Step 2: 由腕心位置解 θ1, θ2, θ3（位置关节）
    theta1_candidates = solve_theta1(p_wc)          # 通常有2解（前/后）
    for theta1 in theta1_candidates:
        theta3_candidates = solve_theta3(p_wc, theta1)  # 肘上/肘下，2解
        for theta3 in theta3_candidates:
            theta2 = solve_theta2(p_wc, theta1, theta3)

            # Step 3: 构造前三轴的变换 T_03
            T_03 = fk(dh_params[:3], [theta1, theta2, theta3])

            # Step 4: 由目标姿态与 T_03 之差，解 θ4, θ5, θ6（姿态关节）
            R_36 = T_03[:3,:3].T @ T_target[:3,:3]
            theta4, theta5, theta6 = euler_ZYZ_from_R(R_36)
            # R_36: 第3到第6坐标系间的旋转，用欧拉角分解

            solutions.append([theta1, theta2, theta3,
                               theta4, theta5, theta6])

    # Step 5: 过滤超出关节限位的解，从合法解中选最优
    valid = filter_joint_limits(solutions, q_min, q_max)
    return select_best(valid, q_current)  # 最小关节位移原则
```


解析解的本质：通过代数变换把 矩阵方程降维拆解成一系列一元三角方程，每步用 atan2 消元。

速度是微秒级，但每换一种机器人构型就得重推一遍。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nsAicIiaQwb1eFDMZwlNcXLBibTvia4qLjYyoM2Do58jX9J71HickLLA3NxCQp6fPljkgY26WIeaoeeYVQ/640?wx_fmt=jpeg&from=appmsg#imgIndex=5)

## 数值解（Numerical IK）：雅可比是核心武器

### 雅可比矩阵：关节速度到末端速度的线性化

雅可比矩阵  是正运动学映射  对  的一阶偏导，描述了当前构型下关节微小运动如何映射到末端速度：

字母含义：

- ：末端速度旋量（前 3 维线速度，后 3 维角速度）
- ：各关节的角速度向量
- ：依赖当前关节角 ，每一步迭代都要重新计算

对于第  个转动关节， 的第  列（几何雅可比例）为：

字母含义：

- ：第  坐标系的  轴在基坐标系中的单位方向向量（即旋转轴方向）
- ：末端位置（基坐标系）
- ：第  关节原点位置（基坐标系）
- ：叉积，生成关节旋转对末端线速度的贡献

关节  转一点，末端的线速度贡献旋转轴  从轴到末端的力臂；角速度贡献旋转轴本身。

### 基于雅可比的迭代 IK 流程


```cs
function jacobian_IK(T_target, q_init, max_iter=100, tol=1e-5):
    q = q_init                          # 初始关节角（热启动关键）

    for k in range(max_iter):
        T_cur = fk(q)                   # 正运动学，得当前末端位姿

        # 计算6D误差向量（位置误差 + 旋转误差）
        e_pos = T_target[:3,3] - T_cur[:3,3]              # 3D位置误差
        R_err = T_target[:3,:3] @ T_cur[:3,:3].T          # 旋转误差矩阵
        e_rot = so3_to_axis_angle(R_err)                   # 转为轴角向量
        e = concatenate([e_pos, e_rot])  # 6维误差旋量

        if norm(e) < tol:
            return q                    # 收敛

        J = compute_jacobian(q)         # 计算当前构型的雅可比

        # ---- 选择求解器 ----
        # 方式A：伪逆（冗余机器人，n>6）
        dq = J.pinv() @ e

        # 方式B：DLS（奇异点附近更稳定）
        lambda_sq = adaptive_lambda(J)   # 根据最小奇异值自适应
        dq = J.T @ inv(J @ J.T + lambda_sq * I) @ e

        # 更新关节角（带步长 α 防止过冲）
        alpha = line_search(q, dq, T_target)   # 可选：Armijo线搜索
        q = q + alpha * dq
        q = clip(q, q_min, q_max)       # 强制关节限位

    return None  # 未收敛
```


### 阻尼最小二乘（DLS）：奇异点的“救命稻草”

纯伪逆在奇异点附近的问题： 的最小特征值趋向 ，逆矩阵元素爆炸， 变得无穷大。

DLS 的修复方案：

字母含义：

- ：本次迭代的关节角修正量
- ：当前末端位姿与目标之间的误差旋量
- ：阻尼系数，控制数值稳定性与精度的权衡
- ：单位矩阵，加到对角线上，保证正定可逆

自适应 （） ：


```apache
def adaptive_lambda(J, eps=0.01, lambda_max=0.1):
    sigma_min = min_singular_value(J)   # SVD最小奇异值
    if sigma_min >= eps:
        return 0.0                      # 远离奇异点，不加阻尼
    else:
        # 奇异点附近，平滑增大阻尼
        ratio = sigma_min / eps
        return lambda_max * (1 - ratio**2)
```


![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4yZibwaOWpB3DrcxuiafpXicx2ibHiaHAZFr7ptU6ud2hsxgCXvV0JGHtTDw/640?wx_fmt=jpeg&from=appmsg#imgIndex=6)

## 奇异性：IK 的死穴

### 三类奇异构型

当  时机器人处于奇异构型，末端在某些方向上瞬间丧失运动能力。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGdGrWlxxFGpibZKk1zIdkHmB5S4CIcAm6zxXfDjqQFic85RMJ4RSWBtlnlWo7BG6V2pPDVQqWX1DfK1M6n2GJZeIYS19gXqBwNBU/640?wx_fmt=png&from=appmsg#imgIndex=7)

### 操纵度指标

Yoshikawa（1985）提出用操纵度 w 量化当前构型的灵活性：

字母含义：

- ：操纵度指标， 时处于奇异点
- ：行列式，捕捉雅可比所有方向上的"体积"缩减程度
- ：操纵度椭球（Manipulability Ellipsoid）的形状矩阵


```makefile
U, sigma, Vt = svd(J)
# sigma: 各方向的放大因子，最小值趋0 = 即将奇异
# 奇异方向 = U中对应最小sigma的列
w = prod(sigma)   # 等价于 sqrt(det(J@J.T))
```


 越大，机器人在当前构型下对末端各方向的控制能力越均衡。

在路径规划阶段把  作为代价函数的一项，可以主动让轨迹绕开奇异区域。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_gif/uwFbeBKoFGc2VBuWuGydv92zxcXuicSJcfZHSxfsZ2F0s6gP7yDjDDXiaBiak5Kia6jribVxY9kGf0kAyoKoA5Dhzicibvw2IRRwkwB1PhbhdkGUEo/640?wx_fmt=gif&from=appmsg#imgIndex=8)▲图源网络 | 在实际工业应用中，最常见的奇异点发生场景是六轴球腕机器人 4 轴和 6 轴接近同轴且进行直线运动时![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4hKdH0P2rRHX1TlxUqlAx7X6m2hcl7XttttyRW05mhbEa1msX7zEzvw/640?wx_fmt=jpeg&from=appmsg#imgIndex=9)

## 冗余自由度：零空间是"免费的算力"

### 零空间的数学结构

当 （如7轴机械臂、人形机器人手臂、灵巧手），关节空间的维度高于任务空间，雅可比的零空间（）非平凡：

对任意向量，投影到零空间：

字母含义：

- ：雅可比的 Moore-Penrose 伪逆
- ： 阶单位矩阵
- ：零空间投影算子（Null Space Projector），将任意向量投影到零空间
- ：任意向量，用来编码次级任务的优化方向

关键性质：

即零空间运动对末端位姿没有任何影响。

### 任务优先级分解（Task Priority Framework）

完整的冗余机器人控制律为：

字母含义：

- ：主任务项——完成末端位姿跟踪（最小范数解）
- ：次级任务项——在不影响末端的前提下优化
- ：次级代价函数，例如：（最大化操纵度）

- （保持在关节中间位置）
- （避障）


```css
def redundant_ik_step(q, x_target, g_func):
    J = compute_jacobian(q)
    J_pinv = pinv(J)

    # 主任务：追踪末端位姿
    x_cur = fk(q)
    dx = compute_error(x_cur, x_target)    # 6D误差旋量
    dq_primary = J_pinv @ dx

    # 次级任务：在零空间中优化 g(q)
    grad_g = numerical_gradient(g_func, q)
    null_proj = I - J_pinv @ J
    dq_secondary = null_proj @ grad_g

    return dq_primary + dq_secondary
```


![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/kaugqJpv9nutPusx7ngVOmag61DHUJmX7OGyOG8gzibCyLX91kbhEcWl0mnLk5Zb5uVRabIn51LEKNicYT8OlZZQ/640?wx_fmt=png&from=appmsg#imgIndex=10)

## 场景痛点 × 解决方案 × 关键代码

### 场景 A：高速工业分拣（实时性第一）

痛点：传送带节拍 ≤ 0.5s，IK 必须在 ≤ 1ms 内给出关节角，还要从多解中选无碰撞最优解。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGduk5kuP8NazCYCSeRgm95I22YJUCczE53UVHuLQ9BOpIHGbQ28micVkqrEhaia2dvLricibo0WHZldymia7tDA7VuB1GLLQ7Ljj938/640?wx_fmt=png&from=appmsg#imgIndex=11)

解决方案：针对固定构型（满足 Pieper 条件）预推解析解，离线生成闭式代码；运行时纯查表+atan2计算，无迭代。

关键代码（IKFAST 风格输出片段）：


```apache
def ik_joint1(px, py, pz, d1, a2, a3, d4):
    # 由腕心(px,py,pz)解第1关节
    # px,py: 腕心在基坐标xy平面的投影
    # d1: 机座到第1关节的高度偏置
    r = sqrt(px**2 + py**2)     # 腕心在水平面的半径
    theta1_up   = atan2(py, px)
    theta1_down = atan2(-py, -px)   # 机器人"翻转"构型
    return theta1_up, theta1_down

def ik_joint3(r, pz, d1, a2, a3, d4):
    # 用余弦定理求肘关节角
    # r: 腕心水平距离, pz: 腕心高度
    D = (r**2 + (pz-d1)**2 - a2**2 - a3**2) / (2*a2*a3)
    # D 即 cos(θ3)，|D|>1 则目标超出工作空间
    if abs(D) > 1.0:
        raise WorkspaceError("Target unreachable")
    theta3_elbow_up   = atan2(+sqrt(1 - D**2), D)
    theta3_elbow_down = atan2(-sqrt(1 - D**2), D)
    return theta3_elbow_up, theta3_elbow_down
```


### 场景 B：7轴冗余臂避障（灵活性第一）

痛点：末端路径固定（如沿缝焊接），但中间连杆必须绕开一个动态障碍物。额外的第7个自由度是解决冗余的钥匙，但如何使用它是个优化问题。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGdT4Xl7cErlbVqBQRAibwxN23CjCbgDy1k8HckNzAfN02YzPItALR6mhWGtyrbJwBKVSphqhnOS3ghHV7s1eETtw8xojNAG29CQ/640?wx_fmt=png&from=appmsg#imgIndex=12)

解决方案：零空间投影，次级任务 = 最大化机器人与障碍物的最短距离。

关键公式：

- ：机器人与障碍物的最短距离，关于  的函数
- ：次级任务增益，调节避障激进程度
- ：距离对关节角的梯度（可用 /bullet 的距离查询 + 数值差分计算）


```apache
def obstacle_avoidance_secondary(q, obstacle_mesh, k0=0.5):
    # 计算当前构型到障碍物的最短距离梯度
    dq = 1e-5
    grad = zeros(n)
    for i in range(n):
        q_plus = q.copy(); q_plus[i] += dq
        q_minus = q.copy(); q_minus[i] -= dq
        d_plus  = min_distance_to_obstacle(fk_all_links(q_plus),  obstacle_mesh)
        d_minus = min_distance_to_obstacle(fk_all_links(q_minus), obstacle_mesh)
        grad[i] = (d_plus - d_minus) / (2 * dq)
    return k0 * grad   # 作为 z 向量传入零空间投影
```


### 场景 C：灵巧手高自由度 IK（维度爆炸）

痛点：五指灵巧手通常有 20+ 自由度，传统数值 IK 在高维关节空间收敛慢，多解问题更加严重。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGesZEdWyp8wpI7Ef8zMapLMMAgTwicIicbg6LLch6yftiaNiaKdflSxGqUkrfFqfW8ASjTHPMJdCRiciaZawHjS960SgK56ibNmY98tS0/640?wx_fmt=png&from=appmsg#imgIndex=13)

解决方案：学习型 IK——用 Normalizing Flow 建模的分布，推理时采样多组解。

IKFlow 核心思路（IKFlow: Generating Diverse Inverse Kinematics Solutions, 2022）：


```bash
训练阶段：
  随机采样大量合法关节角 q ~ U(q_min, q_max)
  计算对应的 T = fk(q)
  用 (q, T) 训练 Normalizing Flow 网络 F：
    F: (z ~ N(0,I), T_target) -> q
    目标：F(F^{-1}(q | T)) = q（可逆映射，精确似然）

推理阶段：
  输入 T_target
  从标准正态 z_1, z_2, ..., z_K 采样 K 个噪声
  q_i = F(z_i, T_target)    # 并行生成 K 个候选解
  过滤 fk(q_i) 距 T_target 误差 > ε 的解
  从合法解中选关节位移最小的
```


- 推理速度：单次前向传播，K=100 个解在 GPU 上约 5ms
- 适用场景：灵巧手抓取规划、人形机器人全身 IK

当灵巧手开展拧瓶盖、USB 插拔等精细装配操作时，受传感器噪声、接触不确定性影响，仅依靠单一逆运动学解鲁棒性较差，需要解集候选池支撑上层重决策。

IKFlow 能够采样多样化逆运动学候选解，天然匹配 “批量生成候选 + 约束择优” 的分层规划范式；

在建模思想层面，该生成式思路与 Diffusion Policy 等基于扩散模型的动作生成框架具备良好兼容性，可构建 “笛卡尔动作规划→多候选 IK 求解→关节轨迹筛选” 的分层控制链路。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/kaugqJpv9nutPusx7ngVOmag61DHUJmX5ne3MfNYQBbic4xIYsEJDKpCRqQXk6gllicSqc7QiabhaIEuCXA1I4xsg/640?wx_fmt=png&from=appmsg#imgIndex=14)

## 工程落地的"脏活"清单

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGdYjqKwtHmCzJ9DbJO1yPKO0IMjwPia5VaRVXxtRcjGctBNicIPrSGZws4WbKpVz0BGN3LOwmI1LUgiaoP9RB0yHK22jL32K0QYBE/640?wx_fmt=png&from=appmsg#imgIndex=15)![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/kaugqJpv9nticBjjcwNOxbPYPicJibngQA96VicG0IjoaMzjTF8ib6fnkhfYvbBCNibwd1oVr5DupoFmnicicw6WgRXczw/640?wx_fmt=png&from=appmsg#imgIndex=16)

## 写在最后

正运动学是一道函数题；逆运动学是一道反函数题，而且这个反函数是多值的、不连续的，在某些点上根本不存在。

从 DH 参数到 SE(3) 李群，从解析剥离到雅可比迭代，从阻尼最小二乘到零空间投影，再到 Normalizing Flow——IK 的每一层解法，都是在用不同的数学工具逼近同一件事：

在物理约束和实时性压力下，找到那个让机器人"够到"目标的关节配置。

具身智能时代让 IK 的重要性再次上升：当机器人需要在非结构化环境中完成开放任务，IK 不再只是一个控制模块，而是连接感知、规划和执行的核心接口。

解的质量、速度和鲁棒性，直接决定了整个系统的上限。

正向/逆向运动学、轨迹生成与运动规划等核心模块，覆盖了本专栏正在连载的完整具身智能基础知识体系⬇️

编辑｜小小怪博士

审编｜具身君

 ****推荐阅读**
[![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGcibJS8986MfCcVATGOkcK6lNQfiaTORbuhSFoATTmZ5kA6nV8l8REia7nm4A4OxC1yOePBqrzWHQQd0ALicYANgOoNmRbibChjcAuQ/640?wx_fmt=png&from=appmsg#imgIndex=17)](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=3824573915845640194&scene=126#wechat_redirect)[![Image](https://mmbiz.qpic.cn/mmbiz_jpg/uwFbeBKoFGcRfEtsGjVkl7cXB7QYAAib4wOMhdRcvsQicHnmiaxqoibw9LUCGGcPGSYnUPeUlZEoiaBlQezclFhp5yZQ6yLcLAjYeI67pJmvhOMw/640?wx_fmt=jpeg&from=appmsg#imgIndex=18)](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653&token=944555238&lang=zh_CN#wechat_redirect)

**![Image](https://mmbiz.qpic.cn/mmbiz_png/qKE443uRvLo6ic3ZPUttmFZ2AefQ4wjHSlQluSDkaxL9icWicpPYYmpo1Wa37Scjhh4AS5VwYJtmlTf5cKMiaIXg5g/640?&random=0.17349735674179656&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&wx_fmt=other#imgIndex=19)**

**【深蓝具身智能】****的原创内容均由作者团队倾注个人心血制作而成，希望各位遵守原创规则珍惜作者们的劳动成果；未经授权禁止任何机构或个人抓取本账号内容，进行洗稿/训练，否则侵权必究⚠️⚠️**


![Image](https://mmbiz.qpic.cn/mmbiz_png/Nabxc8rdYriaKqxCUjcZ8sSCnSNlWpqdI1kyXXQjXbtv95xvACqQoqL2ibbKXt9PB0FLPibKiawGsTcQrnKDGWVw2Q/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1#imgIndex=20)

点击❤收藏并推荐本文**
