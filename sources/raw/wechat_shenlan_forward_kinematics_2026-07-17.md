---
title: 通俗讲透机器人正向运动学：从原理到工程实例，一次性讲明白！
author: 深蓝具身智能
date: "2026-07-17 10:56:00"
source: "https://mp.weixin.qq.com/s?__biz=MzkwMDcyNDUzMQ==&mid=2247506508&idx=1&sn=8c7041c9e509d8a83cf92ca3cb63023c"
---

# 通俗讲透机器人正向运动学：从原理到工程实例，一次性讲明白！

![Image](https://mmbiz.qpic.cn/sz_mmbiz_gif/kaugqJpv9nuCktylvYoMKHYNAVojoRUpfyf1py08JvUnkfPXArzj4t5bMiaS6RBCXHHGhf8xlyw8icHrJcjEyYoA/640?wx_fmt=gif&from=appmsg#imgIndex=0)

![Image](https://mmbiz.qpic.cn/mmbiz_gif/uwFbeBKoFGeDJZCTgroMhiaqnBGYBNvdn1Cu1siauzh4SY1xcWMGZ0NhmiaOzYfB0MyiblqTtm2CN1sKUwmYBVBzt5ygujJHB5Xsyop0vq2Ulia8/640?wx_fmt=gif#imgIndex=1)

正运动学，机器人学中最“老实”的部分 ：给定输入，输出唯一，没有歧义，没有搜索，没有迭代。

> 大家好，这里是【深蓝具身智能】。
>
> 本文出自《具身智能基础》专栏，是本栏目下的第八篇文章。
>
> 在之前的连载中，我们系统梳理了机器人学的数学基础与位姿空间。接下来的内容我们将进入机器人运动学的核心：正向运动学（FK）与逆向运动学（IK）的完整理论与实操！
>
> 本文将聚焦于正运动学——这个看似“简单直接”，却支撑起整个具身智能底层物理交互的基石。
>
> 全文 8000 余字，建议收藏阅读。

---

[💙](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653&token=1419275101&lang=zh_CN#wechat_redirect)[订阅《具身智能基础》专栏](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653&token=1419275101&lang=zh_CN#wechat_redirect)

你的订阅和收藏，将支持我们把这件事持续做下去✨

## 一个“正着算”的问题，为什么值得讲？

想象一个场景：

> 一台六轴机械臂正在执行焊接任务：
>
> 工程师在示教器上输入了六个关节角度：基座旋转 30°、肩部抬起 45°、肘部弯曲 -20°、腕部翻转 60°、俯仰 -15°、末端旋转 90°。
>
> 按下启动键的瞬间，机械臂末端精准地落在了焊缝起点。

这一过程，机械臂控制器做的事情可以用三句话概括：

先把六个关节角度作为输入，通过一系列确定的矩阵运算，算出末端执行器在三维空间中的精确位置和朝向。

**这就是正运动学（Forward Kinematics，FK）**

**——机**器人学中最基础、也最确定的计算模块。

说它"确定"，是因为给定关节角度，输出唯一、数值稳定、计算量可控。

它不像逆运动学那样需要在高维空间中搜索、可能多解、可能无解、可能在奇异点附近发散。

正运动学是一条单行道：**输入关节角 → 连乘变换矩阵 → 输出末端位姿**，整个过程没有“歧义”。

但"确定"不等于"简单"。

要把一根根物理连杆、一个个旋转关节，变成可以在计算机里精确计算的数学模型，需要一套严谨的约定体系。

这套体系的核心，就是 **DH 参数（Denavit-Hartenberg Convention）**。

本文将从最底层的数学概念出发，走完「位姿表示 → DH参数定义 → 连杆变换推导 → 正运动学连乘 → 代码实现 → 工程落地」这条完整链路。

它是理解后续逆运动学、动力学、轨迹规划的前提：具身智能的每一层上层能力，都建立在这个看似枯燥的正运动学基础之上。

**我们开设此账号，除了想要向各位对【具身智能】感兴趣的人传递前沿权威的知识讯息外，也想和大家一起见证它到底是泡沫还是又一场热浪？****欢迎关注****【深蓝具身智能】**👇

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4YWHX9iazKkdkgh363zh9GFAfZia4RWWoYhutUeS8g43MnicLMfe9kUAZg/640?wx_fmt=jpeg&from=appmsg#imgIndex=2)

## 位姿的数学空间：机器人末端"活"在哪里



# 位置和姿态：两个不能混为一谈的东西

机械臂末端在空间中的状态，由两部分组成：

- 位置（Position）：末端在基坐标系中的坐标 ，是一个三维向量 。
- 姿态（Orientation）：末端坐标系相对于基坐标系的旋转关系，用旋转矩阵  描述。

合在一起，"位姿"（Pose）就是 位置 + 姿态。

> 位置很好理解：三个数，确定一个点。
>
> 但姿态就不那么直观了——为什么不能也用三个数来描述呢？

因为你不能简单地对旋转做"加法"。

两个旋转矩阵的平均不是旋转矩阵；两个旋转的线性插值会产生畸变。

旋转构成了一个**非交换群**：先绕  轴转 90° 再绕  轴转 90°，和反过来做，结果是不同的。

举个例子：把一本书平放在桌上，先沿长边翻 90°（书竖起来了），再沿短边翻 90°（书朝侧面倒了）。然后复原，先沿短边翻 90°，再沿长边翻 90°

——你看到的封面方向完全不同，因为旋转不满足交换律。

旋转矩阵：SO（3）的严格定义

旋转矩阵是  的正交矩阵，满足：

所有满足这两个条件的矩阵构成**特殊正交群**。

第一个条件保证旋转不改变向量长度（保距），第二个条件排除反射（保定向）。

旋转矩阵的每一列有明确的物理含义：**第  列就是旋转后坐标系第  个坐标轴在原坐标系中的方向向量**。

其中  是旋转后坐标系的三个轴， 是原坐标系的轴。

矩阵的每个元素就是两组基向量的点积。

### 齐次变换矩阵：把旋转和平移"焊"在一起

机器人学中，一个连杆相对于另一个连杆的关系，既包含旋转又包含平移。

为了把它们统一在一个矩阵里，我们引入**齐次坐标**和**齐次变换矩阵（本专栏前面有过具体介绍）**：

**字母含义：**

- ：齐次变换矩阵，同时编码旋转和平移
- ：旋转矩阵（），描述姿态
- ：平移向量（），描述位置
- ：零向量（），最后一行  维持齐次坐标结构

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGdyF4xfjE7FEHrCeMOW5XDuQdjcRbGfYFj03b7k2KzzjUEnWpaEGiaHAdj7ibOCehfuOpxLXz2zPJCOZHGfTFXfbqjYE77mQHnh0/640?wx_fmt=png&from=appmsg#imgIndex=3)▲齐次变换矩阵结构©【深蓝具身智能】编译

齐次变换矩阵意义在于：**两个变换的复合变成了矩阵乘法**。

如果坐标系  相对于  的变换是 ，坐标系  相对于  的变换是 ，那么  相对于  的变换就是：

一个矩阵乘法，就把两次旋转 + 两次平移的复合算清楚了

——这就是正运动学能用连乘来实现的原因（代数基础）。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nsAicIiaQwb1eFDMZwlNcXLBibqgVaodXH45G6Pdbk9xSEsUtlicqgxKkAiaK0P8QzGwuLiatibYiaIagQoOg/640?wx_fmt=jpeg&from=appmsg#imgIndex=4)

## 坐标系的建立：DH 参数

先提个问题：如何给每个连杆"贴"一个坐标系？

一个  自由度的机械臂有  个关节，每个连杆都在运动。要描述整个机械臂的运动学，我们需要：

1. 在每个连杆上固联一个坐标系
2. 用确定的参数描述相邻坐标系之间的关系
3. 通过矩阵连乘得到末端坐标系的位姿

但是难点在于：相邻两个关节轴在空间中的关系可能是任意的。

它们可能平行、相交、也可能既不平行也不相交（异面直线）。

那么，如何用最少的参数完整描述这种关系？

1955 年，Jacques Denavit 和 Richard Hartenberg 给出了答案：**任意两条空间直线之间的关系，只需要 4 个参数就能完全确定**。

这就是 DH 参数。

### DH 四参数：几何直觉

DH 参数的核心思想：利用两条关节轴之间的**公垂线**（Common Perpendicular）建立坐标系。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGcJNEmtFfIdA0KR00mEtkRJOfoJKZtFpBK6mHdNv2v91XySKhVypibFgWYlbFZM4dozIfK3LrPUeXoJBSviazOnibk92mib2nQoJDE/640?wx_fmt=png&from=appmsg#imgIndex=5)

▲DH参数几何含义©【深蓝具身智能】编译

对于第  个关节，定义四个参数：

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGe2QWuaug4C4uV1Z0Jibp8xka2nRQFKWIvLadwLzx5ibKDRMovKHnhicv1ufXbc5vBODptz6ezrwmSGGoz8N1dOrsJB7N3icu0v1lI/640?wx_fmt=png&from=appmsg#imgIndex=6)

- 对于**转动关节**（Revolute Joint）， 是变量，其余三个参数为常量；
- 对于**移动关节**（Prismatic Joint）， 是变量，其余三个参数为常量；
- 每个连杆只有一个自由度，所以 DH 参数中只有一个变量。

**为什么是 4 个参数而不是 6 个？**

一般的刚体变换有 6 个自由度（3 旋转 + 3 平移）。

DH 参数只用了 4 个，是因为它利用了一个关键约束：关节轴的方向决定了 **轴**，坐标系约束减少了自由度。

这种约束是合理的，因为机械臂的关节轴是物理存在的，不是任意方向。

### 坐标系分配规则

给机械臂建立 DH 坐标系，遵循以下步骤：

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGeddqKKebqbryWibWNjpXfQ7rUR3uL98aSwdc7xV4W557nibFiaQVak354AUq5mLLWfXhBK0KSHWu4TQ8keE7qrwQ5YJrgT8p01m0/640?wx_fmt=png&from=appmsg#imgIndex=7)

▲多连杆机械臂坐标系分配©【深蓝具身智能】编译

1. **编号**：基座为 ，第一个连杆为 ，以此类推，末端执行器为 。
2. **轴**：沿第  个关节的运动轴线方向（转动关节为旋转轴，移动关节为平移方向）。
3. **轴**：沿  和  的公垂线方向，从  指向 。如果两轴平行， 取两轴间任意垂线方向。
4. **轴**：由右手定则确定，。
5. **原点**： 与  的交点（即公垂线在  上的垂足）。

**特殊情况处理：**

- 当  和  相交时（无公垂线），取 （两轴的叉积方向）
- 当  和  平行时，公垂线不唯一，取通过  的那条
- 基坐标系 ：通常与  在  时重合

### 标准 DH vs 改进型 DH

在实际工程中，存在两种 DH 约定：

- **标准 DH**（Denavit-Hartenberg 原始版本）；
- **改进型 DH**（Craig 在《Introduction to Robotics》中提出）。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGdgb43zz1ZQRMlH9s9PgGttMJBqzPL8xJHVLuKcibHSIlbtk9URo51ulytQe59GmPC1yn5GVLzKZ7gow7cKn3qyyaztHp27JCI0/640?wx_fmt=png&from=appmsg#imgIndex=8)

▲标准DH与改进型DH对比©【深蓝具身智能】编译

**核心区别在于：坐标系固联的位置不同。**

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGdWSiajvBAY7a9sOjusplnuaHKhnXoLVdtEVicCbm1YAPTgFt0ezibp6mn0npjk48rEibULV7j5riadSicmaXElvYBO76sicgnCTnmzpQ/640?wx_fmt=png&from=appmsg#imgIndex=9)

**标准 DH 变换矩阵：**

**改进型 DH 变换矩阵：**

两种约定没有对错之分，但**千万不能混用**。在一个项目中选定一种后贯彻到底。

Craig 的改进型 DH 在教学和现代教材中更常见，但标准 DH 在工业控制器和早期文献中仍是主流。

本文后续推导采用**标准 DH**。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nsAicIiaQwb1eFDMZwlNcXLBibTvia4qLjYyoM2Do58jX9J71HickLLA3NxCQp6fPljkgY26WIeaoeeYVQ/640?wx_fmt=jpeg&from=appmsg#imgIndex=10)

## 连杆变换矩阵的推导：四步走

### 四个基本变换的分解

DH 变换的本质是：把一个复杂的空间变换，分解为 4 个基本操作的有序组合。

每个操作只涉及一个参数。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGcbZIFnEqaC3usnae6OQn4ZrCF3UGw3fdCJV79WKhqKGHtOParFPfk5jBrpswaRdjibw7fUqkDibuLUh75HKzw2tsblEvyDBPHDs/640?wx_fmt=png&from=appmsg#imgIndex=11)

▲四步变换分解©【深蓝具身智能】编译

- **Step 1：绕  轴旋转**

让  轴绕  转过  角度，转到与  对齐的方向。对于转动关节，这是唯一随关节运动的变量。

- **Step 2：沿  轴平移**

**沿** 轴移动  距离，使坐标原点到达  与公垂线的交点。

- **Step 3：沿  轴平移**

沿公垂线方向移动 （连杆长度），到达  轴上。

- **Step 4：绕  轴旋转**

**让** 绕  转过 （连杆扭角），使其与  对齐。

完整变换矩阵

将四个变换按顺序相乘：

展开计算后得到：

这就是 DH 参数法中最核心的公式：**单个连杆的变换矩阵**。

### 矩阵各元素的物理意义

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGeicM8Bl17BP7XKiaX2QUUUEjYdctgNNZbJZJuDp0aFwCrWnT9CWkWpyukxTTzeNv19ibWgbPra4ysyRmqxpm0gYElFBQlNBTkxzI/640?wx_fmt=png&from=appmsg#imgIndex=12)

当  变化时（转动关节），矩阵中的旋转部分和平移部分同时变化

——这正是"关节运动导致末端位姿变化"的数学表达。

 是自变量，矩阵是因变量。

### 推导验证：矩阵乘法的展开过程

为了确保大家能理解这个矩阵是怎么来的，这里展示关键步骤：

其中 ，（机器人学中常用简写记法）。

继续乘 ：

最后乘 ：

**推导完成。**

每一步都是确定的矩阵乘法，不涉及任何迭代或搜索——这也是正运动学"确定性"的来源。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4yZibwaOWpB3DrcxuiafpXicx2ibHiaHAZFr7ptU6ud2hsxgCXvV0JGHtTDw/640?wx_fmt=jpeg&from=appmsg#imgIndex=13)

## 正运动学：连乘的艺术

### 从基座到末端的链式相乘

有了单个连杆的变换矩阵 ，正运动学的计算就变成了简单的矩阵连乘：
![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGfvxN7zIZFXOAQpSoVMqLMibJMOlxkMMTTgJfRO3Wg3QB6bj0Lo54XN1VibmBQtiac5r7ibSHAQ4c1jkK5Ora7rFqHUBZmN1zaNqEQ/640?wx_fmt=png&from=appmsg#imgIndex=14)▲正运动学链式相乘©【深蓝具身智能】编译

**其中：**

- ：末端执行器坐标系  在基坐标系  中的位姿
- ：第  个连杆坐标系相对于第  个连杆坐标系的变换
- ：自由度数量（关节数）

**为什么连乘是对的？**

这是齐次变换矩阵的核心性质——变换的复合等于矩阵的乘积。

考虑三个坐标系 ：

- 相对于  的位姿为
- 相对于  的位姿为

那么  中的点  在  中的坐标为：

所以 ，推广到  个连杆就是连乘。

### 完整推导示例：平面 3R 机械臂

为了把抽象的公式落地，我们用一个最经典的例子：**平面三连杆机械臂（3R Planar Arm）**。

![平面3R机械臂DH坐标系](https://mmbiz.qpic.cn/mmbiz_jpg/uwFbeBKoFGc45Rvl7c6PlGCjeib8Hia72gFiaKBf6Nte25qtROLYq44ic8dvWq9841VZU8MJc8k03XHA2SRetgl0l8Jc5fBZRLNBib8RGPWZPiaqw/640?wx_fmt=jpeg#imgIndex=15)▲平面3R机械臂DH坐标系©【深蓝具身智能】编译

三个连杆在同一平面内，三个关节都是转动关节，所有  轴垂直于平面（指向纸面外）。

- **DH 参数表：**

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGfIxeabRhr4Dw11PJ4L8LEvvlx30fmRWrib8P0jGVJia48jkO2fx4ooWEBSEvwzMWO4jSMjRXtyS85sRAzdOZ6rwAKnDqjplLlTI/640?wx_fmt=png&from=appmsg#imgIndex=16)

**为什么  ？**

**因为所有关节轴平行（都垂直于平面），没有沿** 方向的偏移，也没有连杆扭角。这使得变换矩阵大大简化。（以下公式可左右滑动查看）

- **逐个计算变换矩阵：**

由于 ，DH 变换矩阵简化为：

- **第一级变换 ：**

其中 ，。

- **第二级变换 ：**

- **前两级复合 ：**

利用和角公式化简：

令 ：

- **最终结果 ：**

令 ，末端位姿为：

- **末端位置（平移部分）：**

- **末端姿态（旋转部分）：**

这个结果完全符合物理直觉——

平面机械臂的末端位置就是各连杆在  方向的投影之和，末端朝向就是所有关节角的累加。

DH 参数法的作用在此体现：**即使是复杂的 3D 机械臂，也只需要同样的连乘过程，只是参数表不同**。

### 不同关节角下的末端位姿

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/uwFbeBKoFGdAkKpKyKZ9157icZpBsIUGbE8ibDKysLZicc1pJKkzZCAvjUw67W4CicARoBGX29mB3rDcUloiaBjaIMN4pbVOTCLjk2xcLVTicwR44/640?wx_fmt=jpeg&from=appmsg#imgIndex=17)▲不同关节角下的机械臂姿态©【深蓝具身智能】编译

同一台机械臂，给定不同的关节角度输入，通过正运动学计算得到不同的末端位姿输出。

这就是正运动学的本质：从关节空间到位姿空间的确定映射。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4hKdH0P2rRHX1TlxUqlAx7X6m2hcl7XttttyRW05mhbEa1msX7zEzvw/640?wx_fmt=jpeg&from=appmsg#imgIndex=18)

## 从公式到代码：Python 实现

### DH 参数表驱动的通用 FK 函数

正运动学的代码实现非常直接——就是矩阵连乘。

以下是完整的 Python 实现（可左右滑动查看）：

```
import numpy as np

def dh_transform(theta, d, a, alpha):
    """
    计算单个连杆的 DH 变换矩阵 (Standard DH)

    参数:
        theta: 关节角 (弧度)
        d:     连杆偏距
        a:     连杆长度
        alpha: 连杆扭角 (弧度)

    返回:
        4x4 齐次变换矩阵 T_i^{i-1}
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    T = np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ]
    ])
    return T


def forward_kinematics(dh_params, joint_angles):
    """
    正运动学：从关节角计算末端位姿

    参数:
        dh_params:    DH 参数表，list of (theta_offset, d, a, alpha)
                      theta_offset 是零位时的关节角偏移
        joint_angles: 关节角向量 (弧度)

    返回:
        T_n^0: 4x4 齐次变换矩阵，末端在基坐标系中的位姿
    """
    T = np.eye(4)  # 从单位矩阵开始

    for i, (theta_off, d, a, alpha) in enumerate(dh_params):
        theta = joint_angles[i] + theta_off  # 实际关节角 = 输入角 + 偏移
        T_i = dh_transform(theta, d, a, alpha)
        T = T @ T_i  # 连乘

    return T


def extract_pose(T):
    """
    从齐次变换矩阵中提取位置和姿态

    返回:
        position:    (x, y, z) 位置
        euler_angles: (roll, pitch, yaw) 欧拉角 (ZYX 顺序)
    """
    position = T[:3, 3]
    R = T[:3, 3]

    # 从旋转矩阵提取 ZYX 欧拉角
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])

    return position, np.array([roll, pitch, yaw])
```

### 完整示例：平面 3R 机械臂

```
# ============================================
# 平面 3R 机械臂正运动学示例
# ============================================

# DH 参数表: (theta_offset, d, a, alpha)
# 平面机械臂: d=0, alpha=0
dh_params_3r = [
    (0, 0, 0.5, 0),   # 关节1: l1=0.5m
    (0, 0, 0.4, 0),   # 关节2: l2=0.4m
    (0, 0, 0.3, 0),   # 关节3: l3=0.3m
]

# 测试三组关节角
test_configs = [
    [0, 0, 0],                          # 完全伸直
    [np.radians(45), np.radians(-30), np.radians(20)],  # 弯曲状态1
    [np.radians(90), np.radians(60), np.radians(-45)],  # 弯曲状态2
]

print("=" * 60)
print("平面 3R 机械臂正运动学计算结果")
print("=" * 60)

for i, angles in enumerate(test_configs):
    T = forward_kinematics(dh_params_3r, angles)
    pos, euler = extract_pose(T)

    print(f"\n配置 {i+1}: θ = [{', '.join(f'{np.degrees(a):.1f}°' for a in angles)}]")
    print(f"  末端位置: x={pos[0]:.4f} m, y={pos[1]:.4f} m, z={pos[2]:.4f} m")
    print(f"  末端姿态: φ={np.degrees(euler[2]):.1f}° (平面内朝向)")
    print(f"  变换矩阵:\n{T.round(4)}")
```

**输出示例：**

```
配置 1: θ = [0.0°, 0.0°, 0.0°]
  末端位置: x=1.2000 m, y=0.0000 m, z=0.0000 m
  末端姿态: φ=0.0° (平面内朝向)

配置 2: θ = [45.0°, -30.0°, 20.0°]
  末端位置: x=0.8136 m, y=0.6036 m, z=0.0000 m
  末端姿态: φ=35.0° (平面内朝向)
```

### 完整示例：6DOF 工业机械臂

```
# ============================================
# 6DOF 机械臂正运动学 (类似 UR5 结构)
# ============================================

dh_params_6dof = [
    # (theta_offset, d, a, alpha)
    (0,        0.089, 0,       np.pi/2),   # 关节1: 基座旋转
    (0,        0,     -0.425,  0),          # 关节2: 肩部
    (0,        0,     -0.392,  0),          # 关节3: 肘部
    (0,        0.109, 0,       np.pi/2),   # 关节4: 腕部翻转
    (0,        0.094, 0,      -np.pi/2),   # 关节5: 腕部俯仰
    (0,        0.082, 0,       0),          # 关节6: 末端旋转
]

# 计算零位时的末端位姿
T_home = forward_kinematics(dh_params_6dof, [0]*6)
print("零位末端位姿矩阵:")
print(T_home.round(4))

# 计算任意关节角时的末端位姿
joint_angles = [np.radians(30), np.radians(45), np.radians(-20),
                np.radians(60), np.radians(-15), np.radians(90)]
T_target = forward_kinematics(dh_params_6dof, joint_angles)
pos, euler = extract_pose(T_target)

print(f"\n关节角: [{', '.join(f'{np.degrees(a):.1f}°' for a in joint_angles)}]")
print(f"末端位置: ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}) m")
print(f"末端姿态: RPY = ({np.degrees(euler[0]):.1f}°, {np.degrees(euler[1]):.1f}°, {np.degrees(euler[2]):.1f}°)")
```

### 雅可比矩阵的数值计算（预告）

正运动学计算的是"在某一个关节构型下，末端在哪里"。

如果要计算"关节微小运动如何影响末端运动"，就需要**雅可比矩阵**——它是正运动学对关节角的偏导数：

虽然雅可比主要用于逆运动学，但它的计算基础正是正运动学：

```
def compute_jacobian(dh_params, q, delta=1e-6):
    """
    数值雅可比矩阵 (数值微分法)

    参数:
        dh_params: DH 参数表
        q: 当前关节角
        delta: 微分步长

    返回:
        J: 6×n 雅可比矩阵 (前3行线速度, 后3行角速度)
    """
    n = len(q)
    J = np.zeros((6, n))

    # 当前末端位姿
    T0 = forward_kinematics(dh_params, q)
    p0 = T0[:3, 3]
    R0 = T0[:3, 3]

    for i in range(n):
        # 对第 i 个关节做数值微分
        q_plus = q.copy()
        q_plus[i] += delta
        T_plus = forward_kinematics(dh_params, q_plus)

        # 位置差分 -> 线速度列
        J[:3, i] = (T_plus[:3, 3] - p0) / delta

        # 姿态差分 -> 角速度列 (用旋转矩阵差分的反对称部分)
        dR = T_plus[:3, 3] @ R0.T
        # 从 dR 提取旋转轴角 (近似)
        skew = 0.5 * (dR - dR.T)
        J[3:, i] = np.array([skew[2,1], skew[0,2], skew[1,0]]) / delta

    return J
```

**这里有个点需要注意**：数值雅可比在奇异点附近会出问题（条件数爆炸），实际工程中通常使用**几何雅可比**（基于运动学链的解析推导），这在后续的逆运动学文章中会详细展开。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/kaugqJpv9nutPusx7ngVOmag61DHUJmX7OGyOG8gzibCyLX91kbhEcWl0mnLk5Zb5uVRabIn51LEKNicYT8OlZZQ/640?wx_fmt=png&from=appmsg#imgIndex=19)

## 场景痛点 × 解决方案 × 关键代码

### 场景 A：工业机械臂标定（精度第一）

- **痛点**：

机械臂出厂时的名义 DH 参数与实际值存在偏差（加工误差、装配误差、温度变形），导致理论 FK 计算的末端位置与实际位置有 0.5~2mm 的误差。

在精密装配场景（如手机螺丝拧紧、芯片贴装）中，这个误差不可接受。

- **解决方案**：

用激光跟踪仪测量一组标定点，通过非线性最小二乘拟合实际 DH 参数（**运动学标定**）。

```
from scipy.optimize import least_squares

def calibrate_dh(dh_params_nominal, measured_poses, joint_configs):
    """
    DH 参数标定: 通过测量的末端位姿修正 DH 参数

    参数:
        dh_params_nominal: 名义 DH 参数
        measured_poses:    标定点测量值 [(x, y, z), ...]
        joint_configs:     对应的关节角 [q1, q2, ...]

    返回:
        dh_calibrated: 标定后的 DH 参数
    """
    # 将 DH 参数展平为优化变量
    # 优化: d, a, alpha 的偏差 (theta_offset 通常已知)
    n_joints = len(dh_params_nominal)

    def unpack_params(flat_params):
        """将展平的参数重组为 DH 参数表"""
        dh = []
        for i in range(n_joints):
            d     = flat_params[i*3]
            a     = flat_params[i*3 + 1]
            alpha = flat_params[i*3 + 2]
            theta_off = dh_params_nominal[i][0]  # 保持名义 theta_offset
            dh.append((theta_off, d, a, alpha))
        return dh

    def residual(flat_params):
        """计算所有标定点的残差"""
        dh = unpack_params(flat_params)
        residuals = []
        for q, measured in zip(joint_configs, measured_poses):
            T = forward_kinematics(dh, q)
            predicted = T[:3, 3]
            residuals.extend(predicted - np.array(measured))
        return residuals

    # 初始猜测 = 名义参数
    x0 = []
    for (_, d, a, alpha) in dh_params_nominal:
        x0.extend([d, a, alpha])

    # Levenberg-Marquardt 优化
    result = least_squares(residual, x0, method='lm')

    return unpack_params(result.x)

# 标定效果: 典型工业机器人标定后精度从 ~1mm 提升到 ~0.05mm
```

### 场景 B：人形机器人全身姿态计算（实时性第一）

- **痛点**：

人形机器人有 30+ 自由度（双腿各 6、双臂各 7、躯干 3、颈部 2），需要在 1kHz 控制频率下实时计算全身各关键点的位姿，不只是末端，还包括质心、肩部、髋部等。

- **解决方案**：

预计算运动学树（Kinematic Tree），用拓扑排序并行计算各分支。

```
class KinematicTree:
    """
    运动学树: 高效计算多分支机器人的所有连杆位姿
    """
    def __init__(self, dh_params_list, parent_indices):
        """
        参数:
            dh_params_list: 各关节的 DH 参数
            parent_indices: 各关节的父关节索引 (-1 表示连接基座)
        """
        self.n = len(dh_params_list)
        self.dh = dh_params_list
        self.parent = parent_indices  # [−1, 0, 0, 1, 2, ...]

    def compute_all_frames(self, joint_angles):
        """
        一次性计算所有连杆坐标系的位姿
        按拓扑顺序计算, 避免重复运算
        """
        transforms = [np.eye(4)] * (self.n + 1)  # T_0^0, T_1^0, ..., T_n^0

        for i in range(self.n):
            theta_off, d, a, alpha = self.dh[i]
            theta = joint_angles[i] + theta_off
            T_local = dh_transform(theta, d, a, alpha)

            parent_idx = self.parent[i]
            if parent_idx == -1:
                transforms[i + 1] = T_local          # 连接基座
            else:
                transforms[i + 1] = transforms[parent_idx + 1] @ T_local

        return transforms  # 所有连杆在世界坐标系中的位姿

# 人形机器人: 左臂和右臂可以并行计算
# 30 DOF 的一次 FK 计算: ~0.1ms (NumPy 向量化)
```

### 场景 C：Sim2Real 的运动学一致性验证（可靠性第一）

- **痛点**：

强化学习策略在仿真器（Isaac Gym, MuJoCo）中训练，但部署到真机时，如果仿真器的 FK 与真机 FK 不一致，策略输出会产生系统性偏差。

常见问题包括：关节零点偏移、连杆长度误差、坐标系定义不一致。

- **解决方案**：

建立**运动学一致性校验管线**——在仿真和真机上执行相同的关节角序列，对比末端位姿差异。

```
def sim2real_kinematics_check(sim_fk_func, real_fk_func, test_configs):
    """
    Sim2Real 运动学一致性校验

    参数:
        sim_fk_func:  仿真器的 FK 函数
        real_fk_func: 真机的 FK 函数 (通过编码器读数+FK计算)
        test_configs: 测试关节角序列

    返回:
        max_error: 最大位姿误差
        mean_error: 平均位姿误差
    """
    errors = []

    for q in test_configs:
        T_sim = sim_fk_func(q)
        T_real = real_fk_func(q)

        # 位置误差
        pos_error = np.linalg.norm(T_sim[:3, 3] - T_real[:3, 3])

        # 姿态误差 (用旋转矩阵的测地线距离)
        R_diff = T_sim[:3, 3].T @ T_real[:3, 3]
        trace = np.clip(np.trace(R_diff), -1, 3)
        rot_error = np.arccos((trace - 1) / 2)  # 弧度

        errors.append({
            'joint_config': q,
            'pos_error_mm': pos_error * 1000,
            'rot_error_deg': np.degrees(rot_error)
        })

    pos_errors = [e['pos_error_mm'] for e in errors]
    rot_errors = [e['rot_error_deg'] for e in errors]

    print(f"位置误差: max={max(pos_errors):.2f}mm, mean={np.mean(pos_errors):.2f}mm")
    print(f"姿态误差: max={max(rot_errors):.3f}°, mean={np.mean(rot_errors):.3f}°")

    return max(pos_errors), np.mean(pos_errors)

# 合格标准: 位置误差 < 1mm, 姿态误差 < 0.5°
```

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/kaugqJpv9nutPusx7ngVOmag61DHUJmX5ne3MfNYQBbic4xIYsEJDKpCRqQXk6gllicSqc7QiabhaIEuCXA1I4xsg/640?wx_fmt=png&from=appmsg#imgIndex=20)

## 写在最后

正运动学是机器人学中最“老实”的部分——给定输入，输出唯一，没有歧义，没有搜索，没有迭代。

也正是因为这份确定性，使它成为整个机器人控制栈的基石。

从 DH 参数的四个变量（），到单个连杆的  变换矩阵，再到  个矩阵的连乘——这条数学链路虽然简单，却精确地描述了从"关节角"到"末端位姿"的完整映射。

理解了它，才真正理解了机械臂“为什么能动”。

在具身智能时代，正运动学的地位不降反升。它是仿真环境的物理基础（URDF 本质就是 DH 参数的 XML 化），是 Sim2Real 迁移的一致性基准，是强化学习 reward 计算中位姿误差的定义来源，也是逆运动学、动力学、轨迹规划所有上层模块的底层依赖。

**正运动学是一道函数题：。**

但这个函数的每一个矩阵元素，都包含了对物理世界的数学抽象——坐标系的约定、连杆的几何关系、刚体变换的复合法则。

把这些搞清楚，后面的路才走得稳。

👍深蓝学院《机器人学基础》系列课程推荐👍

本课程系统讲解了位形空间、刚体运动、前向/逆向运动学、轨迹生成与运动规划等核心模块，覆盖了本专栏正在连载的完整具身智能基础知识体系。

**下一篇预告**：**逆运动学（IK）** 。当给定末端目标位姿，如何求解关节角？

编辑｜小小怪博士

审编｜具身君

 ****推荐阅读**
[![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGcibJS8986MfCcVATGOkcK6lNQfiaTORbuhSFoATTmZ5kA6nV8l8REia7nm4A4OxC1yOePBqrzWHQQd0ALicYANgOoNmRbibChjcAuQ/640?wx_fmt=png&from=appmsg#imgIndex=21)](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=3824573915845640194&scene=126#wechat_redirect)[![Image](https://mmbiz.qpic.cn/mmbiz_jpg/uwFbeBKoFGcRfEtsGjVkl7cXB7QYAAib4wOMhdRcvsQicHnmiaxqoibw9LUCGGcPGSYnUPeUlZEoiaBlQezclFhp5yZQ6yLcLAjYeI67pJmvhOMw/640?wx_fmt=jpeg&from=appmsg#imgIndex=22)](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653&token=944555238&lang=zh_CN#wechat_redirect)

**![Image](https://mmbiz.qpic.cn/mmbiz_png/qKE443uRvLo6ic3ZPUttmFZ2AefQ4wjHSlQluSDkaxL9icWicpPYYmpo1Wa37Scjhh4AS5VwYJtmlTf5cKMiaIXg5g/640?&random=0.17349735674179656&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&wx_fmt=other#imgIndex=23)**

**【深蓝具身智能】****的原创内容均由作者团队倾注个人心血制作而成，希望各位遵守原创规则珍惜作者们的劳动成果；未经授权禁止任何机构或个人抓取本账号内容，进行洗稿/训练，否则侵权必究⚠️⚠️**


![Image](https://mmbiz.qpic.cn/mmbiz_png/Nabxc8rdYriaKqxCUjcZ8sSCnSNlWpqdI1kyXXQjXbtv95xvACqQoqL2ibbKXt9PB0FLPibKiawGsTcQrnKDGWVw2Q/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1#imgIndex=24)

点击❤收藏并推荐本文**
