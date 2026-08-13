---
title: 聊聊雅可比矩阵，如何统一机器人控制的两条主线？
author: 深蓝具身智能
date: "2026-08-07 17:07:00"
source: "https://mp.weixin.qq.com/s?__biz=MzkwMDcyNDUzMQ==&mid=2247507685&idx=1&sn=f240a287b15dfb1fc7ebb4804f61a359"
---

# 聊聊雅可比矩阵，如何统一机器人控制的两条主线？

![Image](https://mmbiz.qpic.cn/sz_mmbiz_gif/kaugqJpv9nuCktylvYoMKHYNAVojoRUpfyf1py08JvUnkfPXArzj4t5bMiaS6RBCXHHGhf8xlyw8icHrJcjEyYoA/640?wx_fmt=gif&from=appmsg#imgIndex=0)

![Image](https://mmbiz.qpic.cn/mmbiz_gif/uwFbeBKoFGcwVG4n9ClVqXwVZtp5XgVFquIZKxQia7rbAN98tSSzdpQ2VTxiaL8iaxZLXTgjvj2th3zNM0GSVRGIMEer7AeXkXo0Jt0U39wtB4/640?wx_fmt=gif&from=appmsg#imgIndex=1)

正运动学是一张地图，雅可比矩阵是地图上的“比例尺”

> 大家好，这里是【深蓝具身智能】。
>
> 本文出自《具身智能基础》专栏，是本栏目下的第十篇文章，聚焦于机器人建模框架中的雅可比矩阵与微分运动学。

---

[💙](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653&token=1419275101&lang=zh_CN#wechat_redirect)[订阅《具身智能基础》专栏](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653&token=1419275101&lang=zh_CN#wechat_redirect)

你的订阅和收藏，将支持我们把这件事持续做下去✨

「一只六轴机械臂正在执行曲面打磨，工件表面带有弧度，末端工具需要沿着法向保持恒定压力，同时沿切向以 50 mm/s 匀速行进……」

这是操作员在示教器上规划出的轨迹，目标清晰明确：位置、速度、姿态、接触力。这些人能直观理解和规划的量，被全部定义在任务空间里。

但机械臂听不懂这些。

它的底层执行器只接收一种信号：每个电机的目标位置（六个关节各转多少）或速度（何时转、转多快），这些被定义在关节空间里。

这就形成了机器人控制最核心的矛盾：我们在任务空间定义目标，机器人却必须在关节空间执行运动。

翻译这道鸿沟，具体到物理层面，无非就是两个问题：

第一，速度怎么映射？ 第二，力怎么映射？

雅可比矩阵，就是同时回答这两个问题的核心桥梁。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGcMKibVKaS5D6DiaqgwbbFFgaPxsPLVPGSKHKvXLbB8epaaFHlRiaOwp8BL9mY5ZSqm8lIPukmq0Jq9mw6FBbb8fIow9XUmgeLC5Y/640?wx_fmt=png&from=appmsg#imgIndex=2)

这篇文章，我们会从直觉讲到公式，从速度映射讲到力映射，最后看看它如何串起现代机器人算法的主线——

逆运动学IK、全身运动控制WBC、模型预测控制MPC、强化学习RL。

**我们开设此账号，除了想要向各位对【具身智能】感兴趣的人传递前沿权威的知识讯息外，也想和大家一起见证它到底是泡沫还是又一场热浪？****欢迎关注****【深蓝具身智能】**👇

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4YWHX9iazKkdkgh363zh9GFAfZia4RWWoYhutUeS8g43MnicLMfe9kUAZg/640?wx_fmt=jpeg&from=appmsg#imgIndex=3)

## 关节空间与任务空间：两个必须打通的世界

## 机器人控制里始终存在两个空间。

- 关节空间

由每个电机直接驱动，维度等于关节数。一台 6 轴工业臂的关节空间是 6 维，一只 20 自由度的灵巧手则是 20 维。

在这个空间里，控制器直接下达指令：“第 3 个关节以 0.5 rad/s 转动”。

- 任务空间

人真正思考问题的空间。它通常由末端的位置和姿态组成，在三维世界里是 6 维：3 维线速度加 3 维角速度。

“末端向前移动 1 cm”、“工具绕自身 z 轴旋转 5°”，这些指令都在任务空间里。

这两个空间不是一回事。同样的末端位姿，关节可能有多种组合；同一个关节运动，在不同构型下对末端产生的贡献也不同。

把它们连起来的第一步叫正运动学：给定关节角  ，通过连杆变换连乘，算出末端位姿  。

正运动学像一张地图，告诉你从关节到末端的“位置关系”。（这部分内容见本专往期文章）

但控制机器人时，我们真正关心的不仅是“在哪”，更是“怎么变”。

一旦进入"变化"这个话题，雅可比矩阵就自然出现了。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/uwFbeBKoFGdba9hNncbabU72ULzQDUm3nQJFnicoeZF7xVciccCS6dU8cgLU5BIhic6micKRBs00WzBZDHh3BLrmreSM8nUg2fOiaesiaVNGIylCs/640?wx_fmt=png&from=appmsg#imgIndex=4)

▲图1| 关节空间与任务空间之间，正运动学负责"位置映射"，雅可比矩阵负责"速度映射"。©【深蓝具身智能】编译

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nsAicIiaQwb1eFDMZwlNcXLBibqgVaodXH45G6Pdbk9xSEsUtlicqgxKkAiaK0P8QzGwuLiatibYiaIagQoOg/640?wx_fmt=jpeg&from=appmsg#imgIndex=5)

## 雅可比矩阵：局部的"速度翻译器"

从数学上看，雅可比矩阵描述的是一个多变量函数在当前点附近的一阶线性近似：

输入发生微小变化时，输出会怎么跟着变。

把这个关系对时间求导，就得到机器人里最常见的形式：

其中， 是末端的 6 维速度旋量（前 3 维是线速度，后 3 维是角速度），是关节速度向量，是一个 的矩阵，n 是关节数。

这个式子的含义：末端速度等于雅可比矩阵乘以关节速度。

注意这里有个关键词：局部。

雅可比矩阵只在当前构型附近成立：机器人一旦运动，关节角度改变，的数值也会随之更新。

控制器不是一次性解出全局答案，而是在每一小步里都用当前雅可比做一次"就近翻译"。

打个比方：正运动学是一张完整的地图，而雅可比矩阵是地图在当前位置上的比例尺。它告诉你，沿着眼前这一小片区域走，步子会怎么被放大或缩小。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nsAicIiaQwb1eFDMZwlNcXLBibTvia4qLjYyoM2Do58jX9J71HickLLA3NxCQp6fPljkgY26WIeaoeeYVQ/640?wx_fmt=jpeg&from=appmsg#imgIndex=6)

## 几何直觉：每一列都是一个关节的"运动指纹"

雅可比矩阵为什么能完成翻译？答案在它的列向量里。

的第 i 列，描述的是：在其他关节都固定的情况下，第 i 个关节以单位速度运动时，末端会产生怎样的速度贡献。

每一列都是这个关节在当前构型下的“运动指纹”。

对于一个转动关节，关节绕自身的旋转轴  转动，会同时给末端带来两种效果：

- 角速度贡献：末端会获得一个沿方向的角速度。
- 线速度贡献：末端会绕关节轴做瞬时圆周运动，线速度方向垂直于旋转轴和从关节指向末端的力臂。

写成列向量就是：

上式中， 是末端位置， 是第  个关节的位置， 表示叉积。

而对于移动关节，关节直接沿轴平移，只产生线速度：

把所有关节的列向量并排拼起来，就得到完整的雅可比矩阵：

这揭示了一个事实：末端的速度，是所有关节瞬时运动贡献的线性叠加。

这种几何理解比直接对公式求导更直观，也更容易在调试时判断"当前是哪个关节在主导末端朝某个方向运动"。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/uwFbeBKoFGfIlCic7sTvpf2oyUwibJ24F4ibHIca7Mz39Z1jcrMQUSfIMjAeqNrLvVT2zpiatQicGWYKxGcuctkEwToXT5yvyQJLZy0hj6sDY9VM/640?wx_fmt=jpeg#imgIndex=7)

▲图2| 雅可比矩阵的第 i 列，就是第 i 个关节在当前构型下对末端速度的"运动指纹"。©【深蓝具身智能】编译

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4yZibwaOWpB3DrcxuiafpXicx2ibHiaHAZFr7ptU6ud2hsxgCXvV0JGHtTDw/640?wx_fmt=jpeg&from=appmsg#imgIndex=8)

## 从速度到力：雅可比转置的对偶性

雅可比矩阵最早通常出现在速度问题里，所以很多人误以为它只是一个“速度工具”。

真正让它在机器人学中地位如此高的，是它和力之间的对偶关系。

如果末端受到一个广义力（力和力矩），那么关节需要提供的力矩  满足：

这个式子被称为力雅可比。它和速度雅可比形成了一种对称关系：

- 速度映射：，把关节速度映射到末端速度；
- 力映射：，把末端力映射回关节力矩。

这意味着，雅可比矩阵不仅是运动学里的桥梁，也是力学里的桥梁。机器人一旦涉及接触、装配、打磨、抛光、协作搬运，就一定会用到这层关系。

比如末端要沿某个方向以恒定压力压紧工件，这个任务空间的力指令最终必须通过分配到各个关节的力矩上。

阻抗控制、导纳控制、力位混合控制，本质上都是在反复调用这一对映射。

所以更准确地说：雅可比矩阵既告诉机器人“怎么动”，也告诉机器人“怎么发力”。

运动和受力这两件事，在这里被统一了起来。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGczkarak3xnEcibDTGXU8m68ichtBnibT2Brj3XsSqGQ7c3TJLyndW9icPMI7ic7DAfc3I4UvdQkfxn3PqUQ7L2ull8IXBicYpQIB2dQ/640?wx_fmt=png&from=appmsg#imgIndex=9)

▲图3| 速度映射与力映射共享同一座"桥"，只是方向相反，分别对应 J 与 Jᵀ。©【深蓝具身智能】编译

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/kaugqJpv9nuLZSia1RtMfiapaRw4IyTJN4hKdH0P2rRHX1TlxUqlAx7X6m2hcl7XttttyRW05mhbEa1msX7zEzvw/640?wx_fmt=jpeg&from=appmsg#imgIndex=10)

## 一条线串起 IK、WBC、MPC 与强化学习

很多机器人算法表面看差异很大，但往底层走，经常会发现它们共享同一条运动学主线：

把任务空间的目标转成关节空间的执行量，而雅可比矩阵就是这个转换的接口。

![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGct8ACpicFIWO8J0AibNutmR71aMiaiclH8OsiayJSb3Xa9GKicb0tYTbgSbVEzSnFPrIbTf5cr0gkich4EPDscz8Ot5QUyicT6Ue2iaKLc/640?wx_fmt=png&from=appmsg#imgIndex=11)

▲图4| 从 IK 到 RL，现代机器人算法在不同层级反复调用雅可比矩阵描述的局部运动学结构。©【深蓝具身智能】编译

- 数值逆运动学（Numerical IK） 是最直接的例子

末端当前位姿与目标位姿之间存在误差 ，一种自然的做法是让末端沿误差减小的方向运动：

这里的  是雅可比矩阵的伪逆。逆解问题于是从“直接求位置解”变成了“反复做速度层面的修正”。在冗余机器人上，还可以在雅可比的零空间里叠加次级任务，比如避障或保持关节居中。

与逆运动学IK相关的内容，详见本专往期文章

- 全身控制（WBC） 更进一步

它同时处理多个任务：躯干平衡、末端跟踪、足部约束、关节限位。

每个任务都被写成任务空间的速度或力约束，而雅可比矩阵负责把这些约束统一投影到关节层面。

- 模型预测控制（MPC） 则利用雅可比做局部线性化

机器人在未来一小段时间内的运动模型通常是非线性的，雅可比矩阵帮助控制器在当前工作点附近把它近似成线性系统，从而快速求解优化问题。

- 甚至在强化学习里，雅可比也同样扮演着重要角色。

策略网络学习的是从观测到动作的映射，但它最终要作用在一个有物理结构的系统上。

雅可比矩阵提供了系统局部的运动学敏感性，帮助学习算法理解“当前状态下，哪个关节的动作对末端影响最大”。

从这个角度看，雅可比矩阵不是某个算法里的中间公式，而是机器人算法体系的基础接口。它让逆解、控制、优化、学习这些方法有了一个共同的落脚点。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/kaugqJpv9nutPusx7ngVOmag61DHUJmX7OGyOG8gzibCyLX91kbhEcWl0mnLk5Zb5uVRabIn51LEKNicYT8OlZZQ/640?wx_fmt=png&from=appmsg#imgIndex=12)

## 奇异与零空间：雅可比矩阵的"边界提示"

雅可比矩阵虽然强大，但也有自己的“边界"”。

当机器人在某些特殊构型下，雅可比矩阵的秩下降，意味着末端在至少一个方向上的瞬时运动能力丧失了。

- 这种构型称为：奇异位形

在奇异点附近，逆解和力控制都会变得不稳定，因为同样的末端速度可能需要极大的关节速度，或者同样的末端力会产生异常的关节力矩。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_gif/uwFbeBKoFGd52fvIzon8v4jhrCzWWIfAMdJLZ75F4KmRjHTzN8af1b8V7xJDa2stibYTgOoowIt7Doq9siberhlPwx13cbV53QKqmHpxO3w5c/640?wx_fmt=gif&from=appmsg#imgIndex=13)

- 零空间是另一个重要概念

对于自由度多于任务空间维度的冗余机器人，雅可比矩阵的零空间提供了一层宝贵的自由度。



零空间里的关节运动满足，不会改变末端位姿，却可以用来完成避障、姿态优化、能耗最小化等次级任务。



理解奇异和零空间，不是为了把公式写得更复杂，而是为了在工程里知道：

雅可比矩阵不是万能的，它只是在当前局部、当前构型下的最佳线性近似。

越过这个边界，就需要更高级的工具来处理。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_png/kaugqJpv9nutPusx7ngVOmag61DHUJmX5ne3MfNYQBbic4xIYsEJDKpCRqQXk6gllicSqc7QiabhaIEuCXA1I4xsg/640?wx_fmt=png&from=appmsg#imgIndex=14)

## 雅可比矩阵是接口，不是终点

雅可比矩阵本身并不难理解：它是正运动学在当前构型下的一阶线性近似，把关节速度翻译成末端速度，再通过转置把末端力翻译回关节力矩。

它真正的价值，在于把机器人学里许多看似分散的问题串到了同一个框架下。逆运动学、全身控制、模型预测控制、强化学习——这些方法没有绕过运动学，而是在运动学给出的结构之上继续搭建。

如果说正运动学回答的是“机器人在哪里”，那么雅可比矩阵回答的就是“机器人现在能怎么变”。

这个"变"字，才是控制的本质。

编辑｜小小怪博士

审编｜具身君

 ****推荐阅读**
[![Image](https://mmbiz.qpic.cn/mmbiz_png/uwFbeBKoFGcibJS8986MfCcVATGOkcK6lNQfiaTORbuhSFoATTmZ5kA6nV8l8REia7nm4A4OxC1yOePBqrzWHQQd0ALicYANgOoNmRbibChjcAuQ/640?wx_fmt=png&from=appmsg#imgIndex=15)](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=3824573915845640194&scene=126#wechat_redirect)[![Image](https://mmbiz.qpic.cn/mmbiz_jpg/uwFbeBKoFGcRfEtsGjVkl7cXB7QYAAib4wOMhdRcvsQicHnmiaxqoibw9LUCGGcPGSYnUPeUlZEoiaBlQezclFhp5yZQ6yLcLAjYeI67pJmvhOMw/640?wx_fmt=jpeg&from=appmsg#imgIndex=16)](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653&token=944555238&lang=zh_CN#wechat_redirect)

**![Image](https://mmbiz.qpic.cn/mmbiz_png/qKE443uRvLo6ic3ZPUttmFZ2AefQ4wjHSlQluSDkaxL9icWicpPYYmpo1Wa37Scjhh4AS5VwYJtmlTf5cKMiaIXg5g/640?&random=0.17349735674179656&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1&wx_fmt=other#imgIndex=17)**

**【深蓝具身智能】****的原创内容均由作者团队倾注个人心血制作而成，希望各位遵守原创规则珍惜作者们的劳动成果；未经授权禁止任何机构或个人抓取本账号内容，进行洗稿/训练，否则侵权必究⚠️⚠️**


![Image](https://mmbiz.qpic.cn/mmbiz_png/Nabxc8rdYriaKqxCUjcZ8sSCnSNlWpqdI1kyXXQjXbtv95xvACqQoqL2ibbKXt9PB0FLPibKiawGsTcQrnKDGWVw2Q/640?wx_fmt=other&from=appmsg&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1#imgIndex=18)

点击❤收藏并推荐本文**
