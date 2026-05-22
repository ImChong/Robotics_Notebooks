# 彻底搞懂具身智能必备知识：李群、李代数、四元数

> 来源归档（blog / 微信公众号）

- **标题：** 彻底搞懂具身智能必备知识：李群、李代数、四元数
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号；《具身智能基础》专栏第 1 篇）
- **原始链接：** https://mp.weixin.qq.com/s/JviRH2LW-fkCHA5gY7Qflw
- **发表日期：** 2026-05-22（frontmatter）
- **入库日期：** 2026-05-22
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0 安装的 `wechat-article-for-ai`（Camoufox）；正文约 1.0 万字 / 19 图；Jina Reader 对该链接触发微信 CAPTCHA，未采用
- **关联推荐（文内）：** [极具「影响力」的12个VLA开源项目](https://mp.weixin.qq.com/s?__biz=MzkwMDcyNDUzMQ==&mid=2247494473&idx=1&sn=28c95bea437f22cc8e9ed7ca3308a071)、[刷完 Github VLA 后最推荐复现的几个](https://mp.weixin.qq.com/s/k_i-1NEBP-lEzth19HOHkQ)（已入库 [`wechat_shenlan_vla_github_repro_survey_2025.md`](wechat_shenlan_vla_github_repro_survey_2025.md)）、[VLN发展历程中4个代表性项目复现](https://mp.weixin.qq.com/s?__biz=MzkwMDcyNDUzMQ==&mid=2247496491&idx=1&sn=9ed38a5f612d0e3b670f3d6a74a8d5d7)
- **一句话说明：** 从「深度学习把旋转优化暴露到网络前向/反向」出发，系统梳理 SO(3)/SE(3) 李群、so(3)/se(3) 李代数与单位四元数的分工：李群保证物理合法姿态，李代数提供可微线性增量，四元数是 SO(3) 的工程存储格式；并串联 SLAM、轨迹优化、WBC、MPC、VLA 动作微调等落地场景。

## 核心摘录（归纳，非全文）

### 问题重框

- 传统 SLAM / 轨迹优化 / 运动控制里，旋转多在优化器内部用欧拉角、旋转矩阵或「小心扰动」的四元数凑合。
- **具身智能 + 深度学习** 把几何操作推到网络前向与反向传播：策略要输出动作、在大姿态变化下优化、保证轨迹物理可行时，必须在 **旋转流形** 上工作，不能只在 $\mathbb{R}^n$ 里硬回归。
- 专栏定位：不追热点 demo，用「第一性原理」补 **骨骼级数学**；本篇选李群/李代数/四元数为《具身智能基础》开篇。

### 李群：合法刚体运动集合

| 对象 | 含义 | 工程角色 |
|------|------|----------|
| **SO(3)** | 3×3 正交、$\det=1$ 的旋转矩阵集合 | 纯旋转、无万向锁（相对欧拉角） |
| **SE(3)** | 齐次 $4\times4$ 变换 $T=\begin{bmatrix}R&t\\0&1\end{bmatrix}$ | 末端位姿、相机外参、车体「位置+姿态」 |

- SO(3) 约束：$R^\top R=I$，$\det R=1$（排除反射）。
- SE(3) 把旋转与平移合成矩阵乘法，便于运动复合。

### 李代数：切空间上的线性增量

| 对象 | 维度 | 角色 |
|------|------|------|
| **so(3)** | 3 | 旋转增量 $\omega$；反对称矩阵 $[\omega]_\times$ |
| **se(3)** | 6 |  twist：3 旋转 + 3 平移增量 |

- **指数映射** $\exp$：so(3)/se(3) → SO(3)/SE(3)（Rodrigues / 含平移的 SE(3) 公式）。
- **对数映射** $\log$：旋转矩阵 → $\omega$（由 $\mathrm{tr}(R)$ 等得旋转角）。
- 比喻：李群像「弯曲球面」，李代数像「摊平的切平面」——在切空间做加减、求导，再映回流形。

### 四元数：SO(3) 的便捷存储

- 单位四元数 $q=(w,x,y,z)$，$\|q\|=1$；$q$ 与 $-q$ 同一旋转（双倍覆盖）。
- **分工：** 四元数存姿态；so(3) 做优化增量；SO(3) 矩阵做变换复合与约束检验。
- 典型链路：**四元数 → 旋转矩阵 $R$ → $\log(R)\to\omega$ → 在 $\omega$ 上梯度下降 → $\exp(\omega)\to R$ → 回写四元数**。

### 为何具身 / 自动驾驶离不开这套语言（作者三点）

1. **姿态表示：** 欧拉角万向锁；SO(3)/SE(3) 无此缺陷。
2. **可微优化：** 旋转矩阵有正交约束，不宜直接反传；se(3)/so(3) 无约束向量可进 Adam 等。
3. **物理一致：** 李群结构保证刚体运动合法，避免「瞬移/扭曲」类非物理输出。

### 文内列举的下游场景（策展）

- **SLAM / 位姿图：** 位姿在 SE(3)，增量在 se(3)。
- **运动规划：** so(3) 采样控制点，指数映射得平滑姿态轨迹。
- **全身控制：** 任务空间速度为 se(3) 元素，经雅可比映射关节空间。
- **MPC：** se(3) 切空间线性化 → 标准 QP。
- **大模型动作微调：** 在 se(3) 上定义动作增量，保证每步修正几何合法。

### 与站内已有条目的关系

- 教材级系统推导见 [Modern Robotics](../../wiki/entities/modern-robotics-book.md)；DL 侧连续表示对比见 [SE(3) Representation](../../wiki/formalizations/se3-representation.md)。
- 本篇为 **公众号科普 + 工程分工表**，不替代教材证明；公式细节以 Lynch & Park Ch 3 与抓取正文为准。

## 对 wiki 的映射

- [lie-group-rigid-body-motions](../../wiki/formalizations/lie-group-rigid-body-motions.md)（本次升格主页面）
- [se3-representation](../../wiki/formalizations/se3-representation.md)、[modern-robotics-book](../../wiki/entities/modern-robotics-book.md)
- [whole-body-control](../../wiki/concepts/whole-body-control.md)、[model-predictive-control](../../wiki/methods/model-predictive-control.md)、[ekf](../../wiki/formalizations/ekf.md)

## 可信度与使用边界

- 公众号为 **入门导航 + 工程直觉**，部分公式在 OCR/抓取中可能缺符号；关键推导请对照 *Modern Robotics* 或官方 PDF。
- 文内推广与商务信息已剥离；外链图片为微信 CDN，wiki 不复述正文图。
- 未将文内两篇「推荐阅读」单独 ingest（VLA 12 项目、VLN 复现）；若后续入库另开 ingest。

## 当前提炼状态

- [x] Agent Reach + Camoufox 正文抓取与归纳摘要
- [x] SO(3)/SE(3)/so(3)/se(3)/四元数分工与转换链路
- [x] wiki 主页面映射确认
