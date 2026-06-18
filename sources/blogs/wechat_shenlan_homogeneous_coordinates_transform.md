# 补全具身智能L0级必备知识：齐次坐标与齐次变换

> 来源归档（blog / 微信公众号）

- **标题：** 补全具身智能L0级必备知识：齐次坐标与齐次变换
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号；《具身智能基础》专栏第 5 篇）
- **原始链接：** https://mp.weixin.qq.com/s/3vwaizPOgJKCwQ9e5LuKGA
- **发表日期：** 2026-06-18
- **入库日期：** 2026-06-18
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox）；正文约 6000 字 / 22 图；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **专栏专辑：** [《具身智能基础》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkwMDcyNDUzMQ==&action=getalbum&album_id=4525948187102363653)
- **专栏姊妹篇：** [李群、李代数、四元数](wechat_shenlan_lie_group_lie_algebra_quaternion.md)（`JviRH2LW-fkCHA5gY7Qflw`）；[三维世界坐标变换](wechat_shenlan_3d_coordinate_transforms.md)（`P5Jm7bMhaTHsytHStFbbLg`）；[黎曼流形与切空间](wechat_shenlan_riemannian_manifold_tangent_space.md)（`uFTKN5FDvlHQxOSspvxVZw`）；[RL 最小闭环](wechat_shenlan_rl_embodied_minimal_closed_loop.md)（`hHkQqLfIOTn0CoAZNuLWJA`）
- **一句话说明：** 从「旋转矩阵乘法 + 平移向量加法」运算割裂出发，说明齐次坐标（点 $w=1$、方向 $w=0$）如何把刚体运动统一为 $4\times4$ SE(3) 矩阵乘法，支撑多连杆 FK、SLAM 位姿连乘、自动驾驶轨迹迭代与 se(3) 可微优化；定位为专栏 L0 工程底座，衔接李群/四元数理论与代码实现。

## 核心摘录（归纳，非全文）

### 问题重框

- 三维笛卡尔下刚体运动：$p' = Rp + t$ — **旋转是矩阵乘、平移是向量加**，无法统一为单一矩阵运算，多级变换只能分步累加 → 误差累积、实时性差、不利于端到端可微。
- 文内分工表：**笛卡尔 = 原始描述；齐次变换 = 工程转换器；李群/李代数 = 理论优化器；四元数 = 轻量化载体**。没有齐次变换，SE(3) 微分与流形优化缺少数值载体。

### 齐次坐标定义

| 对象 | 笛卡尔 | 齐次（工程约定） |
|------|--------|------------------|
| **空间点** | $(x,y,z)$ | $(x,y,z,1)^\top$ — 具平移属性 |
| **方向向量** | $(x,y,z)$ | $(x,y,z,0)^\top$ — 仅旋转、不平移 |
| **归一化** | — | $(x,y,z,w) \sim (\lambda x,\lambda y,\lambda z,\lambda w)$，$w\neq0$ 时还原 $ (x/w,y/w,z/w)$ |

### 齐次变换矩阵（SE(3)）

$$
T = \begin{bmatrix} R & t \\ 0 & 1 \end{bmatrix}, \quad \tilde p' = T\,\tilde p
$$

- **纯旋转：** $T_R = \mathrm{diag}(R,1)$
- **纯平移：** $T_t = \begin{bmatrix} I & t \\ 0 & 1 \end{bmatrix}$
- **连续运动：** $T_{\text{total}} = T_n \cdots T_2 T_1$（矩阵连乘）

### 五大工程特性（文内）

1. 运算统一（纯矩阵乘）
2. 运动可叠加（多连杆 / 长轨迹）
3. 物理区分（点 vs 方向 $w$ 分量）
4. 物理合法性（继承 $R$ 正交，无拉伸）
5. 理论兼容（SE(3)/se(3) 优化载体）

### 落地场景（文内 + 代码片段）

| 场景 | 齐次变换角色 |
|------|----------------|
| **机械臂 FK/IK** | 关节级 $T_i$ 连乘得末端位姿 |
| **视觉 / SLAM** | 外参 $T_{\text{cam}\leftarrow\text{world}}$；点云跨系变换 |
| **自动驾驶** | 车身位姿序列 $T_t \leftarrow T_\Delta T_{t-1}$ |
| **端到端优化** | SE(3) $\leftrightarrow$ se(3) 李代数，无约束梯度后再 $\exp$ 回群 |

文内给出 NumPy 构造 `make_T`、相机点反投影、里程计位姿更新、SE3↔se3 转换等示例代码（见 raw 落盘）。

### 专栏收束（文内）

- 本篇补全「三维刚体运动描述底层数学框架」的 **代码前提**；下一篇预告继续 RL 梳理。

## 对 wiki 的映射

- [homogeneous-coordinates-transform](../../wiki/formalizations/homogeneous-coordinates-transform.md)（本次升格主页面）
- [shenlan-embodied-ai-fundamentals-series](../../wiki/overview/shenlan-embodied-ai-fundamentals-series.md)（专栏父节点，补第 5 篇索引）
- [lie-group-rigid-body-motions](../../wiki/formalizations/lie-group-rigid-body-motions.md)、[3d-coordinate-transforms-vision-robotics](../../wiki/formalizations/3d-coordinate-transforms-vision-robotics.md)、[se3-representation](../../wiki/formalizations/se3-representation.md)

## 可信度与使用边界

- 公众号为 **L0 工程直觉 + 代码导航**；严格推导见 *Modern Robotics* Ch 2–3 与 [modern_robotics_textbook.md](../papers/modern_robotics_textbook.md)。
- 文内 se(3)↔SE(3) 示例为教学简化（完整 se(3) 指数映射含 $V$ 矩阵），工程优化请用李群页公式。
- 推广/商务信息已剥离；图在微信 CDN，wiki 用 Mermaid 复述链路。

## 当前提炼状态

- [x] Agent Reach v1.5.0 + wechat-article-for-ai 正文抓取与归纳摘要
- [x] 齐次坐标、SE(3) 矩阵、五大特性与四场景映射
- [x] wiki 主页面与专栏父节点更新
