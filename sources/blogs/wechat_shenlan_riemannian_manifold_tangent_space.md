# 具身智能基础｜彻底搞懂：黎曼流形、切空间下的运动

> 来源归档（blog / 微信公众号）

- **标题：** 具身智能基础｜彻底搞懂：黎曼流形、切空间下的运动
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号；《具身智能基础》专栏第 3 篇）
- **原始链接：** https://mp.weixin.qq.com/s/uFTKN5FDvlHQxOSspvxVZw
- **发表日期：** 2026-06-04
- **入库日期：** 2026-06-04
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0 + `wechat-article-for-ai`（Camoufox）；正文约 0.99 万字 / 22 图
- **专栏姊妹篇：** [李群、李代数、四元数](wechat_shenlan_lie_group_lie_algebra_quaternion.md)；[三维世界坐标变换](wechat_shenlan_3d_coordinate_transforms.md)
- **一句话说明：** 论证具身运动状态（关节周期、SO(3)/SE(3) 姿态）天然落在 **黎曼流形** 上，算法在 **切空间** 做线性增量，经 Exp/Log 与流形双向映射；对比「欧式补丁思维」与「流形原生计算」，并梳理局部线性、离散 Exp/Log、固定度量、欧式梯度等工程近似及其误差边界。

## 核心摘录（归纳，非全文）

### 为何欧式空间不够

- 关节角 $0°=360°$、旋转 $360°$ 等价：欧式插值会得到 **340° 而非 20°** 的错误路径。
- SO(3) 9 维矩阵仅 3 自由度；SE(3) 平移可延伸、旋转闭合弯曲——合法状态无法铺满 $\mathbb{R}^n$。

### 黎曼流形 vs 欧式

| 维度 | 欧式 | 黎曼流形（机器人状态） |
|------|------|------------------------|
| 形状 | 全局平直 | 局部平直、整体弯曲 |
| 最短路径 | 直线 | **测地线**（球面大圆类比） |
| 约束 | 后处理补丁 | **内建**于空间结构 |
| 具身 vs 自驾 | — | 文内：人形以转动为主（「化圆为方」）vs 轨迹折线平滑（「化方为圆」） |

### 切空间

- 流形上点 $x$ 的 **切空间** $T_x\mathcal{M}$：该点处所有切向量构成的 **线性** 向量空间，维度等于流形维度。
- **分工**：流形存 **绝对状态**；切空间算 **增量 / 梯度 / 速度**。

### Exp / Log（与 SO(3)/SE(3) 统一）

- **指数映射** $\exp_x: T_x\mathcal{M} \to \mathcal{M}$：切空间增量 → 流形合法新状态。
- **对数映射** $\log_x$：状态差 → 切空间线性向量；测地距离 $\approx \|\log_x(y)\|$。
- 与 [李群页](../../wiki/formalizations/lie-group-rigid-body-motions.md) 中 Rodrigues / twist 为 **同一底层**（欧拉公式为 $\mathrm{SO}(2)$ 特例）。

### 工程近似（文内清单）

| 近似 | 何时误差小 | 何时暴露 |
|------|------------|----------|
| 切空间一阶线性化 | 小增量、小步长 RL | 大姿态旋转、大步长 |
| Exp 一阶泰勒 | 离散控制常用 | 大 $\Delta$ 偏离测地线 |
| Log 差分近似 | 损失、姿态误差 | 大角度 |
| 固定局部度量 | 短程距离 | 长轨迹累积 |
| 欧式梯度代替黎曼梯度 | 快速训练 | 优化方向偏测地线 |
| 切空间线性插值 | 轻量轨迹 | 应用测地线插值 |

### 落地场景（作者归纳）

- **运动控制**：流形轨迹平滑、避免非法姿态。
- **RL**：切空间梯度 + Exp 更新 $x_{k+1}=\exp_{x_k}(-\eta\,\mathrm{grad})$。
- **世界模型 / 表征**：多模态嵌入低维流形隐空间，相似观测邻近、演化沿测地线。
- **定位 / 多模态**：几何不变性适配坐标系切换。

### 核心论断

- 李群/李代数是 SO(3)/SE(3) 的 **特例语言**；黎曼流形 + 切空间是 **统一框架**。
- **欧式补丁**：平直空间算完再约束；**流形原生**：合法性是空间定义的一部分。

## 对 wiki 的映射

- [riemannian-manifold-tangent-space](../../wiki/formalizations/riemannian-manifold-tangent-space.md)（本次升格主页面）
- [shenlan-embodied-ai-fundamentals-series](../../wiki/overview/shenlan-embodied-ai-fundamentals-series.md)
- [lie-group-rigid-body-motions](../../wiki/formalizations/lie-group-rigid-body-motions.md)、[3d-coordinate-transforms-vision-robotics](../../wiki/formalizations/3d-coordinate-transforms-vision-robotics.md)

## 可信度与使用边界

- 偏 **工程科普 + 近似清单**；严格证明见微分几何教材与 *Modern Robotics*。
- 文内部分公式在抓取/OCR 中缺符号，wiki 以标准记号补全。

## 当前提炼状态

- [x] Agent Reach + Camoufox 抓取与归纳
- [x] 流形/切空间/Exp-Log/近似表与落地场景
- [x] wiki 主页面与专栏父节点映射确认
