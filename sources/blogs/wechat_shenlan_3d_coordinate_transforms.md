# 彻底搞懂具身智能的「方向感」：三维世界坐标变换

> 来源归档（blog / 微信公众号）

- **标题：** 彻底搞懂具身智能的"方向感"：三维世界坐标变换
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号；《具身智能基础》专栏第 2 篇）
- **原始链接：** https://mp.weixin.qq.com/s/P5Jm7bMhaTHsytHStFbbLg
- **发表日期：** 2026-05-28
- **入库日期：** 2026-06-04
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0 + `wechat-article-for-ai`（Camoufox）；正文约 0.77 万字 / 23 图；Jina Reader 对该链接触发微信 CAPTCHA，未采用
- **专栏姊妹篇：** [李群、李代数、四元数](wechat_shenlan_lie_group_lie_algebra_quaternion.md)（`JviRH2LW-fkCHA5gY7Qflw`）；[黎曼流形与切空间](wechat_shenlan_riemannian_manifold_tangent_space.md)（`uFTKN5FDvlHQxOSspvxVZw`）
- **一句话说明：** 从四大坐标系（世界 / 相机 / 图像 / 像素）与针孔模型出发，串联内参 $K$、外参 $[R|t]$、深度恢复（双目 / RGB-D）与手眼标定（Eye-in-Hand / Eye-to-Hand），解释具身智能里「看见」≠「理解」≠「能抓」的三条坐标链路，以及 VLA 真机抓空常源于坐标系未对齐的底层原因。

## 核心摘录（归纳，非全文）

### 问题重框

- 传统机器人学中 DH / 齐次变换 / 正逆运动学已「解决」重复定位；具身时代瓶颈变为 **感知输入不确定**（相机、机械臂基座、物体世界系三套语言须统一）。
- 工业场景：方程组输入多为已知常量；具身场景：外参、深度、物体位姿常从 **高维感知推断**，带噪声与歧义。

### 四大坐标系

| 坐标系 | 原点 / 轴 | 单位 | 角色 |
|--------|-----------|------|------|
| **世界** | 环境固定点（如机器人底座） | m | 全局绝对参考 |
| **相机** | 光心；Z 沿光轴 | m | 视角相关 3D |
| **图像** | 主点（光轴与成像面交点） | mm | 物理成像平面 |
| **像素** | 图像左上角 | pixel | 离散图像索引 |

### 正向投影链

世界点 $P_w$ → 外参 $[R|t]$ → 相机坐标 $P_c$ → 内参 $K$ → 像素 $(u,v)$：

$$
Z_c \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K [R|t] \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}
$$

- **内参 $K$**：焦距 $f_x,f_y$、主点 $(c_x,c_y)$、像素尺寸 $d_x,d_y$；由标定固定。
- **外参**：相机在世界中的位姿（旋转 $R$ + 平移 $t$）；刚体变换，形状不变。

### 逆向与深度

- 单目 RGB 投影 **丢失深度**：同射线多点共像。
- 恢复手段：**双目视差** $z = fB/(x-x')$；**RGB-D**（结构光 / ToF）。

### 手眼标定

- **Eye-in-Hand**：相机在末端，$X$ 随臂动。
- **Eye-to-Hand**：相机固定，$X$ 一次标定长期使用。
- 核心方程 $AX=XB$（臂运动 $A$、标定板观测 $B$、待求手眼 $X$）。

### 三条闭环（抓取水杯）

1. **感知链**：世界 → 外参 → 相机 → 内参 → 像素（看见）
2. **理解链**：像素 + 深度 → 反投影 → 相机 3D 点云（理解）
3. **操作链**：手眼标定 → 基座坐标 → 运动指令（能抓）

### 与 VLA / 世界模型的关系（文内观点）

- 仿真丝滑、真机抓空：常见原因之一是 **「相机前方 10 cm」≠「机械臂前方 10 cm」**；坐标变换是算法栈最底层、也最易被大模型掩盖的暗线。

## 对 wiki 的映射

- [3d-coordinate-transforms-vision-robotics](../../wiki/formalizations/3d-coordinate-transforms-vision-robotics.md)（本次升格主页面）
- [shenlan-embodied-ai-fundamentals-series](../../wiki/overview/shenlan-embodied-ai-fundamentals-series.md)（专栏父节点）
- [lie-group-rigid-body-motions](../../wiki/formalizations/lie-group-rigid-body-motions.md)、[se3-representation](../../wiki/formalizations/se3-representation.md)
- [grasp-pose-estimation](../../wiki/methods/grasp-pose-estimation.md)、[vla](../../wiki/methods/vla.md)

## 可信度与使用边界

- 公众号为 **工程直觉 + 标定流程导航**；公式符号以抓取版与教材为准。
- 文内商务/约稿信息已剥离；图在微信 CDN，wiki 用 Mermaid 复述链路。

## 当前提炼状态

- [x] Agent Reach + Camoufox 正文抓取与归纳摘要
- [x] 四大坐标系、内外参、深度、手眼标定与三条链路
- [x] wiki 主页面与专栏父节点映射确认
