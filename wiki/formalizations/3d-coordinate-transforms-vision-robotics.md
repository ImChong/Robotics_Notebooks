---
type: formalization
tags: [kinematics, computer-vision, calibration, hand-eye, camera-model, embodied-ai, shenlan]
status: complete
updated: 2026-06-04
related:
  - ../overview/shenlan-embodied-ai-fundamentals-series.md
  - ./lie-group-rigid-body-motions.md
  - ./se3-representation.md
  - ../methods/grasp-pose-estimation.md
  - ../methods/visual-servoing.md
  - ../methods/vla.md
  - ../entities/april-tag.md
sources:
  - ../../sources/blogs/wechat_shenlan_3d_coordinate_transforms.md
  - ../../sources/raw/wechat_shenlan_3d_coord_transforms_2026-06-04.md
summary: "具身抓取与 VLA 部署的底层暗线：世界 / 相机 / 图像 / 像素四系经内参 K 与外参 [R|t] 串联；单目丢深度须双目或 RGB-D 反投影；手眼标定闭合「看见→理解→能抓」三链。相机前方 10 cm 与机械臂前方 10 cm 常不是同一方向。"
---

# 三维坐标变换（视觉–机器人对齐）

**一句话：** 机器人要把「相机里的一团像素」变成「机械臂能执行的位姿」，必须在 **世界、相机、成像平面、像素** 四套坐标之间做可审计的刚体与投影变换；具身时代难的不是解方程，而是 **外参、深度与手眼关系从不确定感知中估计**。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作多模态基础策略方向 |
| RGB | Red-Green-Blue | 彩色图像通道，常与深度 (RGB-D) 配合 |

## 为什么「已解决」的问题又变成瓶颈

| 时代 | 输入特征 | 典型能力 |
|------|----------|----------|
| **传统工业机器人** | 几何与标定常量为已知 | 微米级重复定位 |
| **具身智能** | 物体位姿、外参、深度来自学习/传感器噪声 | 抓空、系统性偏移、仿真–真机不一致 |

当 **相机系**、**基座系**、**物体世界系** 未统一时，杯子在图像里「看起来对」，在基座系里仍可能差一个固定或随臂变的变换——VLA 策略再强也无法补偿。

## 四大坐标系

| 坐标系 | 原点 / 轴约定 | 单位 | 作用 |
|--------|---------------|------|------|
| **世界** $\{W\}$ | 环境固定（如机器人底座） | m | 全局规划与多传感器融合 |
| **相机** $\{C\}$ | 光心；$Z$ 沿光轴 | m | 视角相关 3D |
| **图像** | 主点（光轴∩成像面） | mm | 连续成像平面 |
| **像素** | 通常左上角 | pixel | 离散图像索引 |

空间点 $P$ 的旅程：$P_W \xrightarrow{[R|t]} P_C \xrightarrow{K} (u,v)$。

## 针孔模型：内参与外参

### 内参 $K$（相机自身）

$$
K = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

- $f_x = f/d_x$，$f_y = f/d_y$：焦距与像素尺寸
- $(c_x,c_y)$：主点，常近图像中心
- **与相机位姿无关**；出厂或标定固定

透视投影（相机坐标 $(X_c,Y_c,Z_c)$）：

$$
Z_c \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K \begin{bmatrix} X_c \\ Y_c \\ Z_c \end{bmatrix}
$$

### 外参 $[R|t]$（相机在世界中）

$$
P_C = R\, P_W + t
$$

其中 $R \in \mathrm{SO}(3)$ 为 **世界系到相机系的旋转矩阵**（$R^\top R=I$，$\det R=1$），$t \in \mathbb{R}^3$ 为相机光心在世界系下的 **平移向量**。

齐次形式为 $4\times4$ **刚体变换**（与 [李群页](./lie-group-rigid-body-motions.md) 中 SE(3) 一致）：只改变位姿，不改变形状。

### 完整投影

$$
Z_c \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K [R|t] \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}
$$

$K[R|t]$ 为 $3\times4$ **投影矩阵**。

## 流程总览

```mermaid
flowchart LR
  PW["世界点 P_W"]
  PC["相机坐标 P_C"]
  UV["像素 (u,v)"]
  DEPTH["深度 Z 或视差"]
  PCLOUD["相机 3D 点云"]
  PB["基座坐标"]
  PW -->|"外参 [R|t]"| PC
  PC -->|"内参 K"| UV
  UV -->|"+ 深度"| DEPTH
  DEPTH --> PCLOUD
  PCLOUD -->|"手眼 X"| PB
```

## 深度丢失与恢复

**正向投影** 沿视线压扁维度：同一条射线上的点共像。

| 手段 | 原理 | 备注 |
|------|------|------|
| **双目** | $z = \dfrac{fB}{x-x'}$ | 基线 $B$、视差 $(x-x')$ |
| **RGB-D** | 结构光 / ToF 直接测距 | 标定与多传感器时间同步仍关键 |
| **单目 + 学习** | 网络估深度 | 须与 $K$、外参一致地反投影 |

**逆向链路**：$(u,v)$ + $Z$ + $K^{-1}$ → $P_C$ → $[R|t]^{-1}$ → $P_W$。

## 手眼标定

目标：求相机系与 **末端** 或 **基座** 系之间的固定（或随臂变的）变换 $X$。

| 模式 | 安装 | $X$ 的性质 |
|------|------|------------|
| **Eye-in-Hand** | 相机在末端 | 随臂运动，每步更新复合变换 |
| **Eye-to-Hand** | 相机固定在外部 | $X$ 一次标定，长期使用 |

经典方程（多姿态）：

$$
A X = X B
$$

$A$：末端两姿态间变换；$B$：标定板在相机下的观测变换；$X$：待求手眼关系。

与 [AprilTag](../entities/april-tag.md)、[grasp-pose-estimation](../methods/grasp-pose-estimation.md) 中「多视点融合需手眼」的表述一致。

## 三条链路（抓取闭环）

1. **感知链**：$P_W \to P_C \to (u,v)$ — 「看见」
2. **理解链**：$(u,v)+Z \to P_C \to$ 点云 — 「理解几何」
3. **操作链**：$P_C \xrightarrow{X} P_{\text{base}}$ — 「能抓」

任一环标定或深度错误，都会在末端表现为 **重复性抓偏**。

## 常见误区

1. **「仿真里点云对就够了」** — 真机外参漂移、时间戳不对齐仍会导致抓空。
2. **「忽略 Eye-in-Hand 的随臂变化」** — 把动态 $X$ 当常量用。
3. **「单目 RGB 直接回归 3D 无需 $K$」** — 网络输出须与标定一致的反投影才有物理意义。
4. **「手眼只做一次永远够用」** — 冲击、温漂、相机松动都会使 $X$ 失效。

## 关联页面

- [《具身智能基础》专栏地图](../overview/shenlan-embodied-ai-fundamentals-series.md)
- [李群、李代数与刚体旋转](./lie-group-rigid-body-motions.md) — 外参中的 $R,t$
- [黎曼流形与切空间](./riemannian-manifold-tangent-space.md) — 位姿优化统一语言
- [Grasp Pose Estimation](../methods/grasp-pose-estimation.md)
- [Visual Servoing](../methods/visual-servoing.md)
- [VLA](../methods/vla.md)

## 参考来源

- [深蓝具身智能：三维世界坐标变换（微信公众号归档）](../../sources/blogs/wechat_shenlan_3d_coordinate_transforms.md)
- [抓取落盘摘要](../../sources/raw/wechat_shenlan_3d_coord_transforms_2026-06-04.md)
- Lynch & Park, *Modern Robotics* — 相机与坐标系（见 [modern_robotics_textbook.md](../../sources/papers/modern_robotics_textbook.md)）

## 推荐继续阅读

- [OpenCV Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)（外部）
- Tsai & Lenz 手眼标定经典工作（Eye-in-Hand / Eye-to-Hand）
