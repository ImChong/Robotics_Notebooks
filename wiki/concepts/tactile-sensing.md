---
type: concept
tags: [perception, manipulation, contact-rich, hardware, sensing]
status: complete
updated: 2026-04-21
related:
  - ./contact-rich-manipulation.md
  - ../tasks/manipulation.md
  - ../methods/visual-servoing.md
  - ../queries/tactile-feedback-in-rl.md
sources:
  - ../../sources/papers/contact_dynamics.md
summary: "触觉感知（Tactile Sensing）使机器人能够测量接触面上的法向力和切向力、滑动分布及材质纹理，是实现接触丰富操作和高精度抓取的核心感官。"
---

# Tactile Sensing（触觉感知）

**触觉感知 (Tactile Sensing)** 是机器人感知系统中的重要组成部分。如果说视觉（Vision）赋予了机器人远距离和全局的场景理解能力，那么触觉则是机器人与物理世界发生**直接物理交互**时的“兜底”防线。对于接触丰富（Contact-rich）的操作任务，如插拔、灵巧抓取、装配和盲索，触觉感知是不可或缺的。

## 核心感知维度

触觉传感器通常用来测量和估计以下物理量：

1. **三维接触力 (3D Contact Force)**：接触点处的法向压力（Normal Force）和切向摩擦力（Shear Force）。
2. **接触几何分布 (Contact Geometry)**：物体在传感器表面的接触面积、形状和压力分布图。
3. **滑移检测 (Slip Detection)**：通过高频振动或压力中心的快速移动，检测抓取物是否正在滑脱。
4. **材质纹理 (Texture/Compliance)**：通过拖动传感器表面，感知物体的粗糙度和软硬程度。

## 主流传感器技术路线

随着机器人灵巧手的普及，触觉传感器的形态也迎来了爆发：

### 1. 基于视觉的触觉传感器 (Vision-based Tactile Sensors)
- **代表**：GelSight, DIGIT.
- **原理**：在弹性硅胶体内部嵌入一个微型摄像头。当硅胶表面与物体接触发生形变时，摄像头捕捉其内表面的形变图像。
- **优点**：极高的空间分辨率（千万像素级），能捕捉极其精细的纹理和法向深度分布。非常适合直接与端到端（End-to-End）基于视觉的强化学习策略结合。
- **缺点**：体积较大（难以塞入指尖），帧率受限于摄像头（通常在 30-60Hz），存在盲区。

### 2. 电阻/电容式阵列 (Piezoresistive / Capacitive Arrays)
- **代表**：BioTac, 各种柔性薄膜阵列。
- **原理**：利用导电聚合物或电容器阵列，当受到压力时，电阻或电容值发生变化。
- **优点**：易于做成柔性贴片包裹在机械手上，成本较低。
- **缺点**：存在迟滞现象（Hysteresis），容易受到温度干扰，长期使用会老化漂移。

### 3. 多轴力矩传感器 (Multi-axis Force/Torque Sensors, F/T)
- **原理**：通常安装在机械臂的手腕处（Wrist F/T）或指尖内部（如应变片构成的六维力传感器）。
- **优点**：测量极其精准，带宽极高（可达 1000Hz+），是工业阻抗控制（Impedance Control）的标配。
- **缺点**：只能提供一个合力/合力矩点，无法提供接触面上的空间分布信息。

## 在机器人学习中的应用

在最新的 Robot Learning 研究中，触觉反馈正在成为突破视觉遮挡（Occlusion）瓶颈的关键：

- **跨模态融合 (Cross-modal Fusion)**：将视觉 RGB 特征与 GelSight 图像特征在 Transformer 晚期融合，使得模型在“手遮挡了目标”时，依然能依靠触觉完成插入孔位的微调。
- **作为奖励信号 (Reward Signal)**：在强化学习中，将“维持特定的法向压力范围”且“切向力不超过摩擦锥（Friction Cone）”作为稠密奖励，引导策略学会稳定的抓取。

## 关联页面
- [接触丰富操作 (Contact-Rich Manipulation)](./contact-rich-manipulation.md)
- [Manipulation 任务](../tasks/manipulation.md)
- [Visual Servoing (视觉伺服)](../methods/visual-servoing.md)
- [Friction Cone (摩擦锥) 形式化](../formalizations/friction-cone.md)

## 参考来源
- Yuan, W., et al. (2017). *GelSight: High-resolution robot tactile sensors for estimating geometry and force*.
- [sources/papers/contact_dynamics.md](../../sources/papers/contact_dynamics.md)
