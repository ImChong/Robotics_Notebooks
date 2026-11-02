---
type: overview
tags: [overview, vla, history, robot-learning, technology-map, x-article]
status: complete
updated: 2026-11-02
related:
  - ../entities/paper-saycan.md
  - ../entities/paper-rt-1.md
  - ../entities/paper-rt-2.md
  - ../entities/paper-open-x-embodiment.md
  - ../methods/vla.md
  - ../methods/action-chunking.md
sources:
  - ../../sources/blogs/x_starlitmatcha_vla_history_rt1_to_molmoact2_2026-11-01.md
summary: "Khushi 长文《From RT1 to MolMoAct2》的阅读坐标：按时间线把 SayCan、RT-1、RT-2、Open X-Embodiment、Octo、OpenVLA、π0、SmolVLA、MolmoAct、MolmoAct 2 十个模型对应到唯一节点，并梳理动作表示的三条路线。"
---

# VLA 演进简史：从 SayCan / RT-1 到 MolmoAct 2（Khushi 长文阅读坐标）

> **本页定位**：为 [Khushi（@starlitmatcha）的 X Article](https://x.com/starlitmatcha/status/2116918927913804223)（2026-11-01）提供按时间线组织的阅读坐标；文中 10 个模型各有唯一详情节点（新建 2 / 复用 8）。数字与论断为作者表述，各节点另有论文核对。

## 一句话观点

**作者的结论：没有单一赢家架构——不同设计在泛化、动作精度、推理延迟与所需机器人数据之间取舍；demo 越来越惊艳，但「做成一次」与「在不受控环境里可靠做成」之间仍有很大差距。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作模型 |
| VLM | Vision-Language Model | 视觉语言模型 |
| RT | Robotics Transformer | Google 的机器人 Transformer 系列 |
| OXE | Open X-Embodiment | 跨机器人开放数据集合作 |
| ARM | Action Reasoning Model | MolmoAct 提出的动作推理模型 |

## 时间线与节点

| # | 模型 | 节点 |
|---|------|------|
| 1 | SayCan | [paper-saycan](../entities/paper-saycan.md)（复用） |
| 2 | RT-1 | [paper-rt-1](../entities/paper-rt-1.md)（复用） |
| 3 | RT-2 | [paper-rt-2](../entities/paper-rt-2.md)（复用） |
| 4 | Open X-Embodiment / RT-X | [paper-open-x-embodiment](../entities/paper-open-x-embodiment.md)（复用） |
| 5 | Octo | [paper-octo](../entities/paper-octo.md)（复用） |
| 6 | OpenVLA | [paper-openvla](../entities/paper-openvla.md)（复用） |
| 7 | π0 | [paper-pi0](../entities/paper-pi0.md)（复用） |
| 8 | SmolVLA | [paper-smolvla](../entities/paper-smolvla.md)（新建） |
| 9 | MolmoAct | [paper-molmoact](../entities/paper-molmoact.md)（新建） |
| 10 | MolmoAct 2 | [paper-molmoact2](../entities/paper-molmoact2.md)（复用） |

## 作者的叙事线

- **为什么需要 VLA**：机器人在条件稍变（如料筐挪两英寸）就失效；VLM 能理解但不能动，经典控制能动但不泛化。VLA 的洞见：给 VLM 增加动作输出，用机器人示范端到端训练。
- **VLA 如何工作**：相机图像经视觉编码器成视觉 embedding，指令分词成文本 embedding，许多模型还输入关节角、夹爪开度等状态；Transformer 联合处理后由动作头预测末端位移、旋转与开合——可以是离散动作 token，也可以是连续值或 diffusion / flow matching，常一次预测多步动作 chunk；控制器执行后再观测、循环。
- **动作表示的三种路线**：离散动作 token（RT-1、RT-2、OpenVLA）→ 扩散出的连续动作 chunk（Octo）→ VLM + flow matching 动作专家（π0、SmolVLA、MolmoAct 2）。
- **开放问题（作者观点）**：(1) 示范只能教起点，机器人也应从自身错误中学习，VLA 与 RL 是互补而非对立；(2) 把 VLA 与世界模型结合，预测动作后果、比较候选动作；作者承认预测不准与额外计算延迟是风险。

## 动作表示三条路线（作者归纳）

| 路线 | 代表 | 取舍（作者表述） |
|------|------|------------------|
| 离散动作 token | RT-1（256 bins，约 3 Hz）、RT-2、OpenVLA | 把动作当语言，简单；但量化范围与分辨率相互牵制（见来源页的离散化实验） |
| 扩散连续动作 chunk | Octo | 可表达多模态可行动作，避免均值化；Transformer 只跑一次、迭代去噪在小动作头内 |
| VLM + flow-matching 动作专家 | π0、SmolVLA、MolmoAct 2 | 兼得 VLM 语义与连续高频动作；每个 chunk 仍需多步精炼，chunk 内动作不一定用到新观测 |

## 关联页面

- [VLA 方法总览](../methods/vla.md)
- [动作分块 Action Chunking](../methods/action-chunking.md)
- [SayCan](../methods/saycan.md)
- [RT 系列](../methods/robotics-transformer-rt-series.md)
- [Octo 模型](../methods/octo-model.md)

## 参考来源

- [x_starlitmatcha_vla_history_rt1_to_molmoact2_2026-11-01.md](../../sources/blogs/x_starlitmatcha_vla_history_rt1_to_molmoact2_2026-11-01.md)
- [原推文](https://x.com/starlitmatcha/status/2116918927913804223)
