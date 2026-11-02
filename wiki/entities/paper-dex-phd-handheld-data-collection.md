---
type: entity
tags:
  - paper
  - humanoid
status: complete
updated: 2026-11-02
arxiv: "2610.35462"
project_id: dex-phd-handheld-data-collection
related:
  - ../overview/senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map.md
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
sources:
  - ../../sources/blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md
  - ../../sources/papers/dex-phd-handheld-data-collection_arxiv_2610_35462.md
  - ../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md
summary: "DEX-PHD：面向多模态数据采集的便携式灵巧手持设备（arXiv:2610.35462）：将灵巧机器人手、逐指控制、腕部相机、关节状态和指尖接触传感集成到无系留手持设备；采集的数据可在实体人形机器人上直接开环回放。"
---

# DEX-PHD：面向多模态数据采集的便携式灵巧手持设备

**DEX-PHD**（*DEX-PHD: A Dexterous Portable Handheld Device for Multimodal Data Collection*，[arXiv:2610.35462](https://arxiv.org/abs/2610.35462)）收录于 senlanke「具身运控lab」10-26–10-30 前沿论文动态（人形/四足）。本页依据**周报条目 + arXiv 摘要页**整理，流程图依据论文方法部分（见参考来源），未逐页核对实验数字；指标与结论以论文为准。

## 一句话定义

**DEX-PHD：面向多模态数据采集的便携式灵巧手持设备——将灵巧机器人手、逐指控制、腕部相机、关节状态和指尖接触传感集成到无系留手持设备；采集的数据可在实体人形机器人上直接开环回放。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 强化学习 |
| WBC | Whole-Body Control | 全身控制 |

## 为什么重要

- 周报所述待解决的问题：机器人运动链遥操作能够准确记录状态，但不便携；纯视觉采集便携，却缺少准确的手指状态和接触数据。

## 流程总览

```mermaid
flowchart TD
  A["操作者握持 DEX-PHD<br/>扳机(食/中/无名/小指/拇指) + 拇指旋钮"] --> B["按会话前标定的上下限归一化<br/>线性映射到机械手关节范围"]
  B --> C["Ability Hand 执行<br/>(Teensy 4.0, 100 Hz 控制与记录)"]
  C --> D["microSD 记录: 指令输入、关节位置、<br/>电流、30 个 FSR 指尖力"]
  A --> E["腕部相机 (GoPro) 录 RGB + IMU"]
  E --> F["视觉 SLAM (类 UMI 管线)<br/>得到腕部/末端轨迹"]
  D --> G["同步: 拇指扳机 500 ms 内双击触发 LED<br/>后处理按 LED 闪烁对齐手部数据与视频"]
  F --> G
  G --> H["各平台逆运动学 (Pinocchio)<br/>关节限位 + 速度/加速度限幅 + 初始位姿偏置"]
  H --> I["开环回放: 仿真双臂 / Unitree G1 humanoid 实机"]
  I --> J["对比示教与回放:<br/>关节位置 MAE、相关性、接触相关性"]
```

图依据：论文 §III-A–D（Fig.2）、§IV-B–D（Fig.4–5）

## 核心机制

| 项 | 内容 |
|----|------|
| **arXiv** | [2610.35462](https://arxiv.org/abs/2610.35462) · [PDF](https://arxiv.org/pdf/2610.35462) |
| **作者（arXiv）** | Hong, Jooyoung, Myers, Noboru, Kwon, Obin, Yang, Yixian, Kim, Joohyung |
| **机构（周报所述，未逐条核实）** | 伊利诺伊大学厄巴纳—香槟分校 KIMLAB |
| **arXiv 评论栏** | （无） |
| **开源** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02） |
| **周报所述创新点** | 将灵巧机器人手、逐指控制、腕部相机、关节状态和指尖接触传感集成到无系留手持设备；采集的数据可在实体人形机器人上直接开环回放。 |

### arXiv 摘要节选（英文原文）

> Existing manipulation data collection methods typically offer either accurate robot-state acquisition through kinematic-chain-based teleoperation or convenient, portable operation through vision-based approaches. We present DEX-PHD, a portable handheld device that combines these advantages for dexterous manipulation by integrating a dexterous robot hand, continuous per-finger control, and multimodal logging.

## 源码运行时序图

**不适用**：arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布），无可对齐的官方代码流程。

## 实验与评测

- 周报与摘要给出的数字/场景已写入上表「创新点」与摘要节选；其余实验细节、基线与成功率定义请回到论文正文。
- 读法：先对齐任务设定、传感器与成功定义（仿真/真机），再解读 headline 数字，勿跨论文直接比较。

## 与其他工作对比

| 维度 | 读法 |
|------|------|
| **同周对照** | 见 [人形/四足周报技术地图](../overview/senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map.md)，按方向浏览同期论文 |
| **开源状态** | arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布） |

## 结论

**DEX-PHD：面向多模态数据采集的便携式灵巧手持设备。**

1. 本页是入库级摘要，复现前请核对论文正文、附录与代码仓库。
2. 指标须连同实验条件解读。
3. 关注 arXiv 版本更新与代码发布。

## 关联页面

- [senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map](../overview/senlanke-weekly-2026-10-26-30-humanoid-quadruped-technology-map.md)
- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)

## 参考来源

- [dex-phd-handheld-data-collection_arxiv_2610_35462.md](../../sources/papers/dex-phd-handheld-data-collection_arxiv_2610_35462.md)
- [wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md](../../sources/blogs/wechat_senlanke_weekly_humanoid_quadruped_2026-10-26_30.md)
- [arXiv:2610.35462](https://arxiv.org/abs/2610.35462)
- [arXiv PDF](https://arxiv.org/pdf/2610.35462)：已通读方法部分（实验与附录以略读为准）：§I–§V（§III 硬件/关节映射/轨迹生成/同步，§IV 演示、IK 与实机回放）

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2610.35462)
