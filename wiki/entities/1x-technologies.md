---
type: entity
tags: [hardware, humanoid, industry, teleoperation]
status: complete
updated: 2026-05-07
related:
  - ./humanoid-robot.md
  - ./figure-ai.md
  - ../queries/humanoid-hardware-selection.md
  - ../tasks/loco-manipulation.md
sources:
  - ../../sources/repos/1x-technologies.md
summary: "1X Technologies（前身为 Halodi Robotics）是一家挪威与美国双总部的通用人形机器人公司，产品覆盖轮式人形 EVE 与面向家庭场景的双足 NEO，强调真实世界部署数据与 AI 能力迭代。"
---

# 1X Technologies

## 一句话定义

**1X Technologies** 专注于「能在真实环境里长期运行的人形机器人」，当前公开产品线以 **轮式人形 EVE**（面向仓储 / 安防 / 医疗等结构化场景）与 **双足 NEO**（强调家庭与通用场景）为主轴。

## 为什么重要

- **硬件形态分叉**：同时押注「轮式 + 上半身人形」与「双足全身人形」，反映了商业化上对续航、通过性与成本的权衡。
- **数据闭环叙事**：家庭场景产品常与大规模真实交互数据、模仿学习与遥操作数据采集联系在一起（具体能力与时间表以官方为准）。
- **产业节点**：与美国人形赛道（Figure、Tesla、Agility 等）并行，代表欧洲与美国西海岸混合供应链路线的一支独立力量。

## 产品与路线（归纳）

| 系列 | 典型定位 | 备注 |
|------|-----------|------|
| **EVE** | 机构 / 工业场景的轮式人形 | 强调可部署性与导航–操作组合 |
| **NEO** | 家庭与通用场景的双足平台 | Beta / Gamma 等代际迭代见官网；面向消费者的交付与条款变化较快 |

## 常见误区或局限

- **不要把融资报道当规格书**：估值与融资轮次与整机参数无直接对应关系。
- **学术可得性**：与 Unitree 等科研平台相比，1X 更偏产品与试点部署，论文级 URDF / SDK 开放度需单独核实。
- **自主 vs 遥操作**：公开演示中常见 VR 遥操作与高监督采集；「完全自主居家」仍是整个行业未解决的问题。

## 关联页面

- [人形机器人](./humanoid-robot.md)
- [Figure AI](./figure-ai.md)（美国人形与 VLA 路线的可比节点）
- [Query：人形机器人硬件怎么选](../queries/humanoid-hardware-selection.md)
- [Loco-Manipulation](../tasks/loco-manipulation.md)

## 参考来源

- [1X Technologies 原始资料](../../sources/repos/1x-technologies.md)

## 推荐继续阅读

- [1X 官网 About](https://1x.tech/about)
- [IEEE Robots — EVE](https://robots.ieee.org/robots/eve)
