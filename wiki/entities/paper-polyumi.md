---
type: entity
tags: [paper, manipulation, tactile, audio, imitation-learning, data-collection]
status: complete
updated: 2026-09-26
arxiv: "2609.29760"
code: https://github.com/polyumi/PolyUMI-platform
related:
  - ../overview/embodied-research-12-papers-technology-map.md
  - ../methods/vla.md
  - ../tasks/manipulation.md
sources:
  - ../../sources/papers/polyumi_arxiv_2609_29760.md
  - ../../sources/repos/polyumi-platform.md
  - ../../sources/blogs/wechat_embodied_station_12_papers_research_checklist_2026-09-26.md
summary: "PolyUMI（arXiv:2609.29760）：手持 VisTA 同步视触听+本体采集并迁移感知手指；滑移控制视触听 8/10 vs 纯视觉 2/10（80 条示范）；polyumi/PolyUMI-platform 已开源。"
---

# PolyUMI

**PolyUMI**（*Accessible Visual-Tactile-Audio Data Collection for Object Inference and Manipulation*，[arXiv:2609.29760](https://arxiv.org/abs/2609.29760)，[代码](https://github.com/polyumi/PolyUMI-platform)，[项目页](https://polyumi-vista.github.io/)）收录自 [具身智能小站 12 篇清单](../../sources/blogs/wechat_embodied_station_12_papers_research_checklist_2026-09-26.md)。

## 一句话定义

**把接触音频与光学触觉与腕部视觉、本体同步采进示范，并用 token 级 VisTA 融合，让采集端接触线索能进机器人执行端。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉-语言-动作策略 |
| IL | Imitation Learning | 模仿学习 |
| SR | Success Rate | 任务成功率 |
| WM | World Model | 世界模型 |

## 为什么重要

- 纳入 [12 篇具身研究清单](../../wiki/overview/embodied-research-12-papers-technology-map.md) 主线，与同期 VLA / 接触 / 规划 / 安全论文可横向对照。
- 公众号强调的可操作读法：先看 **任务信息需求**（如 PolyUMI 旋灯泡仍以视觉最优）与 **评测口径**（如 Self-Adaptive 多 trial、BeyondRetarget 仿真片段非真机 SR）。
- 开源状态（步骤 2.5）：**已开源**。

## 核心信息

| 项 | 内容 |
|----|------|
| **arXiv** | [2609.29760](https://arxiv.org/abs/2609.29760) |
| **项目页** | https://polyumi-vista.github.io/ |
| **代码** | https://github.com/polyumi/PolyUMI-platform |
| **开源** | **已开源** |

## 实验与评测（公众号口径）

| 任务 / 设定 | 文内要点 |
|-------------|----------|
| 物体推断 / 擦板 / 旋灯泡 | 覆盖 VisTA 多模态融合与 **感知手指** 迁移 |
| 螺丝刀 **滑移控制** | 视+触+听 **8/10** vs 纯视觉 **2/10**（**80** 条示范，规模较小） |
| 旋灯泡 | **视觉策略最好**——多模态并非每项都赢 |
| 采集 | 腕部图像 + 光学触觉 + **接触音频** + 本体同步 |

- 读法：按任务 **信息需求** 选模态，而不是默认「模态越多越好」。

## 源码运行时序图

```mermaid
sequenceDiagram
  autonumber
  participant U as 用户
  participant R as 官方仓库
  participant M as 训练/推理入口
  participant E as 仿真或真机
  U->>R: clone + 依赖安装
  U->>M: 配置与权重
  M->>E: rollout / 控制
  E-->>U: 指标日志
```


## 结论

**总判：PolyUMI 适合作为「把接触音频与光学触觉与腕部视觉、本体同步采进示范，并用 token 级 VisT…」方向的入口页；细节以 arXiv 与项目页为准。**

1. 与 [12 篇技术地图](../overview/embodied-research-12-papers-technology-map.md) 对照选型，避免与同名不同 arXiv 的工作混淆（如 RAPID vs RAPID-VLM-RL）。
2. 开源为 **已开源** 时优先从项目页 Code 区核实，再写复现计划。
3. 长程 / 部署类条目（AdaHVLA、HarnessPAI、Self-Adaptive VLA）同时记录 **成功率定义** 与 **失败恢复预算**。

## 与其他工作对比

| 对照 | 差异读法 |
|------|----------|
| [UMI](./paper-rcl-ref-e064f89fc62c8df5da9f-universal-manipulation-interface-in-the-wild-rob.md) | 手持夹爪示范以腕部视觉 + 本体为主；PolyUMI 加 **光学触觉 + 接触音频**，无线、无需系留工作站 |
| [HiFi-UMI](./paper-hifi-umi.md) | 追视觉保真、时间同步与采集规模；PolyUMI 追 **接触模态**，且感知手指可直接迁到机器人末端、保持采集与执行的传感几何一致 |
| [DexUMI](./paper-notebook-dexumi-using-human-hand-as-the-universal-manipul.md) | 外骨骼式人手接口面向灵巧手；PolyUMI 是夹爪式平台 |
| [PROPRA](./paper-propra-fingertip-anchoring.md) | 同期指尖传感工作：PROPRA 攻「稀疏、分相位信号难从少量示范学」的表征预训练；PolyUMI 攻采集硬件 + token 级 VisTA 融合策略 |
| [OmniTacTune](./paper-omnitactune-tactile-residual-adaptation.md) | 冻结视觉基策略、用触觉残差在线 RL 修正；VisTA 直接从多模态示范学习接触感知动作 |

## 关联页面

- [具身研究 12 篇技术地图](../overview/embodied-research-12-papers-technology-map.md)
- [Manipulation](../tasks/manipulation.md)
- [VLA](../methods/vla.md)

## 参考来源

- [polyumi 论文归档](../../sources/papers/polyumi_arxiv_2609_29760.md)
- [公众号 12 篇清单](../../sources/blogs/wechat_embodied_station_12_papers_research_checklist_2026-09-26.md)

## 推荐继续阅读

- [arXiv:2609.29760](https://arxiv.org/abs/2609.29760)
- [项目页](https://polyumi-vista.github.io/)
