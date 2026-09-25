---
type: entity
tags:
  - paper
  - vla
  - distillation
  - edge-deployment
  - iros-2026
status: complete
updated: 2026-09-25
arxiv: "2601.20262"
code: https://icsl-jeon.github.io/shallow-pi/
related:
  - ../methods/vla.md
  - ../tasks/manipulation.md
  - ../queries/vla-deployment-guide.md
  - ./paper-geovla.md
  - ../overview/iros-2026-six-trends-technology-map.md
sources:
  - ../../sources/papers/shallow_pi_arxiv_2601_20262.md
  - ../../sources/sites/shallow-pi.md
  - ../../sources/blogs/wechat_ai_tech_review_iros_2026_six_trends_2026-09-25.md
summary: "Shallow-π（arXiv:2601.20262，IROS 2026 最佳论文候选）：flow-based VLA 层蒸馏 18→6；>2× 推理加速、SR 绝对降幅 <1%；Jetson Orin/Thor 真机；代码待发布。"
---

# Shallow-π：Flow-based VLA 的层蒸馏

**Shallow-π**（*Knowledge Distillation for Flow-based VLAs*，[arXiv:2601.20262](https://arxiv.org/abs/2601.20262)，[项目页](https://icsl-jeon.github.io/shallow-pi/)）由 **42dot、首尔大学、Samsung Research** 提出（**IROS 2026** Best Paper / Best Student Paper 候选）：对 **π 类 flow-based VLA** 的 **VLM 骨干与 flow 动作头** 同时做知识蒸馏，将 transformer 深度 **18 → 6 层**，在标准操作基准上 **>2× 推理加速** 且成功率 **绝对降幅 <1%**。

## 一句话定义

**VLA 部署瓶颈不只在 token 数——Shallow-π 用层蒸馏把 flow 动作头与 VLM 一起压浅，换边缘设备上的实时闭环。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| VLA | Vision-Language-Action | 视觉–语言–动作策略 |
| KD | Knowledge Distillation | 知识蒸馏 |
| FM | Foundation Model | 大视觉–语言骨干 |
| SR | Success Rate | 任务成功率 |
| Orin | NVIDIA Jetson Orin | 边缘 GPU 平台 |

## 为什么重要

- [IROS 2026 六趋势](../../sources/blogs/wechat_ai_tech_review_iros_2026_six_trends_2026-09-25.md) 将 VLA 主线从「证明能做」转向 **效率与系统部署**；Shallow-π 是 **Best Paper 候选** 的效率代表。
- 在 **Jetson Orin / Thor** 与人形等多平台上做工业级真机验证，衔接 [VLA 部署指南](../queries/vla-deployment-guide.md)。
- **待发布** 官方代码/权重（步骤 2.5，2026-09-25）。

## 核心机制

- **问题：** flow-based VLA 动作头 **每层** 与 VLM 交叉注意力；传统 layer-skipping 对 π 类结构无效。
- **方法：** 全中间层注入 conditioning 的 **联合蒸馏**，同时压缩 VLM 与动作头深度。
- **结果：** 18→6 层、**>2×** 更快；操作基准 SR **绝对 <1%** 下降；Orin 上 ~**10 Hz** 端到端（文内/项目页口径）。

## 实验与评测

| 维度 | 报告 | 读法 |
|------|------|------|
| 仿真操作基准 | SR 降幅 <1% vs 教师 | 与 token 剪枝类方法比 **层数** 维度 |
| 边缘真机 | Orin/Thor，多机器人含人形 | 关注 **延迟–精度** 曲线而非单点 SR |
| 开源 | 项目页 + arXiv | 复现待官方权重 |

## 与其他工作对比

| 对照 | 差异 |
|------|------|
| Visual token 剪枝 | Shallow-π 动 **transformer 深度** |
| [GeoVLA](./paper-geovla.md) | 补 **3D 几何**；Shallow-π 补 **推理速度**（可叠加问题） |
| Shallow-π₀.₅（其他工作） | 不同蒸馏对象与 π 变体；勿混为同一方法 |

## 结论

**Shallow-π 说明 IROS 2026 上 VLA 竞争维度已含「边缘实时」：蒸馏必须尊重 flow 动作头与 VLM 的层间耦合。**

1. Best Paper 候选 + 多平台真机，适合作为 **部署向** 索引页。
2. **待发布** 代码前，仅可引用 arXiv/项目页数字。
3. 与几何增强（GeoVLA）正交：先决定 **快** 还是 **准（3D）**，再谈组合。
4. 人形/动态场景结果提示：蒸馏学生仍须保留 **跨层 conditioning**，不能简单跳层。

## 源码运行时序图

不适用：截至 2026-09-25 无官方 GitHub 训练/推理仓库；待发布后按「教师 π 模型 → 联合层蒸馏 → 6 层学生 → Jetson 部署」补序图。

## 关联页面

- [VLA](../methods/vla.md)
- [IROS 2026 六趋势地图](../overview/iros-2026-six-trends-technology-map.md)

## 参考来源

- [shallow_pi_arxiv_2601_20262.md](../../sources/papers/shallow_pi_arxiv_2601_20262.md)
- [wechat_ai_tech_review_iros_2026_six_trends_2026-09-25.md](../../sources/blogs/wechat_ai_tech_review_iros_2026_six_trends_2026-09-25.md)

## 推荐继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2601.20262)
- [Shallow-π 项目页](https://icsl-jeon.github.io/shallow-pi/)
