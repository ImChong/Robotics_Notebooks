---
type: entity
tags: [repo, world-model, video-generation, robbyant, physical-ai]
status: complete
updated: 2026-09-19
related:
  - ./lingbot-vla.md
  - ./lingbot-vla-v2.md
  - ../methods/generative-world-models.md
  - ./paper-sa-2601-20540-advancing-open-source-world-models-lingbot-world.md
sources:
  - ../../sources/repos/lingbot-world-v2.md
  - ../../sources/blogs/wechat_robot_yanfa_opensource_algorithms_compendium.md
summary: "robbyant/lingbot-world：蚂蚁灵波开源世界模型主线（视频生成 + 高保真动力学）；文内 antgroup/lingbot 链接已校正。"
---

# LingBot-World

[**robbyant/lingbot-world**](https://github.com/robbyant/lingbot-world)（蚂蚁 **Robbyant / 灵波**）是 **开源世界模型** 主线仓库：从视频生成出发构建 **高保真、鲁棒动力学** 的交互式世界模拟器；后继版本见 [lingbot-world-v2](https://github.com/robbyant/lingbot-world-v2) 与论文 [Advancing Open-source World Models](./paper-sa-2601-20540-advancing-open-source-world-models-lingbot-world.md)。

## 一句话定义

**灵波世界模型代码入口** — 长时序视频–动作联合建模与开源世界模拟器；与 [LingBot-VLA](./lingbot-vla.md) 操纵基础模型分工。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| WM | World Model | 环境前向预测 / 模拟 |
| VLA | Vision-Language-Action | 姊妹线 LingBot-VLA |
| Physical AI | Physical Artificial Intelligence | 文内「具身大模型」语境 |
| MJCF | MuJoCo XML Format | 与机器人仿真资产可组合 |

## 为什么重要

- **微信清单校正：** 原文 `antgroup/lingbot` **404**；canonical 仓为 **robbyant** org 下 `lingbot-world` / `lingbot-world-v2`。
- **长时序任务：** 文章强调「长周期连续任务预测、视频–动作联合建模」，对应世界模型而非纯 VLA 推理。
- **与 VLA 栈并列：** 同一团队 [LingBot-VLA](./lingbot-vla.md) 负责操纵；World 负责 **预测 / 仿真**。

## 工程实践

1. 权重与推理脚本见 GitHub README 与 [HF 集合](https://huggingface.co/collections/robbyant/lingbot-world-v2)。
2. v2 栈依赖 torch≥2.4、flash-attn 等；按仓内 `requirements` 钉版本。
3. License 常为 **CC BY-NC-SA**（以 README 为准）。

## 局限与使用注意

- **非机器人低层控制器：** 不能直接替代 [rsl-rl](./rsl-rl.md) 步态训练。
- **org 易混：** 勿使用失效的 `antgroup/lingbot` 链接。

## 关联页面

- [LingBot-VLA](./lingbot-vla.md)
- [Generative World Models](../methods/generative-world-models.md)
- [LingBot-World 论文索引](./paper-sa-2601-20540-advancing-open-source-world-models-lingbot-world.md)

## 参考来源

- [sources/repos/lingbot-world-v2.md](../../sources/repos/lingbot-world-v2.md)
- [wechat_robot_yanfa_opensource_algorithms_compendium.md](../../sources/blogs/wechat_robot_yanfa_opensource_algorithms_compendium.md)

## 推荐继续阅读

- GitHub：<https://github.com/robbyant/lingbot-world>
- 技术页：<https://technology.robbyant.com/lingbot-world-v2>
