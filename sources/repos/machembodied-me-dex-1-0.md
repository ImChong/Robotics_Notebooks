# ME-Dex-1.0（MachEmbodied/ME-Dex-1.0）

- **URL：** <https://github.com/MachEmbodied/ME-Dex-1.0>
- **组织：** MachEmbodied
- **许可证：** Apache-2.0
- **关联项目页：** [ME-Dex 1.0 项目页](../sites/me-dex-1-0.md)
- **关联论文：** [me_dex_1_0_arxiv_2609_21449](../papers/me_dex_1_0_arxiv_2609_21449.md)

## 一句话说明

RoboTwin Clean50 训练的 video-action-tactile 策略 **推理 runtime**；通过 XPolicyLib 跑标准化 RoboTwin leaderboard；训练代码与数据待发布。

## 运行时入口（README 口径）

| 步骤 | 入口 |
|------|------|
| 依赖 | `pip install -r runtime/requirements.txt` |
| 权重 | HF `liuxuetao/ME-Dex-1.0-RoboTwin-Clean2Random-Leaderboard` + Wan2.2-TI2V-5B 骨干资产 |
| Leaderboard | 评测接口供 RGB；**无触觉观测** 时当前帧触觉置零、保留 sensor mask，未来触觉由模型预测 |
| 视觉 | `input_color_order: bgr`；不做 mean/std 归一化 |

## 交叉链接

- [ME-Dex 1.0 论文实体](../../wiki/entities/paper-me-dex-1-0.md)
- [RoboTwin](../../wiki/entities/robotwin.md)
