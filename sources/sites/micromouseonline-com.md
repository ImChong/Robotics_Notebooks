# Micromouse Online（micromouseonline.com）

> 来源归档

- **标题：** Micromouse Online
- **类型：** site（经典 Micromouse 技术站：规则、传感、控制、路径规划、高速跑）
- **作者：** Peter Harrison
- **链接：** https://micromouseonline.com/
- **入库日期：** 2026-07-27
- **一句话说明：** 自 1970 年代末竞赛传统延伸出的权威学习站：迷宫规则、电池/电机/墙传感、命令与控制、实际鼠例与速度相关文章。
- **备注：** 用户清单中的「IEEE Micromouse 官方」与「Micromouse Online」均指向本 URL。本站 **不是** IEEE 法人官方门户，而是社区公认的经典技术站；IEEE 系竞赛规则/历史在文中多有引用与转述。
- **相关：** [UKMARS](ukmars-org.md) · [WolfieMouse](../repos/wolfiemouse.md) · [emstef Webots](../repos/emstef-micromouse.md)
- **沉淀到 wiki：** [Micromouse](../../wiki/concepts/micromouse.md)

---

## 为什么值得保留

- **compilation 级教材**：控制、传感器、路径规划、打滑修正、直道时间等长文，适合作为 wiki 概念页「推荐继续阅读」主外链。
- **规则与历史**：1977 IEEE Spectrum Amazing Micromouse → 1980 Billingsley 中心目标规则等脉络清晰。
- **与开源鼠互证**：多篇工程经验可直接对照 UKMARSBOT / WolfieMouse 实现。

## 主题地图（站点导航摘要）

| 主题 | 内容 |
|------|------|
| Introduction / History | 竞赛起源与早期鼠 |
| Rules / Maze | 尺寸、自主、传感与推进约束 |
| Batteries / Motors / Sensors | 电源、驱动、墙传感设计 |
| Command and Control | 软硬件权衡、状态机 |
| Micromouse Book | 成体系章节 |
| Actual Mice | 历史名鼠档案 |
| 博客 | 打滑对定位、直道时间等工程笔记 |

## 开源核查（2026-07-27）

| 项 | 结论 |
|----|------|
| 站点内容 | **公开可读**（教学文章） |
| 代码 | 站点本身非单一 monorepo；作者另有社区 GitHub 引用（见 Webots 项目 References） |
| IEEE 官方身份 | **否** — 勿写成 IEEE 官方站 |

## 对 wiki 的映射

- [Micromouse](../../wiki/concepts/micromouse.md)
- [UKMARSBOT](../../wiki/entities/ukmarsbot.md)
- [A*](../../wiki/methods/a-star.md)
- [PID Control](../../wiki/methods/pid-control.md)
