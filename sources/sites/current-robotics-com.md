# Current Robotics（current-robotics.com）

> 来源归档（site / company / research product）

- **标题：** Current Robotics
- **类型：** site / company
- **官方入口：** <https://current-robotics.com/>
- **入库日期：** 2026-08-17
- **一句话说明：** 现行机器人（Current Robotics）公司站：可穿戴采数外骨骼产品 + 研究博客；2026-06 发布 [Curr-0](https://current-robotics.com/blog/curr-0) 人形 loco-dex 策略栈，2026-08 发布 [CurrentWorld-0](https://current-robotics.com/blog/currentworld) 交互世界模拟器。
- **与本次 ingest 的关系：** 作为 CurrentWorld-0 的 **项目页核查入口**（步骤 2.5）；代码开放状态以本页与博客实际链接为准。

## 开源与资源（2026-08-17 核查）

| 资源 | URL | 备注 |
|------|-----|------|
| 首页 | <https://current-robotics.com/> | 品牌叙事；Trusted by 智元 / 星海图 / 北京人形创新中心 / 字节 Seed / 舞肌 / NVIDIA |
| CurrentWorld-0 博客 | <https://current-robotics.com/blog/currentworld> | 世界模型主文；**无代码链接** |
| Curr-0 博客 | <https://current-robotics.com/blog/curr-0> | 策略全栈主文 |
| 代码 | — | **确认未开源**；页头/页脚/Research 区无 GitHub、Hugging Face、Zenodo |
| Hi-WM 论文 | <https://arxiv.org/abs/2604.21741> | 首页 Research 列出的后训练论文；作者含 Li / Zhou / Chen / Guo / Zhu 等 |

### 产品（首页，非本 ingest 深读）

- Head Module — Egocentric Vision Unit（立体 RGB + 鱼眼 + 6 轴 IMU，60 fps）
- Hand Module — Dexterous Exoskeleton（关节编码器 + 力传感器）
- Full-Body Exoskeleton（关节编码器 + IMU 融合）

与 Curr-0 的 **HumanEx** 可穿戴叙事同一产品线，但本页不把产品规格升格为 wiki 硬件实体。

## 对 wiki 的映射

- [`wiki/entities/current-robotics-currentworld.md`](../../wiki/entities/current-robotics-currentworld.md) — CurrentWorld-0 交互世界模拟器
- [`wiki/entities/current-robotics-curr0.md`](../../wiki/entities/current-robotics-curr0.md) — Curr-0 策略全栈
- 博客深归档：[current_robotics_currentworld.md](../blogs/current_robotics_currentworld.md)、[current_robotics_curr0_loco_dexterous_manipulation.md](../blogs/current_robotics_curr0_loco_dexterous_manipulation.md)
