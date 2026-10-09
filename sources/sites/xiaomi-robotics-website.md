# Robotics @ Xiaomi 官网 Research 列表

- **类型：** 官方站点索引（研究 / 技术博文列表）
- **链接：** https://robotics.xiaomi.com/
- **核查日期：** 2026-10-09
- **访问说明：** 本次环境中该站 HTTPS 证书与域名不匹配，经 HTTP 读取首页与 XR-0 项目页；内容以官网为准。
- **用途：** 核对公司路线「小米机器人」是否遗漏官方技术博文。

## 官网 Research 列表（2026-10-09 共 4 篇）

| 官网日期 | 标题 | 官网入口 | 本库节点 |
| --- | --- | --- | --- |
| 2026-07-16 | Xiaomi-Robotics-1: Scaling Vision-Language-Action Models with over 100K Hours of Real-World Data | `/xiaomi-robotics-1.html` | [Xiaomi-Robotics-1](../../wiki/entities/xiaomi-robotics-1.md) |
| 2026-07-15 | Xiaomi-Robotics-U0: Unified Embodied Synthesis with World Foundation Models | `/xiaomi-robotics-u0.html` | [Xiaomi-Robotics-U0](../../wiki/entities/xiaomi-robotics-u0.md) |
| 2026-04-27 | Open-Sourcing Post-Training Pipeline for Xiaomi-Robotics-0 | `/xiaomi-robotics-0.html#pack-earbuds` | 并入 [Xiaomi-Robotics-0](../../wiki/entities/xiaomi-robotics-0.md)「后训练管线开源」小节 |
| 2026-02-12 | Xiaomi-Robotics-0: An Open-Sourced Vision-Language-Action Model with Real-Time Execution | `/xiaomi-robotics-0.html` | [Xiaomi-Robotics-0](../../wiki/entities/xiaomi-robotics-0.md) |

## 2026-04-27 后训练更新要点

- 官网 XR-0 项目页顶部注明 "Update (Apr 27, 2026): The full post-training pipeline is now available."；它是项目页内的更新段落，不是独立页面。
- 案例 "Learning to Pack Earbuds with 20h Data"：把耳机按左右放进充电盒，要求高精度抓取与空间对位；用 20 小时数据后训练后，可连续放好三副耳机且无失败。
- [GitHub README](https://github.com/XiaomiRobotics/Xiaomi-Robotics-0) News：2026-04-27 开源后训练代码（`xr0/` 目录，含安装、数据准备、训练与部署说明及样例数据）；2026-02 发布技术报告、预训练与 LIBERO / CALVIN / SimplerEnv 微调权重、推理与评测脚本。

## 未列入官网 Research 的小米机器人论文

TacRefineNet、ViTacPhys、UCAG-P 不在官网 Research 列表中，本库按 arXiv 与项目页单独收录；官网列表不代表团队全部论文。
