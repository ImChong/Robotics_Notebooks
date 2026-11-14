# LingBot-VLA 2.0 原文核对 / From Foundation to Application: Improving VLA Models in Practice

- **类型：** paper / 项目技术报告
- **论文：** https://arxiv.org/html/2607.06403v1
- **项目：** https://technology.robbyant.com/lingbot-vla-v2
- **代码：** https://github.com/Robbyant/lingbot-vla-v2
- **团队：** 蚂蚁灵波（Robbyant）
- **核对范围：** 论文 arXiv HTML 摘要、§1–7（数据清洗、55D 统一动作、MoE、Dual-Query 蒸馏、GM100/移动操作与消融），以及官方代码仓库的开放入口；对照既有 `sources/papers/lingbot_vla_v2_tech_report.md` 与 wiki 页面。
- **通读：** 已完成

## 一手核对要点

- 论文表述**先收集约 90,000 小时**，清洗筛得 **50,000 小时机器人轨迹（20 类构型）**，再混入 **10,000 小时 egocentric 人视频**，形成约 60,000 小时预训练配方；原转载“5 万+1 万”是训练后有效数据，不是总采集量。
- 统一动作表示为 **55 维**，含臂关节 14、双 EEF pose 14、gripper 2、灵巧手 12、腰 4、头 2、移动底盘 3、预留 4；缺失维度按平台掩码填充。
- MoE VLA 将视觉语言和动作专家衔接，Dual-Query 从深度与未来特征蒸馏信息；其作用是训练信号，不等于推理时全量生成未来视频。
- 结果包含 GM100 与真实平台长程移动操作；该论文多组评估对照和训练配方需要分开解读，不能与任意模型的 LIBERO 结果直接排序。

## 对 wiki 的映射

- [LingBot-VLA 2.0 唯一详情](../../wiki/entities/lingbot-vla-v2.md)
