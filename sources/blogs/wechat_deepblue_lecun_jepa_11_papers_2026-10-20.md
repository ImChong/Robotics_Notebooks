# 复盘 Yann LeCun 2026 全套研究路线｜从 JEPA 到能规划的世界模型

> 来源归档（blog / 微信公众号）

- **标题：** 复盘 Yann LeCun 2026 全套研究路线｜从 JEPA 到能规划的世界模型
- **类型：** blog
- **作者 / 账号：** 深蓝学院 深蓝具身智能（编译来源标注为「深蓝AI」）
- **原始链接：** https://mp.weixin.qq.com/s/0a7q-_2WVMK2lJgLhBQLLA
- **发表日期：** 2026-10-20（文内标注 Oct 20, 2026）
- **入库日期：** 2026-10-20
- **抓取方式：** `scripts/fetch_wechat_article.py`（Chromium + 微信 UA，`domcontentloaded` 后等待 3s 会得到空正文，改为 `load` + 等待 8s 后取得正文）
- **一句话说明：** 按训练效率、表示与规划、理论边界、真实控制四个方向，盘点 LeCun 2026 年参与的 11 项 JEPA / 世界模型相关研究。

## 文内点名 → 本库节点

| 文内项目 | 本库节点 | 状态 |
|----------|----------|------|
| LeWorldModel（LeWM） | [paper-lewm](../../wiki/entities/paper-lewm.md) | 已有节点，未改动 |
| LeVJEPA | [paper-levjepa](../../wiki/entities/paper-levjepa.md) | 已有节点，未改动 |
| TDV | [paper-tdv](../../wiki/entities/paper-tdv.md) | 本次新建 |
| Causal-JEPA | [paper-causal-jepa](../../wiki/entities/paper-causal-jepa.md) | 本次新建 |
| Value-Guided Action Planning | [paper-value-guided-jepa-planning](../../wiki/entities/paper-value-guided-jepa-planning.md) | 本次新建 |
| Music-JEPA | [paper-music-jepa](../../wiki/entities/paper-music-jepa.md) | 本次新建 |
| When Does LeJEPA Learn a World Model? | [paper-when-does-lejepa-learn-world-model](../../wiki/entities/paper-when-does-lejepa-learn-world-model.md) | 已有节点，未改动 |
| Rectified LpJEPA | [paper-rectified-lpjepa](../../wiki/entities/paper-rectified-lpjepa.md) | 本次新建 |
| LpWM | [paper-lpwm](../../wiki/entities/paper-lpwm.md) | 已有节点，未改动 |
| SkyJEPA | [paper-skyjepa](../../wiki/entities/paper-skyjepa.md) | 本次新建 |
| Patch Policy | [paper-patch-policy](../../wiki/entities/paper-patch-policy.md) | 已有节点，未改动 |

## 核心摘录（归纳，非全文）

- 文章主线：LeCun 2025 年底离开 Meta 并成立 AMI Labs 后，研究重点从「JEPA 能否学表示」转向世界模型的训练成本、表示结构与规划。
- 训练效率：LeWM 把可调超参数从 6 个减到 1 个；LeVJEPA 去掉防坍缩架构补丁；TDV 去掉视觉归纳偏置，只用时序因果做预训练。
- 表示与规划：Causal-JEPA 以物体而非 patch 为单位；Value-Guided Planning 让潜空间距离反映到达目标的难度；Music-JEPA 把框架用于音乐。
- 理论与稀疏：When Does LeJEPA Learn a World Model? 给出「何时算世界模型」的理论条件；Rectified LpJEPA 与 LpWM 学稀疏非负表示。
- 真实控制：SkyJEPA 把 JEPA 用于四旋翼高频闭环；Patch Policy 用稠密 patch 特征做轻量控制。

## 文章本身的局限

- 文中数字（如 LeVJEPA 的 FLOPs 倍数、SkyJEPA 的误差与参数量、TDV 的 KNN 指标）均为文章转述，入库时以各论文 / 项目页为准，未逐一核对原文表格。
- 文章为编译稿，无作者署名到个人。

## 相关归档

- 同主题对照：[腾讯科技 · 世界模型路线问题综述](wechat_tencent_world_model_questions_2026-09-05.md)
