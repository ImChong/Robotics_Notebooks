# WorldScape Policy 2.0 项目页（manifoldai-research.github.io/WorldScape-Policy）

> 来源归档

- **标题：** WorldScape Policy 2.0 — Empowering Steerable World Action Modeling with Reasoning-Augmented Memory
- **类型：** site / project-page
- **URL：** <https://manifoldai-research.github.io/WorldScape-Policy/>
- **论文：** <https://arxiv.org/abs/2607.18840> — 归档见 [`sources/papers/worldscape_policy_2_arxiv_2607_18840.md`](../papers/worldscape_policy_2_arxiv_2607_18840.md)
- **代码：** <https://github.com/manifoldai-research/WorldScape-Policy>（**占位**）— 归档见 [`sources/repos/worldscape-policy.md`](../repos/worldscape-policy.md)
- **权重：** <https://huggingface.co/manifoldai-research/WorldScape-Policy-2>（**占位**，声明 Apache-2.0）
- **机构：** Manifold AI · WorldScape Team（论文署名含清华大学、上海交通大学）
- **入库日期：** 2026-08-03
- **一句话说明：** 项目页提供 PDF、arXiv、Code、Model 四个入口与真机演示视频；Code / Model 链接虽在，但打开均为「coming soon」占位。

## 公开信息要点（截至入库日）

- **代码：** 链接存在但仓库为占位（README + `.gitignore`），训练 / 推理 / 评测代码 **待发布**。
- **权重：** HF 卡片存在但无文件，预训练模型与 RoboTwin 2.0 后训练 checkpoint **待发布**。
- **数据：** ManipEvent-5M **4.89M** 事件段 / **744K** episode / **512M** 帧；页面未给数据下载入口。
- **仿真结果（页面口径）：** RoboTwin 2.0 平均成功率 **94.3%**。
- **真机结果（页面口径，dual-arm PiPER）：** 自主规划（叠衣服）**75%**、记忆推理（shell game）**75%**、技能迁移（叠积木）**70%**、指令跟随（清桌）**80%**。
- **方法自述：** 事件级推理 + 帧级视觉记忆双记忆；三阶段训练 = 事件接地预训练 → 带 semantic forcing 的记忆感知 mid-training → 交互式后训练。

## 关联资料

- 论文摘录：[`sources/papers/worldscape_policy_2_arxiv_2607_18840.md`](../papers/worldscape_policy_2_arxiv_2607_18840.md)
- 仓库归档：[`sources/repos/worldscape-policy.md`](../repos/worldscape-policy.md)
- Wiki 实体：[`wiki/entities/paper-worldscape-policy-2.md`](../../wiki/entities/paper-worldscape-policy-2.md)
