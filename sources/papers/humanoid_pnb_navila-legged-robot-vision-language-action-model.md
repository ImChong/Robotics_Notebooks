# NaVILA: Legged Robot Vision-Language-Action Model for Navigation

> 来源归档（ingest · Robot Learning Paper Notebooks 深读笔记）

- **标题：** NaVILA: Legged Robot Vision-Language-Action Model for Navigation
- **类型：** paper
- **笔记链接：** <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/08_Navigation/NaVILA_Legged_Robot_Vision-Language-Action_Model_for_Navigation/NaVILA_Legged_Robot_Vision-Language-Action_Model_for_Navigation.html>
- **分类：** 08_Navigation
- **arXiv：** <https://arxiv.org/abs/2412.04453>
- **项目页：** <https://navila-bot.github.io/>
- **代码：** <https://github.com/AnjieCheng/NaVILA>（**已开源**，Apache-2.0；训练 / 评测 / 权重 / 标注已发布）
- **入库日期：** 2026-06-07
- **通读：** 已完成
- **一句话说明：** NaVILA 提出了一个两层分层框架，将高层视觉 - 语言 - 动作（VLA）推理出的自然语言指令转换为低层足式运动控制，实现了人类自然语言指令到复杂地面导航的高效映射。

## 核心摘录（均出自原文，arXiv:2412.04453）

- **两层接口：** 高层 VILA-8B（单视角 8 帧 RGB history + 当前帧 → 中层参数化语言动作 `move forward X cm` / `turn left/right Y deg`）；低层视觉 RL locomotion policy（Isaac Lab 单阶段 PPO，输入本体状态 + 速度命令 + Unitree L1 360×90 LiDAR 2.5D heightmap，训练吞吐 60k FPS on RTX 4090，输出 12 关节控制，避免 2-stage ROA 蒸馏：线速度误差 0.066 vs 0.161、碰撞率 0.81 vs 3.09）。
- **多源 SFT 数据：** R2R-CE / RxR-CE（合并最多 3 步连续动作生成多样距离标签）、2k YouTube touring videos（MASt3R 估尺度位姿生成 20k trajectories；仅发布 IDs 与 annotations）、EnvDrop 轨迹摘要、ScanQA 与通用 VQA。
- **评测结果：** R2R-CE Val-Unseen SR **54.0%** / SPL **49.0%**（NaVid 37%）；RxR-CE SR 49.3% / SPL 44.0%；VLN-CE-Isaac（1,077 条 Go2/H1 物理轨迹）Go2 Vision SR 50.2% vs Blind 36.2%，H1 Vision SR 45.3% vs Blind 24.4%；真机 Unitree Go2 & Booster T1 25 条指令 × 3 次重复整体成功率 **88%**；W4A16 AWQ 量化部署延迟 367.8 ms / 显存 8.6 GB（对比 FP16 594.6 ms / 18.5 GB）。
- **开源边界：** core repo、HF 权重 / 标注、`yang-zj1026/NaVILA-Bench` 和 `yang-zj1026/legged-loco` 均有入口；旧 Habitat 0.1.7 与多仓库组合仍是主要复现成本。
- 知识归纳见 wiki 实体页：[paper-notebook-navila-legged-robot-vision-language-action-model](../../wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md).

## 对 wiki 的映射

- [paper-notebook-navila-legged-robot-vision-language-action-model](../../wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md)
- [NaVILA 项目页归档](../sites/navila.md)
- [NaVILA 仓库归档](../repos/navila.md)
- 分类父节点：[paper-notebook-category-08-navigation](../../wiki/overview/paper-notebook-category-08-navigation.md)

## 参考来源（原始）

- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/08_Navigation/NaVILA_Legged_Robot_Vision-Language-Action_Model_for_Navigation/NaVILA_Legged_Robot_Vision-Language-Action_Model_for_Navigation.html>
- 论文：<https://arxiv.org/abs/2412.04453>
