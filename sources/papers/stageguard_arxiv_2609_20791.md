# StageGuard — 原文核对记录

- **标题：** StageGuard: Learning Stage Transitions for Long-Horizon Robot Tasks via Agentic Distillation
- **作者：** Jinbang Huang、Yuanzhao Hu、Zhiyuan Li、Ran Qi、Yixin Xiao、Yangzheng Wu、Tengyue Ba、Zhanguang Zhang、Yingxue Zhang
- **原文：** <https://arxiv.org/html/2609.20791>（v1，2026-09-17）
- **PDF：** <https://arxiv.org/pdf/2609.20791>
- **入库日期：** 2026-11-14
- **核对范围：** arXiv HTML v1 的问题定义 §III–IV、方法 §V、实验 §VI–VIII、结论与局限 §IX，表 I–III。
- **通读：** 已完成

## 原文要点

1. 对长时程分层规划的终止信号建模：`continue` / `advance` / `skip`，并预测下一子任务与执行历史。
2. Teacher 层级 agent 分析机器人状态、物体空间关系、任务依赖、决策理由，再用标注轨迹纠错；训练期生成高质量解释。
3. Student 为 Qwen3.5-0.8B，先生成短解释，再由教师挑选并加权 SFT（解释 token 权重 0.5；结构化输出权重 1）。
4. 表 I–II：LIBERO 转阶段完成率 96.23%，BEHAVIOR-1K 为 90.18%；两者推理吞吐 1.03 / 1.74 Hz（按各自测试环境）。
5. 表 III：BEHAVIOR-1K 闭环成功率从端到端 0.30 提升到 0.53，oracle transition 为 0.63。
6. §VIII：UR5e 18/20 次成功，Piper 16/20 次；剩余失败涉及感知遮挡及底层抓取。
7. §IX：依赖转阶段标注、昂贵教师样本生成，局部遮挡和低层控制错误仍限制效果。

## 开源状态

截至 2026-11-14，论文 HTML 未给出可核验的作者官方训练/推理仓库或模型发布入口；不能把同名个人 GitHub 账号当作论文实现。

## 对 wiki 的映射

- [StageGuard 论文实体](../../wiki/entities/paper-stageguard.md)
