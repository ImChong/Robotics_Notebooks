# Real-Time Optimal Control via Deep Neural Networks: Study on Landing Problems

> 来源归档（ingest）

- **标题：** Real-time optimal control via Deep Neural Networks: study on landing problems
- **作者：** Carlos Sánchez-Sánchez, Dario Izzo（ESA Advanced Concepts Team, ESTEC）
- **论文：** <https://arxiv.org/abs/1610.08668>（v1，2016-10-27）
- **出版物：** arXiv 页未列期刊信息；期刊版本次未核对
- **类型：** arXiv 预印本
- **入库日期：** 2026-11-06
- **核对范围：** 读了 arXiv 摘要页与 HTML 版 §I、§III（问题设置与表 1）的原文；网络结构、训练超参数与表 6 成功率来自对后续章节的转述，未逐句核对。
- **通读：** 未完成
- **一句话说明：** 先用间接法（Pontryagin + 打靶 + 同伦）批量求出最优着陆轨迹，再监督训练前馈网络把状态直接映射成最优控制，作为机载实时最优控制的替代。

## 核心摘录

1. **方法（§I）。** "feed-forward DNNs architectures can be trained directly in a supervised manner on the optimal state-action pairs obtained via an indirect method (based on single shooting)." 作者提到早先用直接法生成的数据有 "chattering noise"。
2. **问题集（§III，表 1）。** 四个平面精确着陆模型：四旋翼、简单航天器、带反作用轮航天器、推力矢量火箭（TVR）；目标函数含最短时间、最省质量、二次控制；航天器模型用月球重力。
3. **结论（摘要）。** "the resulting landings, driven by the trained networks, are close to simulated optimal ones."（训练区域外泛化下降，见后文转述，未逐句核对）

## 对 wiki 的映射

- [动力下降制导（Powered Descent Guidance）](../../wiki/concepts/powered-descent-guidance.md) — 学习型路线
- [火箭回收控制纵深路线](../../roadmap/depth-rocket-landing.md) — Stage 6
