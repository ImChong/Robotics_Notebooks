# PointWAM: 3D World Action Modeling for Dexterous Robotic Manipulation

> 来源归档（paper；核对 arXiv HTML v1 与官方项目页；2026-10-06）

- **作者：** Chunghyun Park, Beomjun Kim, Seungcheol Park, Heeseung Kwon, Yashu Shukla, Seunghoon Sim, Jinwoo Shin, Minsu Cho
- **机构：** POSTECH、KAIST、Hanyang University、RLWRLD
- **论文：** <https://arxiv.org/abs/2610.02840> · [HTML v1](https://arxiv.org/html/2610.02840v1)
- **项目页：** <https://chrockey.github.io/PointWAM/>
- **代码：** 项目页标注 “Code soon”；截至 2026-10-06 无官方仓库 URL，代码待发布。
- **一句话说明：** PointWAM 在共享三维时空坐标系内联合预测场景点与手部关键点轨迹，再将未来手轨迹重定向成灵巧机器人动作。

## 核心摘录

1. **问题定位。** 传统 VLA 多从当前图像直接预测动作；视频式 WAM 能建模未来但不一定显式保留灵巧接触几何；点策略能预测手的 3D 点，却通常不预测环境如何共同演化。
2. **表示。** 输入彩色点云和语言指令，分解 world 为 scene 与 hands，两者在一个共同 3D space-time frame 中预测轨迹；不需要人为预选任务相关物体或场景关键点。
3. **动作头。** Transformer 轨迹预测器用双 head 预测 scene points / hand keypoints；action retargeter 结合预测的手部轨迹与机器人状态输出动作块，scene trajectory 作为预测监督。
4. **人类视频预训练。** EgoDex 与 VITRA 合计 1.15M 人类演示片段，没有机器人 action 标签；统一成场景/手轨迹表征后预训练，再用机器人示教微调。
5. **结果。** 人类视频预训练令 DexJoCo 平均成功率增加 56.9 个百分点；scene 轨迹监督比只预测手部多 10.9 个百分点；十项 DexJoCo 多任务平均 69.0%，较强基线高 11.7 个百分点。真机实验用 OpenArm 与 Inspire 手展示两个任务。
6. **复现状态。** 项目页有交互轨迹查看器、结果、视频和 “Code soon” 按钮；未列仓库或可下载权重，代码待发布。

## 对 Wiki 的映射

- [PointWAM 论文实体](../../wiki/entities/paper-pointwam.md)
- [World Action Models 概念页](../../wiki/concepts/world-action-models.md) — 已补入站内实例链接。
- [VLA 方法页](../../wiki/methods/vla.md)；[Manipulation 任务](../../wiki/tasks/manipulation.md)

## 参考来源（原始）

- arXiv HTML v1：<https://arxiv.org/html/2610.02840v1>
- 官方项目页：<https://chrockey.github.io/PointWAM/>
