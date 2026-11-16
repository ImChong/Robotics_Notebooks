# DreamVLA（arXiv:2611.10872）

> 来源归档（paper）

- **标题：** DreamVLA: From Generated Human Motions to Humanoid Vision-Language-Action Policies
- **类型：** paper
- **作者：** Kevin Park（Princeton University）、Izat Rasyad、Dvij Kalaria、Shankar Sastry（UC Berkeley）；前两位同等贡献
- **arXiv：** <https://arxiv.org/abs/2611.10872>（v1，2026-11-07，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2611.10872v1>
- **项目页：** <https://ember-center-berkeley.github.io/DreamVLA/>
- **代码：** 论文与项目页均未给出代码仓库链接（2026-11-11 核查）
- **会议：** CoRL 2026（论文首页脚注：10th Conference on Robot Learning, Austin TX）
- **入库日期：** 2026-11-11
- **一句话说明：** 用 DreamControl 生成的人体动作经 OmniRetarget 重定向到 Unitree G1，在 SONIC 动作 token 接口上训练残差 RL 专家，收集成功轨迹微调 GR00T，再用「参考后缀条件的恢复式 DAgger」从学生失败状态补采纠正轨迹，全程仅在仿真。
- **核对范围：** arXiv v1 PDF 全文 13 页（摘要、引言、相关工作、方法 3.1–3.4、结果 4.1–4.3、局限、结论、参考文献、表 1–4、图 1–3 的图注）已逐段读完；论文无附录 / 补充材料；图内像素（仿真截图、视频帧）仅看图注未做像素级核对；引用的 DreamControl、SONIC、GR00T、VIRAL 等原文未读（仅按本文转述）。项目页静态读取了 HTML 文本（摘要、两张结果小表），未运行任何代码，视频「coming soon」未核对。
- **通读：** 已完成

## 开源状态

- 论文正文与项目页（Abstract / Overview / Results / BibTeX / 页脚）均无 GitHub、Hugging Face 等链接，仅有 PDF 与引用的 DreamControl 项目页；截至 2026-11-11 未见代码或数据发布，也未见「将开源」承诺。
- 项目页的视频区标注 "Overview video coming soon" / "Result videos are coming soon"。

## 核心摘录

1. **管线（§3，图 1）：** DreamControl 生成任务条件人体动作（OmniControl 约束关节与时间，式 1）→ OmniRetarget 重定向到 G1 → HumanoidMimicGen 改接触点做增广 → SONIC 之上残差 token RL 专家（式 5–7）→ 成功 rollout 存 LeRobot 格式 D0（式 8）→ 微调 GR00T（式 9）→ 参考后缀条件的恢复式 DAgger（式 10–11）。
2. **学生输入（§3.3）：** 仅机器人第一视角相机图像 + 本体感觉 + 语言指令；特权物体状态与参考上下文只给专家；VLA 输出 SONIC 运动指令 + 二值手部指令。
3. **任务与成功判据（表 1）：** 平台拾取（抬高 ≥15 cm 并保持）、开抽屉（位移 >0.15 m）、按按钮（右手食指接触）、行走+拾取（同拾取判据）。每项成功率用 100 次 rollout（§4.1）。
4. **DAgger 消融（表 2）：** 初始 200 条专家轨迹，每轮加 200 条成功纠正轨迹，k=3 共 800 条。Pick 0/17/30/54%，Drawer 51/55/54/60%，Button 59/58/62/69%，Walk and Pick 0/1/7/18%；等量 800 条离线对照为 0/54/57/0%。
5. **残差 RL 消融（表 3，专家成功率，3000 迭代）：** 名义 SONIC 0/30.0/68.0/17.30%，残差 token RL 88.6/74.0/81.0/75.95%，绝对 token RL 全部 0%；MPJPE 24.84 / 25.51 / 62.29 mm。
6. **数据规模（表 4）：** 参考数 10/50/200（episode 数同），Pick 与 Walk and Pick 始终 0%；Drawer 13.0/51.0/51.0%，Button 0/14.0/59.0%。
7. **局限（§5）：** 仅四个仿真任务、单一人形本体；生成动作可能与机器人运动学或接触要求不相容；任务相关的引导、过滤、物体参考和奖励权重仍是管线必要部分；成功过滤可能低估困难配置；参考条件专家在学习者访问状态上可能给出不可靠或突兀的纠正。结论明确硬件部署不在本文范围。
8. **项目页与论文的数值差异：** 项目页「DAgger task success」小表为 Drawer 55/57/54/62、Button 50/55/61/72；「Training-data task success」为 Drawer 4/15/55、Button 0/15/50。这与论文表 2（Drawer 51/55/54/60，Button 59/58/62/69）、表 4（Drawer 13/51/51，Button 0/14/59）不一致，项目页还把横轴写成 episodes（10/50/200）。本归档与 wiki 页以论文表格为准。

## 对 wiki 的映射

- [paper-dreamvla](../../wiki/entities/paper-dreamvla.md)
