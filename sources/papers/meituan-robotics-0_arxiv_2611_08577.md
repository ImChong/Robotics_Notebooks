# Meituan-Robotics-0（arXiv:2611.08577）

> 来源归档（paper）

- **标题：** Meituan-Robotics-0: A Vision-Language-Action Foundation Model for Desktop Manipulation
- **类型：** paper（技术报告，cs.RO）
- **arXiv：** <https://arxiv.org/abs/2611.08577>
- **PDF：** <https://arxiv.org/pdf/2611.08577v1>（v1，提交日 2026-11-05，首页标注 arXiv:2611.08577v1 [cs.RO] 5 Nov 2026）
- **作者 / 机构：** Meituan Robotics Team（首页署名；项目负责人 Jie Hu，赞助人 Guanglu Wan、Xunliang Cai、Yuchen Xie，见 Contributions 节）
- **项目页：** <https://meituan-robotics.github.io/Meituan-Robotics-0/>（2026-11-09 可访问）
- **代码：** <https://github.com/Meituan-Robotics/Meituan-Robotics-0>（首页与项目页均给出链接；本次未能核实仓库内容）
- **入库日期：** 2026-11-09
- **一句话说明：** 面向固定底座桌面操作的 VLA 基础模型 + 事件驱动具身智能体：147M 视觉语言样本与 15,060 小时 12 种本体机器人数据，三阶段训练（FAST 离散预训练、MMDiT+JiT 连续中训练、真机后训练），并提出 Rollback 采集与 CFG-DAgger。

## 阅读核对记录

- **核对范围：** 通读 arXiv v1 PDF 正文第 1–6 节（引言、数据整理、模型与三阶段训练、推理系统、实验、结论）、Contributions、附录 A（真机设置与数据准备）、B（异步执行细节与算法 1）、C（后训练细节，含 HG-DAgger / RECAP / CFG-DAgger 的公式与超参数表 8）、参考文献列表，并查看了图 1、图 10–14 的渲染页面。未核对：引用的外部工作（RoboTwin 2.0、RoboDojo、RLinf RECAP 文档等）的原文；RoboDojo 榜单与基线数字仅按论文表 5 转述，未回官方榜单核对；项目页已读，GitHub 仓库访问失败（raw README 返回 404，无 GitHub API 权限），代码 / 模型 / 数据集的实际内容未核实。
- **通读：** 已完成

## 开源状态

- 论文首页与项目页均列出 Code / Model / Data 链接（项目页另有 Technical Report、Code、Model、Dataset 入口）。**仓库内容、权重与数据集的实际可用性本次未核实**（步骤 2.5：项目页已打开，GitHub 仓库未能读取）。

## 核心摘录

1. **数据（第 1、2 节，图 2–3，表 2）：** 147M 视觉语言样本（通用 78%、具身 22%，训练重采样前）；机器人数据 15,060 小时、12 种本体、仅限固定底座桌面操作：RoboDesk 3,235 h（21.5%，6 个开源真机数据集）、授权真机 7,530 h（50.0%）、开源仿真 4,295 h（28.5%）。清洗为 3 个过滤阶段（静止首尾、关节与末端位姿 FK 一致性、视频缺陷）加 2 个对齐阶段（跨本体动作规范化、分层任务标注）。
2. **动作规范化（式 2–3）：** 每臂 10 维（肩中心系下绝对末端位置 3 维 + 连续 6D 旋转 + 夹爪闭合度），双臂 20 维；单臂用 10 维。
3. **架构与三阶段训练（第 3 节）：** 骨干由 Qwen3.5-4B 初始化并全程可训练。阶段 1：FAST 动作 token 与 VQA 联合自回归预训练（λ_vla=1.0，λ_vlm=0.1，500,000 步，256 GPU）；阶段 2：丢弃 token 接口，换 MMDiT 连续动作头（JiT clean-sample 参数化的流匹配，128 GPU），用掩码损失处理不同动作宽度（d_max=20）；阶段 3：真机后训练（SFT、HG-DAgger、RECAP、CFG-DAgger 为并列方案，均从同一中训练检查点 π_mid 重新训练，不是依次叠加）。预训练动作池为约 13.6k 小时（表 3），与 15,060 小时总清单不同。
4. **Rollback 与 CFG-DAgger（第 3.2.3 节，图 6，附录 C）：** 操作员暂停、回到更早的锚点状态、给出纠正续作；被替换的策略片段 τ⁻ 标 I=0，人工替换 τ⁺ 与示范标 I=1；推理时动作空间 CFG：a_CFG = a⁻ + γ(a⁺ − a⁻)，γ=1.25。RECAP 对照为价值学习 + 优势条件（β=1.75，优势阈值取聚合数据 70 分位，策略训练排除失败轨迹——脚注 2 称这样效果更好）。
5. **部署系统（第 4 节）：** 异步动作块合并（f_c=30 Hz，f_π=3 Hz，H=50，K_lat=4，表 7）；事件驱动智能体：Monitor（2 Hz，8 帧 2 秒窗口）、Memory Manager、Planner，三者用 Qwen3.5-35B-A3B（4 个 SGLang 实例）。
6. **RoboTwin 2.0（第 5.1 节，表 4）：** 50 任务、每任务每划分 100 回合；Clean 93.2%，Randomized 93.9%，Overall 93.6%（Overall 为两者均值）；仅评估 VLA 策略本身，不含智能体和 RL 后训练。表中 GalaxeaVLA (G0.5) 的 Clean 为 93.7，高于 MR0 的 93.2；Overall 与 Random 列 MR0 最高，但 Overall 领先 G0.5（93.3）仅 0.3 个百分点、领先 InternVLA-A1.5（93.2）0.4 个百分点；π0.5 为 79.8。
7. **RoboDojo（第 5.1 节，表 5）：** 42 个双臂仿真任务；带历史信息（首帧参考 + CD-LAM 潜在动作历史）的 MR0 平均 Score 20.15 / SR 14.63%，在表中所列 9 个 VLA 条目里按 SR 排第五（排在 Simate-beta 27.96、Rex-M1 Preview 27.44、DM0.5 19.34、GalaxeaVLA 14.88 之后；其后依次为 Xiaomi-Robotics-1 13.93、Hy-Embodied-0.5-VLA 8.80 等）；开放词汇指令一项仅 3.59 / 3.38%。论文称 MR0 结果为同一检查点两次完整评估的均值。
8. **真机（第 5.2 节，表 6，图 10–12）：** 三任务（字母块、叠衣服、网线插接），双臂 AgileX PiPER，每策略每任务自主评估 30 次。匹配 HG-DAgger 下 MR0 与 π0.5：平均 SR 约 0.78 对 0.69，平均 PS 0.91 对 0.85（图 1 / 项目页数值；图 10 柱标 79 对 69）。字母块两者 SR 同为 83%，叠衣服 83% 对 80%，网线 70% 对 43%（图 10 柱标；网线该对比使用 1,400 多条 Ethernet 回合，见表 6 注）。两种 SFT 在叠衣服上 SR 均为 0。
9. **Rollback 对比（5.2.2 节，图 11，附录 C）：** 网线插接，示范集固定，每轮三种方法各增加 50 条 Rollback 回合（每轮 150 条共享池，共 3 轮 450 条）；完整 Rollback 数据下 CFG-DAgger SR 0.70 / PS 0.90，RECAP SR 0.60 / PS 0.67；HG-DAgger 的末轮 SR 在图 11 中约 0.5（读图估计，正文未给数字）。
10. **智能体评估（5.2.3 节，图 13–14）：** 两个受控定性案例（跨任务事件检索、子任务等待与恢复），无聚合成功率。

## 论文自述的局限 / 未验证

- 智能体部分仅为定性展示；论文明确说跨任务检索案例评估的是事件保留与检索，不涉及外部干预下对所有已满足目标条件的保持（5.2.3 节）。
- RoboDojo 开放词汇指令仍弱（SR 3.38%），在 Align_blocks 任务中策略会偏向熟悉动作（5.1 节）。
- HIL 数据收益依任务与已有数据覆盖而异，字母块和叠衣服先快速提升后趋于平台，网线仍在上升（5.2.2 节，图 12）。
- 未来工作指向让智能体与 VLA 更紧密交互（第 6 节）。

## 对 wiki 的映射

- [paper-meituan-robotics-0](../../wiki/entities/paper-meituan-robotics-0.md)
