# RL-Co：基于强化学习的仿真–真机协同训练（arXiv:2602.12628）

> 来源归档（paper）

- **标题：** Beyond Imitation: Reinforcement Learning-Based Sim-Real Co-Training for VLA Models（arXiv v1 的标题为 *RLinf-Co: Reinforcement Learning–Based Sim–Real Co-Training for VLA Models*，v5 起改为现名）
- **类型：** paper
- **作者：** Liangzhi Shi*、Shuaihang Chen*、Feng Gao、Yinuo Chen、Kang Chen、Tonghe Zhang、Hongzhi Zang、Jiakai Zhou、Weinan Zhang、Chao Yu（通讯）、Yu Wang（*共同一作；v5 署名；v1 作者列表略有不同）
- **机构（v5 首页）：** 清华大学；哈尔滨工业大学；北京大学；卡内基梅隆大学；上海人工智能实验室；北京中关村学院
- **arXiv：** <https://arxiv.org/abs/2602.12628>（v1 2026-02-13，v5 2026-10-22）
- **PDF：** <https://arxiv.org/pdf/2602.12628>
- **项目页：** <https://rl-co-training.github.io/>
- **代码：** 论文正文未给出代码仓库链接，仅列项目页（v5 首页 Project Page）
- **会场线索：** CoRL 2026 阅读清单（Glen Berseth）收录，OpenReview 链接 <https://openreview.net/forum?id=7hdkIcfGRw>（本环境无法打开，未核对）
- **入库日期：** 2026-11-12
- **一句话说明：** 两阶段 RL-Co：先用真机加仿真示范做 SFT 协同训练热启动，再在仿真里做 RL，同时加一项真机示范的 SFT 损失作锚点，缓解遗忘与域漂移；在 Franka 四个桌面任务、OpenVLA 与 π0.5 上优于纯真机 SFT 与 SFT 协同训练。

## 阅读记录

- **核对范围：** 下载 arXiv v1（16 页）与 v5（28 页，最新）PDF，用 pdftotext 抽文本逐段读完，范围为 v5 正文第 1–7 节（含局限）与附录 A–J 全部文字、表 1–8 数字、图注；v1 正文第 I–VI 节与表 I/II。以 v5 为准，v1 仅用于比对版本差异。图中曲线（图 5a、5c、6、11、13、14、15）只读了图注与正文描述，未逐点目视判读曲线数值；参考文献未逐条核对；未运行任何代码。
- **通读：** 已完成

## 核心摘录（均出自 v5，节号为 v5 节号）

1. **方法（第 4 节，式 2）：** Stage I 用 Dreal 与 Dsim 按比例 α 做 SFT 协同训练；Stage II 在仿真中做 RL，总损失 Ltotal = LRL + β·LSFT(θ; Dreal)，β 平衡 RL 更新与真机知识保留。
2. **实现（第 5.1 节、附录 I）：** OpenVLA 用 PPO（沿用 Liu 等 [12] 的协议）加 LoRA（rank 32）；π0.5 用 ReinFlow（经 RLinf 实现），只更新 action expert，视觉语言骨干冻结。
3. **数据（第 5.1 节、附录 B）：** 每个真机任务 20–50 条遥操作成功示范；仿真侧以回放的真机轨迹为种子，用 MimicGen 生成每任务 1000 条成功轨迹。
4. **仿真（附录 E.1）：** ManiSkill 刚体仿真，不追求照片级；相机外参手调对齐；仅用一条遥操作轨迹对 Franka 关节刚度与阻尼做系统辨识，回放 TCP 误差约 2 cm。
5. **主结果（表 1）：** 真机成功率四任务平均，OpenVLA 为 9.1（纯真机）/ 40.0（SFT 协同）/ 64.0（RL-Co）；π0.5 为 26.7 / 45.9 / 66.1。
6. **泛化（图 4）：** Pick and Place、π0.5：未见物体 25.0 / 31.3 / 56.3，未见初始状态 40.0 / 55.0 / 70.0（纯真机 / SFT 协同 / RL-Co），域内 71.9 / 68.8 / 81.3。
7. **视觉多样性对照（表 2，π0.5，Pick and Place）：** 重度域随机化零样本 10.9；Cosmos-Transfer 增强 67.2；RL-Co 81.3。
8. **消融（第 5.3 节、图 5b、附录 H）：** 去掉 Stage II 真机 SFT 正则 40.3；去掉 Stage I 真机 SFT 12.5；两阶段都去掉 6.3；单阶段 RL 协同训练真机 0.0；RialTo 式蒸馏 75.0（表 5，各只评一次）。
9. **数据效率（第 5.4 节、图 6）：** Open Drawer、π0.5：20 条真机示范的 RL-Co 不低于两个监督基线用 200 条的水平（文内表述为「不低于或相当」，具体数值只在图中）。
10. **额外任务（附录 F，表 4）：** Peg Insertion，π0.5，50 条真机加 500 条仿真轨迹：SFT 协同 32.5，RL-Co 42.5。
11. **评测口径（第 5.1 节、附录 D.1）：** 每个条件只评一个训练好的检查点、两个评测会话；每会话 Pick and Place 32 次、Push Cube 30 次、Open Drawer 10 次、Close Drawer 10 次；表中 ± 为两个会话的总体标准差，不是多训练种子的方差。
12. **资源（附录 J，表 8）：** Open Drawer 的 MimicGen 生成约 1.5 小时（RTX 4090），RL 协同训练约 9.43 小时（2×H100）；建仿真环境约 1 人日，四任务共享；收集 200 条真机示范约 2 人日。
13. **作者承认的局限（第 7 节）：** 仿真仅刚体；四个桌面任务、单一机器人形态、每条件两个评测会话；更难任务与其他形态未探索；成功率在若干设置下仍远低于 100%。

## 版本差异（v1 与 v5）

- 标题由 RLinf-Co 改为 Beyond Imitation；v5 增加附录 F–J（灵巧任务、仿真–真机相关性、额外基线、资源核算）与视觉多样性对照（表 2）、统计口径说明。
- 数字微调：OpenVLA Pick and Place 的 RL-Co 由 58.8 ± 10.0 改为 57.8 ± 10.9；OpenVLA 纯真机平均由 16.5 ± 13.3 改为 9.1 ± 5.8（按四个任务均值 6.3、20.0、0.0、10.0 计算应为 9.1，v1 的 16.5 与自身表内数据不符）；π0.5 Push Cube 的 RL-Co 由 18.4 改为 18.3，平均 66.2 改为 66.1。

## 对 wiki 的映射

- [paper-rl-co-sim-real-co-training](../../wiki/entities/paper-rl-co-sim-real-co-training.md)
