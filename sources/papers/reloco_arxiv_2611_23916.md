# ReLoco: Unified Humanoid Locomotion and Fall Recovery via Behavior-Conditioned Motion Priors（arXiv:2611.23916v1）

> 一手论文来源归档；本页是对原文的摘要与核对记录，不替代论文。

- **类型：** paper / humanoid / locomotion / fall-recovery / AMP / Sim2Real
- **英文标题：** ReLoco: Unified Humanoid Locomotion and Fall Recovery via Behavior-Conditioned Motion Priors
- **中文标题：** ReLoco：通过行为条件运动先验统一人形行走与跌倒恢复
- **作者：** Lequn Fu、Yufeng Hu、Yijun Zhong、Jie Zhang、Xiao Li、Lingxiang Ji、Zhiyuan Xu、Jian Tang、Shiqi Li
- **机构：** 华中科技大学（HUST）；北京人形机器人创新中心（Beijing Innovation Center of Humanoid Robots）
- **论文：** https://arxiv.org/abs/2611.23916
- **全文 HTML：** https://arxiv.org/html/2611.23916
- **PDF：** https://arxiv.org/pdf/2611.23916
- **版本：** arXiv:2611.23916v1，2026-11-18（预印本）
- **项目主页：** https://lequn-f.github.io/ReLoco/
- **入库日期：** 2026-11-21
- **核对范围：** 通读 arXiv v1 主文 I–IV（含算法公式、Table I–VII、图及参考文献）；打开项目主页核对演示与开源入口。
- **通读：** 已完成
- **代码开放：** 截至 2026-11-21 官方项目页未列训练/推理/部署源码仓库、权重或数据下载链接；可在线体验 MuJoCo 交互演示，但演示≠可复现源码。详情见 [项目页归档](../sites/reloco.md)。
- **沉淀到 wiki：** [ReLoco 独立论文节点](../../wiki/entities/paper-reloco-unified-locomotion-fall-recovery.md)

## 摘录一：问题与单策略主张（§I）

过去的行走、起身常分别学习和切换控制器。ReLoco 将 standing / walking / running / fall / get-up / command resumption 交给同一个纯本体感知 actor；根据躯干姿态**只在训练期**切换专家采样池、AMP 判别器输出头与任务奖励，部署时没有外部模式切换器。

## 摘录二：双阶段参考生成（§II-C，Table III–IV）

- 自采人体动作 + BONES-SEED 选定片段经 GMR 运动重定向；离线 tracking policy 在机器人/执行器仿真模型里将运动轨迹“跑出来”，选定机器人 rollout 形成 locomotion / get-up 专家库。
- 43 组配对运动：地面穿透率 28.99%→0.75%，自碰撞率 12.91%→0.35%，关节限位违规率 12.90%→0.06%，body-position jerk 406.88→262.49 m/s³（均为该实验口径）。
- 每个条件各一次训练对照，20k iter 时 refined 参考的 logged 起身率约 95%，raw 约 85%，并非多随机种子统计显著性。

## 摘录三：行为条件 AMP 与训练（§II-A–D，Table I–II）

- 策略：非对称 PPO；actor 观测躯干角速度、投影重力、速度指令、关节位置/速度、上一步动作，**不直接观测 torso 线速度或行为路由标签**；23 维 action 是相对于 default joint angles 的偏移，PD 跟踪。
- **监督路由：** 当躯干高度 h≥0.9 m 且躯干竖直对齐 u≥0.7 走 upright route，否则为 non-upright / recovery route。二值标志 b 不提供给部署 actor。
- **AMP：** 10 帧历史，每帧 76 维运动特征 + 1 维当帧条件 → 770 输入；共享判别器 backbone + 2 个行为专用 score heads。末帧 b 选择 head 和专家库，专家历史保留其所属库的条件标签；与统一任务奖励、共享正则共同训练一个 actor。
- **奖励/优势：** upright route 追踪平面速度/偏航；recovery route 鼓励抬升躯干；力矩、加速度、关节限位、动作变化的正则两路共享；PPO advantage 分路归一化。
- **探索 curriculum：** 一半环境初始化于 get-up 帧，其余取全库；15–20 s 间隔施加速度扰动；跌倒时可给头部向上辅助力，起身统计超过 0.8 后每次重置递减 20 N，训练结束减至 0；真机无辅助。
- **Sim2Real：** 依据 PACE 辨识机器人与执行器，再随机化接触、质量、惯量、命令延迟等；训练使用 Isaac Lab。

## 摘录四：仿真实验与消融（§III，Table V–VI）

- 同一训练迭代 20k：ReLoco **独立起身 96.80±1.47%**、**push-recovery 96.96±0.73%**（后者必须恢复指令跟踪）；torso 平面速度 RMSE **0.1266±0.0009 m/s**；相对条件单头 AMP RMSE 降 **14.7%**。5 组配对评测 batch，每种恢复任务每批 250 次；策略消融各一个原始训练 run。
- 任务奖励-only 的 RMSE 更低（0.0717 m/s），但独立起身 **0%**；无条件 AMP 起身率 94.64%，但速度 RMSE 高达 0.8599 m/s；恢复与 tracking 必须联合评估。
- 与 HoST 的 1000 组随机静置倒地比较：ReLoco **98.3%**、起身 **2.02 s**、机械功 **3.1 kJ**；HoST **99.9%**、**1.57 s**、**10.7 kJ**；ReLoco 不应宣传为所有起身指标最佳。单参考 tracker 起身率 49.7%。
- 判定：独立起身需 h>1.0 m、u>0.8，5 s 内达到并再保持 0.5 s；push-recovery 要求重新跟踪命令，固定 5 s 测量窗速度 RMSE≤0.3 m/s。

## 摘录五：真机效果与边界（§III-D，Table VII）

- **天工 3.0：** 1.70 m、64.5 kg，31 DoF 中策略控制 23 DoF；zero-shot 真机部署，无单独 recovery controller、无真机微调。
- 站立被踢 **47 次全部记录**：34 次摔倒且控制继续，均自主恢复；5 次未倒；**8 次触发硬件保护**。绝不能将“34/34 有效倒地恢复”改述为“47/47 全成功”。
- 平地行走/跑步各 10 次可评估扰动试验：16 次未倒继续执行，4 次倒地起身后恢复运动；另有 1 次跑步硬件保护（单独记账）。恢复时间 30 次均衡姿态子集均值 **2.64±0.64 s**。
- 演示还覆盖砖石地面、草地及草坡；脚踢/推力未做力传感定量测量，无法给出可恢复最大冲量或复杂地形成功率。

## 核心判断及工程映射

- **一体化监督不是一体化模式切换：** 多路奖励/判别器只存在于训练期，部署仅一个 actor。
- **物理可执行参考优于直接运动学重定向：** 高接触/限位违规率会污染 AMP expert pool。
- **安全保护单独计入：** 评估要分“未跌倒继续 / 跌倒恢复继续 / 硬件保护”，不能只筛选策略有效的 trial。
- **关联路线：** [人形 RL Locomotion](../../roadmap/depth-rl-locomotion.md)；[Balance Recovery](../../wiki/tasks/balance-recovery.md)；[HoST](../../wiki/entities/paper-host-humanoid-standingup.md)；[SD-AMP](../../wiki/entities/paper-unified-walk-run-recovery-sdamp.md)。

## 一手资料

- [arXiv:2611.23916v1 全文](https://arxiv.org/html/2611.23916)
- [ReLoco 官方主页](https://lequn-f.github.io/ReLoco/)
