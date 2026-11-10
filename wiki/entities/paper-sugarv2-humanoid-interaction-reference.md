---
type: entity
tags:
  - paper
  - humanoid
  - loco-manipulation
  - visual-policy
  - reinforcement-learning
  - sim2real
  - pku
status: complete
updated: 2026-11-10
project_id: sugarv2-humanoid
arxiv: "2611.09266"
related:
  - ../tasks/loco-manipulation.md
  - ../methods/ppo.md
  - ../methods/teacher-student-dagger-training.md
  - ../methods/dagger.md
  - ../methods/sonic-motion-tracking.md
  - ../concepts/whole-body-control.md
  - ../concepts/domain-randomization.md
  - ../concepts/sim2real.md
  - ../entities/paper-viral-humanoid-visual-sim2real.md
  - ../entities/paper-grail.md
  - ../entities/paper-humanoidmimicgen.md
  - ../entities/paper-loco-manip-161-076-sugar.md
  - ../entities/unitree-g1.md
sources:
  - ../../sources/papers/sugarv2_arxiv_2611_09266.md
original:
  - ../../sources/papers/sugarv2_arxiv_2611_09266.md
summary: "SUGARv2（arXiv:2611.09266）：用 LLM 生成的可复用任务模板适配不同空间配置，合成机器人与物体的交互参考，再以特权教师、教师蒸馏加 PPO（共用参考型 mimic 奖励）训练第一视角视觉策略；五任务仿真均值 91.0%，G1 真机五任务合计 38/50，各任务单独训练。"
---

# SUGARv2（arXiv:2611.09266）

**SUGARv2**（*Learning Diverse Visual Humanoid Loco-Manipulation Skills from Synthesized Interaction References*，[arXiv:2611.09266](https://arxiv.org/abs/2611.09266)，北京大学 CFCS / 上海科技大学 / 北京航空航天大学）把「人形视觉 loco-manipulation」拆成两步：先从任务模板低成本合成交互参考，再用同一套参考型模仿奖励训练特权教师与视觉学生。本页依据 v1 全文，细节见 [来源归档](../../sources/papers/sugarv2_arxiv_2611_09266.md)。

## 一句话定义

**把可复用的任务级交互模板适配到不同机器人 / 物体 / 目标位置，合成「机器人姿态 + 物体轨迹 + 接触标签」参考；用这些参考定义统一的 mimic 奖励，先训特权教师，再经教师蒸馏 + PPO 训出只看第一视角 RGB 与本体感觉的视觉策略（§3）。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SUGARv2 | （论文未展开缩写） | 本文框架名；wiki 里另有同名系列前作页，见关联页面 |
| WBC | Whole-Body Controller | 冻结的预训练全身控制器 [26]，接收潜变量命令 z（§3.3.1） |
| PPO | Proximal Policy Optimization | 视觉策略的 RL 部分（式 4） |
| DAgger | Dataset Aggregation | 在学生访问的状态上查询教师动作（§3.3.2，引 [35]）；VIRAL 基线用在线 DAgger + 行为克隆 |
| MSE | Mean Squared Error | 蒸馏损失（式 3） |
| RGB | Red-Green-Blue | 第一视角彩色图像输入 |
| PnP | Pick and Place | 抓取放置，图 2 中任务名 |
| VLA | Vision-Language-Action | 作者设想用合成参考支撑多任务 VLA（§5），本文未实现 |

## 为什么重要

- **作者主张（§1）：** 视觉人形 loco-manipulation 扩展到新任务 / 新空间配置时，成本主要在演示采集与任务专用的训练设计；HumanoidMimicGen 依赖初始遥操作演示且上下半身解耦，GRAIL 每个配置都要视频生成 + 运动重建、代价高且可能物理不合理（表 1）。
- **论文给出的证据：** 合成一条参考平均 0.4 分钟，对 GRAIL 的 36.1 分钟；教师成功率 98.0% 对 62.4%（表 5）。
- **我的判断：** 其价值在于把「参考合成 + 统一模仿奖励」做成可跨任务复用的管线，减少逐任务写奖励；但是否真的「免任务专用设计」要等附录与代码（见局限）。

## 核心机制

**流程总览**

```mermaid
flowchart LR
  A[任务描述] --> B[GPT-6 Astra 生成任务模板]
  B --> C[按空间配置实例化关键帧与时序]
  C --> D[运动生成模型 + 过滤排序]
  D --> E[由手部运动构造物体轨迹与接触区间]
  E --> F[交互参考 q_robot q_obj 接触标签]
  F --> G[特权教师 PPO + mimic 奖励 + 冻结 WBC]
  G --> H[视觉学生 蒸馏 + PPO + 同一 mimic 奖励]
  H --> I[视觉域随机化 + 动力学随机化]
  I --> J[Unitree G1 零样本部署]
```

**1. 交互参考合成（§3.2）**

- 参考为 τ_c = {q_t^robot, q_t^obj, l_t}（式 1），不绑定具体物体实例，可与同类物体复用。
- 模板由 GPT-6 Astra 为每个任务生成：有序关键帧，含全身姿态、机器人 - 物体 / 手 - 物体空间关系、各手接触标签（§3.2.1）。
- 实例化（§3.2.2）：根据物体初始位姿与目标，用手 - 物体关系和物体对称性生成候选手部目标，在可达性与场景约束下调整机器人位置、朝向与全身姿态，取移动与姿态调整最少者；按相邻姿态间的运动需求定时序；再用现成运动生成模型 [34]（附加任务文本条件）生成连续全身运动，每个配置生成多条候选，过滤违反运动学 / 场景约束者并按平滑、少冗余排序；物体轨迹由交互手运动与手 - 物体关系构造，并修正使其经过关键帧指定位置。

**2. 特权教师（§3.3.1）**

- 教师输入特权状态与参考轨迹，输出 WBC 的潜变量命令 z 与双手开合命令 h；奖励 r = 跟踪 + 交互 + 正则（式 2）。正则项还鼓励物体与目标留在相机视野内。
- 训练回合从首次接触前的参考状态起，加随机扰动，偏离过大或进入非法状态则提前终止。具体奖励定义与超参在附录，v1 PDF 未附。

**3. 视觉学生（§3.3.2）**

- 视觉编码器 [32] 投影图像特征，与本体感觉拼接，MLP 预测 (z, h)。
- 先做教师蒸馏（在学生访问的状态上查询教师动作 [35]），再继续用同一 mimic 奖励做 RL 且保留蒸馏项：L_policy = L_PPO + λ_distill·L_distill（式 4）；特权 critic 在蒸馏阶段预热。整个视觉策略端到端训练。
- 在线渲染多种物体实例、背景、材质、光照；随机相机位姿、图像增强与噪声、本体感觉噪声、动力学随机化与外部扰动（两阶段均用）。

## 实验结果

| 项 | 论文数字（出处） |
|----|------------------|
| 任务 | Tabletop Pickup、Floor-to-Table、Box Carrying、Basket Transport、Same-Table；每任务 100 条合成参考，3 个种子（§4.1） |
| 主结果 | 91.0%（92.8 / 91.7 / 87.6 / 84.3 / 98.5）（表 2c） |
| 学习方法对比（同参考与教师） | VIRAL 39.6%，GR00T N1.7 23.4%（表 2a） |
| 参考来源对比（同学习管线） | GRAIL 参考 28.2%（表 2b） |
| 未见背景 / 物体 / 两者 | 86.9 / 87.7 / 83.2（表 3） |
| 消融 | 无蒸馏 0.0；无参考型 RL 41.4；无持续蒸馏 70.6（表 4） |
| 参考数量 | 20→100 条提升，之后增益有限（图 5，§4.4） |
| 真机 G1 | 9/10、6/10、8/10、8/10、7/10，合计 38/50（表 6） |

注意：表 2(a) 中基线用的是 SUGARv2 自己的参考与教师，比较的是学习方法；表 2(b) 换参考源比较。两者都不是各系统「原配置」的对照（我的判断）。

## 与其他工作对比

| 工作 | 参考 / 数据来源 | 与 SUGARv2 的关系（论文表 1、表 2、表 5） |
|------|-----------------|------------------------------------------|
| [HumanoidMimicGen](./paper-humanoidmimicgen.md) | 初始遥操作演示扩增，上下半身解耦 | SUGARv2 用 LLM 模板合成参考，不需要初始演示 |
| [GRAIL](./paper-grail.md) | 每个配置视频生成 + 运动重建 | 合成一条参考 0.4 对 36.1 分钟；同管线下 GRAIL 参考仅 28.2% |
| [VIRAL](./paper-viral-humanoid-visual-sim2real.md) | 在线 DAgger + 行为克隆训视觉策略 | 同参考与教师下 39.6%，对 SUGARv2 的 91.0% |
| [GR00T N1.7](./paper-gr00t-n1-7.md) | VLA 模型 | 同参考与教师下 23.4% |

## 结论

**SUGARv2 用「LLM 任务模板合成交互参考 + 统一参考型 mimic 奖励」训练视觉人形 loco-manipulation，仿真五任务均值 91.0%，G1 真机 38/50。**

1. 参考合成成本低：一条 0.4 分钟，教师成功率 98.0%，对 GRAIL 的 36.1 分钟与 62.4%（表 5）。
2. 教师蒸馏是必需项：无蒸馏 0.0%，无持续蒸馏 70.6%，无参考型 RL 41.4%（表 4）。
3. 对未见背景 / 物体仍有 83.2–87.7%（表 3），视觉随机化起作用。
4. 基线是受控比较（共用 SUGARv2 参考与教师），且策略按任务单独训练；附录与代码缺失，奖励细节暂无法核对。

## 工程实践

- **作者给出的设置：** 每任务一个特权教师 + 一个视觉策略；每任务 100 条参考；仿真报告 3 种子均值 ± 标准差（§4.1）。真机为 Unitree G1，无真机微调，每任务 10 次、改变初始位置（§4.6）。
- **依赖：** GPT-6 Astra（模板）、现成运动生成模型 [34]、预训练 WBC [26]（SONIC，见 [SONIC 页](../methods/sonic-motion-tracking.md)）、视觉编码器 [32]。复现需要这些外部组件，具体型号与超参论文正文未全部给出。
- **代码：** 项目页标注 "Code (Coming Soon)"，截至 2026-11-10 无仓库，故不写 `code:`。
- **使用提示（推测）：** 若想复用思路，可先在自己的 WBC 上验证「参考 → 教师」这一步的跟踪成功率，再接视觉蒸馏；蒸馏阶段不可省（表 4：无蒸馏为 0.0%）。

## 局限

**作者自述（§4.6、§5）：**

- 参考合成目前只覆盖简化场景中的短时程交互；长时程、杂乱环境需更自动化的合成。
- 视觉策略是任务专用的，不是多任务。
- 抓空后策略可能不重试而继续后续动作，失败恢复有限。

**我看到的未验证之处：**

- 附录缺失：正文引用附录给奖励定义和训练细节，v1 PDF 无附录，奖励项、λ_distill、网络结构无法核对。
- 真机每任务仅 10 次，无置信区间，无真机基线对比，也未报告真机在未见背景 / 物体上的结果；Floor-to-Table 真机为 6/10，明显低于仿真 91.7%（sim-to-real 差距论文未量化分析）。
- 基线 VIRAL、GR00T N1.7 使用 SUGARv2 的参考与教师，且训练预算同设；这是受控比较，不等同于与原论文系统的端到端对比。基线调参程度正文未说明。
- 参考质量评估只与 GRAIL 一家对比；模板由闭源 LLM 生成，换用其他模型的敏感性未测。
- 「Cost」列（表 1）为定性标注，无统一度量。
- 代码未开放，结果暂无法独立复现。

## 关联页面

- [Loco-Manipulation 任务页](../tasks/loco-manipulation.md)
- [VIRAL：人形视觉 sim2real（基线之一）](paper-viral-humanoid-visual-sim2real.md)
- [GRAIL（参考合成对比对象）](paper-grail.md)
- [HumanoidMimicGen（遥操作演示扩增对照）](paper-humanoidmimicgen.md)
- [SUGAR（2605.20373，wiki 中的同名系列页）](paper-loco-manip-161-076-sugar.md)
- [SONIC 动作跟踪（论文 WBC [26]）](../methods/sonic-motion-tracking.md)
- [教师 - 学生蒸馏训练](../methods/teacher-student-dagger-training.md)、[DAgger](../methods/dagger.md)、[PPO](../methods/ppo.md)
- [全身控制](../concepts/whole-body-control.md)、[域随机化](../concepts/domain-randomization.md)、[Sim2Real](../concepts/sim2real.md)
- [Unitree G1](unitree-g1.md)

## 参考来源

- [sources/papers/sugarv2_arxiv_2611_09266.md](../../sources/papers/sugarv2_arxiv_2611_09266.md)
- [arXiv:2611.09266](https://arxiv.org/abs/2611.09266)
- 项目页：<https://tianshuwu.github.io/sugarv2-humanoid/>

## 继续阅读

- [arXiv PDF v1](https://arxiv.org/pdf/2611.09266v1)
- [VIRAL 页](paper-viral-humanoid-visual-sim2real.md)：理解表 2(a) 基线的做法
- [GRAIL 页](paper-grail.md)：理解表 2(b)、表 5 的对比对象
