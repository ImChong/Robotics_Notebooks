---
type: entity
tags: [paper, humanoid, rl, motion-control, world-model, jepa, bfm, sjtu, shanghai-innovation-institute, zju, tongji, fudan]
status: complete
updated: 2026-11-01
project_id: bejepa
arxiv: "2610.23127"
project: https://ylyem9x.github.io/BeJEPA-project/
summary: "BeJEPA 用 Dynamic 一步物理预测与 Successor 长期行为摘要共同训练世界锚定的人形控制表征，在 G1 上实现姿态、位置和朝向联合控制。"
related:
  - ./paper-bfm-zero.md
  - ./paper-behavior-foundation-model-humanoid.md
  - ../concepts/behavior-foundation-model.md
  - ../concepts/whole-body-control.md
  - ../overview/embodied-station-2026-11-01-56-papers-technology-map.md
sources:
  - ../../sources/papers/bejepa_arxiv_2610_23127.md
  - ../../sources/sites/bejepa-project-page.md
  - ../../sources/blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md
---

# BeJEPA

**BeJEPA**（arXiv:2610.23127）是一个通过无监督强化学习训练的目标条件人形行为世界模型，用统一潜在表征连接身体姿态、世界位置与朝向控制。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| JEPA | Joint-Embedding Predictive Architecture | 在嵌入空间预测状态表征，不重建全部观测细节 |
| RL | Reinforcement Learning | 通过机器人与环境交互优化长期目标行为 |
| MPC | Model Predictive Control | 用短时预测和滚动优化修正当前动作 |
| BFM | Behavior Foundation Model | 以可复用行为表征支持多种人形控制目标 |

## 为什么重要

动作跟踪器可能把姿态复现得很准，却让机器人停在参考动作原本不该出现的位置。BeJEPA把世界坐标中的根位置和朝向与身体构型共同纳入目标，使控制器能判断“先迈一步、暂时偏离姿态”是否有助于最终回到目标位姿。

## 核心原理

### Dynamic 与 Successor 分工

1. **共享编码器**将身体状态、根位置及朝向投到同一嵌入空间；当前状态和目标状态的嵌入相似度定义目标奖励。
2. **Dynamic Predictor**接收状态嵌入与动作，预测下一步状态嵌入。它从已执行的转移中学习可预测的物理后果，不依赖某个具体目标。
3. **Successor Predictor**以当前状态与目标为条件，预测策略未来将访问的状态特征摘要。它提供长期行为价值，不必递归想象很长的动态轨迹。
4. **Actor**把 Dynamic 的一步动作后果交给 Successor 评估。两者共同训练表征：Dynamic 在训练早期提供直接物理监督，Successor 随策略改进逐步组织长期行为。
5. **观测历史编码器**把训练用状态表征蒸馏到部署可用的传感器与动作历史；对时间索引的动作跟踪，固定 Dynamic 模型还可用于在线动作残差优化。

人类动作先验提供目标与运动先验，在线机器人交互数据提供物理转移。方法重点不是重建像素或完整未来视频，而是学习对目标控制有用的身体—世界状态表征。

## 方法与流程

- **训练：** 在 Isaac Sim 的 4,096 个并行环境中预训练；策略动作是送入低层 PD 控制器的 29 维关节目标。
- **跟踪：** 使用 AMASS 与 LAFAN 动作片段评估 20 秒长程执行，保留大空间偏差轨迹，并测全局人体关节误差、根平移/朝向误差及手腕误差。
- **真机接口：** Unitree G1 使用本体感觉与 LiDAR/FAST-LIO 提供的位置、朝向估计，让世界锚定目标在机器人受扰后仍保持意义。
- **在线精修：** 对时间索引参考动作，Dynamic 预测短时嵌入轨迹；有界动作残差优化降低空间/时间偏差，执行首个修正动作后再根据新观测滚动规划。
- **源码运行时序图：** 不适用（截至 2026-10-23，项目页源码区尚未核实，arXiv 页面未列可运行代码入口）。

## 实验与评测

- **长程跟踪：** BeJEPA 在 AMASS、LAFAN 上改善全局身体位置与根位姿误差；仅看局部关节姿态会掩盖整体漂移。
- **在线 MPC 精修：** 相同预训练权重下，Balanced 设置把 AMASS 六项误差平均相对降低 31.2%，LAFAN 降低 13.4%；AMASS 平均有符号跟踪时延由 94.8 ms 降至 21.2 ms。
- **对照摘要：** 论文摘要报告 BeJEPA 的在线 Dynamic 精修相对 world-pose 输入的 BFM-Zero 在 AMASS 全局位置误差降低 61.7%，相对 SONIC 降低 58.7%。不同指标不可互换；该数字不代表所有误差项都同步降低。
- **消融：** 合并预测器或去掉任一编码器梯度都会削弱世界目标控制，支持“Dynamic 锚定物理变化、Successor 建模长期行为”的分工。
- **Unitree G1 真机：** 展示了组合姿态/位置/朝向目标、从地面起身、受踢后回到原站立位置，以及姿态与方向切换后返回地面标记。

## 与其他工作对比

- **BFM-Zero**（[paper-bfm-zero](./paper-bfm-zero.md)）以 Forward–Backward 表征学习可提示的人形行为；BeJEPA 针对世界锚定控制显式联合身体姿态与位置/朝向，并采用 Dynamic + Successor 双预测器。
- **SONIC**重点扩展动作跟踪的规模与泛化；BeJEPA将长期目标控制和空间位置恢复纳入同一潜在目标。
- 与直接依赖长时域潜在动态 rollout 的模型相比，BeJEPA通过 Successor 直接汇总目标条件的未来行为，Dynamic 保留短时物理预测能力。

## 结论

**BeJEPA的关键价值是把“身体要摆成什么样”和“身体要到世界中的哪里”放进同一个可学习目标，并把即时物理预测与长期行为评价拆开。**

1. **受扰后回归世界目标** — 对走开再返回、恢复站位等任务，应同时看根位置与朝向误差，不能只看局部姿态跟踪。
2. **不要混淆两种预测** — Dynamic回答动作会造成什么局部变化；Successor回答这些变化之后策略预计会到达哪些状态。
3. **MPC 是跟踪增强项** — Balanced 在线修正明显降低时空误差，但需要额外预测与优化；不应把在线 MPC 的性能全部归于 actor 单独执行。
4. **指标要按场景读** — 全局 MPJPE 与根误差揭示位置漂移，局部 MPJPE 测姿态；表内某一项变好不等于所有项都变好。
5. **部署依赖世界位姿估计** — 真机使用 LiDAR/FAST-LIO 位置与朝向输入；定位误差、时延和失效恢复是复现时需单独验证的系统条件。
6. **源码仍待核实** — 目前 arXiv 提供论文与 HTML；项目页内容未成功读取，不能据此判断代码是否已开放。

## 局限与风险

- 真机演示能证明指定目标序列可执行，不足以说明任意环境下的自主导航或长期定位鲁棒性。
- 世界锚定依赖根位置/朝向估计；定位漂移会直接影响目标奖励和闭环返回表现。
- 在线动作精修增加推理计算，实际控制频率和计算预算应结合论文附录及部署平台复测。
- 项目页本次无法读取，代码开放状态留待核验；归档记录见[项目页来源](../../sources/sites/bejepa-project-page.md)。

## 关联页面

- [BFM-Zero](./paper-bfm-zero.md) — 无监督强化学习的人形行为基础模型对照。
- [Behavior Foundation Model](../concepts/behavior-foundation-model.md) — 行为基础模型的概念与方法谱系。
- [Whole-Body Control](../concepts/whole-body-control.md) — 人形全身姿态与运动控制问题。

## 参考来源

- [BeJEPA 论文来源归档](../../sources/papers/bejepa_arxiv_2610_23127.md)
- [BeJEPA 项目页核查记录](../../sources/sites/bejepa-project-page.md)

## 推荐继续阅读

- [BeJEPA arXiv HTML](https://arxiv.org/html/2610.23127v1) — 方法、消融和附录细节。
- [BFM-Zero 项目页](https://lecar-lab.github.io/BFM-Zero/) — Forward–Backward 行为表征对照。
