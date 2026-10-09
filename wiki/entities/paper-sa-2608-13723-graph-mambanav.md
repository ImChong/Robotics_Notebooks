---
type: entity
tags: [paper, objectnav, navigation, graph-mamba, mamba, ssm, llm, wuxi-university, njupt]
status: complete
updated: 2026-10-09
topic: [navigation]
project_id: graph-mambanav
arxiv: "2608.13723"
venue: "IEEE RA-L accepted; ICRA 2027 presentation planned"
related:
  - ../tasks/zero-shot-object-navigation.md
  - ../concepts/state-space-model-ssm.md
  - ../concepts/mamba.md
  - ./ai2-thor.md
sources:
  - ../../sources/papers/graph_mambanav_arxiv_2608_13723.md
summary: "面向 ObjectNav 的目标条件时空图策略：用 LLM 物体关系先验同时确定局部边权与 Graph-Mamba 节点顺序，再用逐物体时间 Mamba 建模历史。"
---

# Graph-MambaNav: Spatial-Temporal Graph Mamba Leveraging Object-Relation Knowledge for Object-Goal Navigation

**Graph-MambaNav** 是一个面向物体目标导航（ObjectNav）的时空图策略：它让目标相关性控制图信息传播的节点顺序，并用 Mamba 汇聚物体关系与逐物体历史。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| ObjectNav | Object-Goal Navigation | 在陌生环境中按目标物体类别导航 |
| LLM | Large Language Model | 生成对象共现/功能关系的离线常识先验 |
| GINE | Graph Isomorphism Network with Edge features | 使用边属性做局部图消息传递 |
| SSM | State Space Model | Mamba 执行长序列选择性扫描的模型族 |
| DETR | DEtection TRansformer | 冻结的物体检测器，提供对象外观、框与置信度 |
| A3C | Asynchronous Advantage Actor-Critic | 论文用来优化导航策略的 actor-critic 算法 |
| SR | Success Rate | 满足终止条件的导航 episode 占比 |
| SPL | Success weighted by Path Length | 同时衡量成功率与路径效率的指标 |
| RGB | Red, Green, Blue | 智能体逐步接收的单目彩色图像 |
| FIFO | First-In First-Out | 时间记忆按时间顺序移除最旧帧、追加新帧 |

## 为什么重要

ObjectNav 的目标物体在单帧里可能看不到。仅识别“现在看见了什么”还不够，智能体还要根据目标判断先去哪一类对象附近探索，并记住此前观察过什么。Graph-MambaNav 把关系先验从普通特征权重提升到**计算顺序**：目标相关的节点在因果扫描中靠后处理，能读取此前对象累积的上下文。

## 核心方法

每一步输入为目标类别和 egocentric RGB。论文用 COCO 预训练 DETR 得到对象级特征，同时用 ResNet-18 编码全局画面。ChatGPT-5 生成的目标条件对象关系亲和矩阵作为**离线先验**，不等于策略每一步都在线调用 LLM。

```mermaid
flowchart TB
  rgb["目标类别 + 当前 RGB"] --> perception["ResNet-18 全局视觉 + DETR 对象检测"]
  prior["离线 LLM 关系亲和矩阵"] --> graph["局部 GINE + 目标排序 Graph-Mamba"]
  perception --> graph
  graph --> memory["逐物体 FIFO 历史 + 时间 Mamba"]
  memory --> fusion["目标查询注意力 + 视觉/动作融合"]
  perception --> fusion
  fusion --> policy["LSTM + A3C 策略"]
  policy --> action["前进 / 转向 / 抬头低头 / Done"]
  action --> rgb
```

### 空间关系：局部边 + 全局有序扫描

每个物体类别作为一个图节点。当前帧里检测到的对象之间构造边；LLM 亲和度经映射成为可学习的边属性，由 GINE 聚合邻近对象关系。另一路计算每个节点与目标的相关度 $h_i=A[i,i^*]$，从低到高排序，并将目标节点强制置于末尾，再由 Graph-Mamba 因果扫描。两路信息相加，经 FFN 得到当前时刻空间表示。

这一步的直觉是：当目标是遥控器时，电视等相关节点排在扫描后段；它们能利用较早节点传播来的上下文，为目标节点提供关系线索。若扫描顺序随机，论文消融中的 AI2-THOR SR 会从 83.22% 降到 73.67%。

### 时间记忆：按对象跟踪历史

模型将最近 $T=35$ 帧的节点特征放进 FIFO 记忆，再按物体类别转置成逐对象时间序列，对每个对象的历史运行时间 Mamba。这样空间模块回答“这一帧里物体彼此有什么关系”，时间模块回答“同一类物体在最近观察中如何变化”。目标文本作为 query 对该记忆做 cross-attention，随后与当前全局视觉和上一动作融合，交给两层 LSTM 与 A3C 策略预测离散导航动作。

### 核心信息

| 项目 | 内容 |
|------|------|
| 主论文 | [arXiv:2608.13723v1](https://arxiv.org/abs/2608.13723v1) |
| 作者 | Leyuan Sun、Genxin Chen、Linwei Ye、Yan Zhang、Xi Kan、Yanfei Sun |
| 机构 | 无锡学院；Yan Zhang 同时署名南京邮电大学；Yanfei Sun 同时署名无锡市人工智能与安全重点实验室 |
| 发表 | 作者主页称已接收 IEEE RA-L，计划转至 ICRA 2027 展示 |
| 项目/代码 | 截至 2026-10-09，论文页和作者主页未链接官方代码仓库；IEEE 条目包含论文/Demo 入口 |
| 观测与动作 | 单目 egocentric RGB + 目标类别；MoveAhead、左右转、上下看、Done |
| 成功条件 | Done 时目标在视野内且距离低于阈值（论文举例 1.5 m） |

## 实验与评测

### 仿真结果

论文在 AI2-THOR 和 RoboTHOR 的未见室内环境划分上评测 SR 与 SPL；表中为论文报告值（%）。

| 基准 | 全轨迹 SR / SPL | 长轨迹（最短路 $L\geq5$）SR / SPL |
|------|------------------|--------------------------------------|
| AI2-THOR | 83.22 / 46.52 | 76.09 / 46.20 |
| RoboTHOR | 49.82 / 28.67 | 37.38 / 22.49 |

AI2-THOR 全轨迹对照：TSOG 为 80.04% SR / 41.44% SPL，Memory-MambaNav 为 81.24% SR；Graph-MambaNav 报告 83.22% SR / 46.52% SPL。在 RoboTHOR 上为 49.82% SR / 28.67% SPL。每组结果取 3 次运行，报告均值和标准差。注意：Memory-MambaNav 是作者重实现，去掉了其额外 reward 设计；其他基线多取自原论文的相同或可比协议，因此这些数字不是完全同条件复现的严格横向排名。

### 关键消融与计算量

- AI2-THOR SR：不加扫描模块为 63.83%；加入局部扫描后为 66.15%；再加全局有序扫描后为 78.34%；再加时间扫描后为 83.22%。消融显示全局顺序贡献最大，时间扫描继续提升整体与长路径结果。
- T=35 的计算量对照：Graph-MambaNav 为 12.58M 参数、23.47 GFLOPs、2.53 GB 显存、22.12 FPS；Transformer 变体为 62.13M、82.39 GFLOPs、6.72 GB、11.32 FPS。该 FPS 是论文实现的模型计算对比，不应直接视为机器人控制频率。
- 将目标相关节点的排序从 top-5 扩至 top-10/top-15，AI2-THOR SR 从 77.05% 上升到 82.63%；继续扩到全部 22 类后为 83.22%，呈现边际收益递减。
- 关系矩阵 10%/20%/30% 边项受随机噪声污染时，SR 分别降至 82.31%/80.46%/77.92%，说明先验误差会实质影响策略。

### 真机演示

作者在一台轮式移动机器人上演示客厅找书：0.65 m 高处安装单目 RGB 相机，Raspberry Pi 5 运行 ROS 2 低层控制，导航模型运行于外部主机，两者经局域网 ROS 2 topics 通信。论文展示机器人先靠近 laptop，再搜索椅子和电视区域，最后找到书。该部分是一个演示路线，没有报告多场景 SR、路径效率或统计重复次数，不能据此判断真机泛化幅度。

## 与其他工作对比

| 对照 | 关键区别 |
|------|----------|
| TSOG | 同属物体图导航；Graph-MambaNav 用目标亲和度显式决定全局扫描顺序，并以时间 Mamba 处理逐物体历史 |
| Memory-MambaNav | 都使用 Mamba 时间建模；Graph-MambaNav 进一步加入物体图结构与目标条件扫描顺序 |
| 普通 GNN/Graph Transformer | 目标相关性常由注意力或特征融合处理；本文把相关性用于控制因果计算顺序，以偏置长程信息流 |

## 源码运行时序图

**不适用**（截至 2026-10-09，arXiv 论文页与作者主页未提供官方可运行代码仓库或训练/推理入口；不能据论文架构虚构源码调用时序）。

## 结论

**Graph-MambaNav 的主要贡献是让 ObjectNav 的目标相关性同时决定“图上连什么”和“扫描时先后看什么”，并以逐对象时间记忆延长推理跨度。**

1. **全局扫描顺序是主要增益来源。** 消融中局部扫描 SR 增幅较小；加入目标相关全局排序带来最大跃升。
2. **时间 Mamba 对长轨迹有用。** 去掉 temporal scan 后，$L\geq10/15/20$ 的 SR 为 61.23/41.02/27.32%，完整模型为 65.92/42.63/33.40%。
3. **效率数据有边界。** 参数量、GFLOPs、显存和 FPS 是特定模型实现的对照，不代表外部主机到机器人控制端的端到端延迟。
4. **常识关系会引入偏置。** 论文自身指出 pot 等布局不符合先验时会走错方向，目标关联较弱时也可能降低探索多样性。
5. **真机证据仍是演示级。** 目前展示一个轮式机器人找书案例，未提供多场景、重复试验的量化结果。

## 局限与风险

- **固定类别集：** AI2-THOR 使用 22 类，RoboTHOR 使用 12 类；论文把扩展到开放词汇类别和多实例建模列为未来工作。
- **先验依赖：** 目标相关矩阵由 LLM 常识产生，房间布局偏离常识时可能误导探索；噪声实验也显示性能随污染程度下降。
- **评测复现边界：** 部分基线取自原论文，协议标为相同或可比；Memory-MambaNav 是作者自重实现，横向数字需结合这一点阅读。
- **真机量化不足：** 机器人控制由 Pi 5 执行、模型由外部主机运行；当前单一找书演示不能证明系统在复杂房屋中的稳定成功率。
- **代码状态：** 截至入库日官方论文页和作者主页未提供代码仓库链接。论文架构可用于理解方法，但目前不能按官方入口复现训练流程。

## 关联页面

- [零样本目标导航（ObjectNav）](../tasks/zero-shot-object-navigation.md) — 任务设定、探索与核验流程
- [状态空间模型与 Mamba](../concepts/state-space-model-ssm.md) — Mamba 选择性扫描的基本机制
- [Mamba](../concepts/mamba.md) — 选择性状态空间架构
- [AI2-THOR](./ai2-thor.md) — 论文使用的室内导航模拟环境

## 参考来源

- [Graph-MambaNav arXiv v1 摘录](../../sources/papers/graph_mambanav_arxiv_2608_13723.md)
- [arXiv:2608.13723v1](https://arxiv.org/abs/2608.13723v1)
- [arXiv v1 HTML 正文](https://arxiv.org/html/2608.13723v1)
- [Leyuan Sun 作者主页（RA-L 接收 / ICRA 2027 信息）](https://leyuan-sun.github.io/)

## 推荐继续阅读

- [Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective State Spaces（arXiv:2402.00789）](https://arxiv.org/abs/2402.00789)
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces（arXiv:2312.00752）](https://arxiv.org/abs/2312.00752)
