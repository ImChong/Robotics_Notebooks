---
type: entity
tags: [paper, humanoid-paper-notebooks, manipulation, teleoperation, augmented-reality, data-collection, sim2real, cloud-simulation, mit]
status: complete
updated: 2026-09-28
arxiv: "2411.02214"
related:
  - ../overview/paper-notebook-category-06-manipulation.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ../tasks/teleoperation.md
  - ../concepts/sim2real.md
  - ../concepts/domain-randomization.md
  - ../methods/action-chunking.md
  - ./paper-notebook-armada-augmented-reality-for-robot-manipulation.md
sources:
  - ../../sources/papers/humanoid_pnb_dexhub-and-dart.md
summary: "构建通才机器人系统受制于多样高质量数据的稀缺。本文提出 DART：一个借云端仿真与增强现实（AR）做可扩展机器人数据采集的众包遥操作平台。采集的数据自动存入 DexHub ——一个云端托管数据库，意在成为机器人学习的公共仓库。用户研究表明 DART 相比真机遥操作实现更高采集吞吐、更低体力疲劳，并能成功 sim-to-real 迁移、对视觉扰动鲁棒。"
---

# DexHub and DART

**DexHub and DART: Towards Internet Scale Robot Data Collection** 收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：06_Manipulation），深读笔记已完成。本页编译自深读笔记与 arXiv 论文正文（实验数字、开源状态于 2026-09-28 核对），细节以论文 PDF 为准。

## 一句话定义

构建通才机器人系统受制于多样高质量数据的稀缺。本文提出 DART：一个借云端仿真与增强现实（AR）做可扩展机器人数据采集的众包遥操作平台。采集的数据自动存入 DexHub ——一个云端托管数据库，意在成为机器人学习的公共仓库。用户研究表明 DART 相比真机遥操作实现更高采集吞吐、更低体力疲劳，并能成功 sim-to-real 迁移、对视觉扰动鲁棒。

## 英文缩写速查

| 缩写 | 含义 |
|---|---|
| DART | 众包遥操作采集平台（云仿真 + AR） |
| DexHub | 云端公共机器人数据仓库 |
| Crowdsourcing | 众包 |
| Cloud Simulation | 云端仿真 |
| AR | 增强现实 |
| Throughput | 采集吞吐量 |

## 为什么重要

- **"云仿真 + AR 众包"是互联网规模采集的可行路径**，绕开本地硬件；
- **公共数据库**对社区共享与基础模型训练意义重大；
- 对人形（硬件稀缺）尤其友好；
- 与 ARMADA（AR 无机器人采集）思路相通、规模更大。

## 解决什么问题

通才机器人缺**互联网规模**数据： - 真机遥操作**吞吐低、疲劳高、难规模化**； - 缺**公共、可众包**的采集平台与仓库。

DexHub/DART 要：用**云仿真 + AR 众包**采集，建**公共数据库**，迈向互联网规模。

## 核心机制

1. **DART 众包采集平台**：云仿真 + AR，可扩展、低门槛；
2. **DexHub 公共数据库**：意在成为机器人学习公共仓库；
3. **优于真机遥操作**：吞吐更高、疲劳更低；
4. **sim-to-real + 视觉鲁棒**：采集数据可迁移真机。

方法拆解（深读笔记小节）：DART：云仿真 + AR 众包遥操作；DexHub：云端公共数据库；验证；🧭 整体流程（mermaid）。

## 核心信息

| 字段 | 内容 |
|------|------|
| 分类 | 06_Manipulation |
| 深读笔记 | <https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/DexHub_and_DART__Towards_Internet_Scale_Robot_Data_Collection/DexHub_and_DART__Towards_Internet_Scale_Robot_Data_Collection.html> |
| arXiv | <https://arxiv.org/abs/2411.02214> |
| 源码 | **未确认**：论文指向 DexHub 平台 <https://dexhub.ai/project>，2026-09-28 从本环境访问 TLS 握手失败，无法核实是否提供代码或应用下载；论文正文未给出 GitHub 仓库 |
| 作者 | Younghyo Park、Jagdeep Singh Bhatia、Lars Ankile、Pulkit Agrawal（MIT） |
| 发表 | 2024 年 11 月 |
| 笔记阅读日期 | 2026-06-21 |

## 实验与评测


**DART**：在 Apple Vision Pro 上把云端仿真的资产作为 AR 物体渲染，只传输仿真状态而非图像以降低延迟；手部追踪经 IK 驱动机器人，支持一键复位与任务切换。**DexHub** 为云端数据仓库，记录所有演示并提供上 / 下游 API。

**用户研究**：
- 与真机遥操作（双 ViperX + 运动学同构主手、Rainbow RB-Y1）相比，20 名参与者在 4 个双臂任务上（每人每任务 7 分钟），真机采集吞吐约低 **2 倍**；真机大量时间花在手动复位与处理硬件故障。
- 视觉反馈消融：把渲染图像经网络传输（立体或单目）都会明显降低吞吐，单目更差；固定瞳距的立体渲染会让部分参与者头晕；去掉主动视角（头动改变视点）吞吐下降 **21.7%**。
- 控制方式：运动学同构主手相对手部追踪 IK 没有显著提高成功率，参与者更偏好手部追踪（更轻、更省力）。

**Sim2Real 与泛化**（两种 RGB ACT 策略，各用 50 分钟采集投入；DART 数据额外随机化相机内外参、背景与光照）：

| 场景 | 取杯入篮：真实数据 / DART | 小物分拣：真实数据 / DART |
|------|------|------|
| 实验室原场景 | 65% / 80% | 45% / 60% |
| 光照变化 | 45% / 45% | 25% / 65% |
| 背景变化 | 10% / 60% | 5% / 40% |
| 相机位姿变化 | 5% / 35% | 0% / 40% |
| 未见干扰物 | 0% / 70% | 0% / 45% |
| 公共厨房（换地点） | 0% / 50% | 0% / 35% |

- DART 数据训练的策略零样本迁移到真机，且在大多数扰动场景下明显更鲁棒。

## 与其他工作对比

| 方案 | 采集方式 | 与 DART 的差异 |
|------|------|------|
| ALOHA 类运动学同构主手 | 真机 + 主从臂 | 需要硬件、复位慢、易疲劳，吞吐约低一半 |
| [ARMADA](./paper-notebook-armada-augmented-reality-for-robot-manipulation.md) | AR 叠加虚拟机器人、在真实物体上演示 | 数据来自真实世界；DART 在仿真里采集，可做大量视觉增广 |
| [Sim-and-Real Co-Training](./paper-notebook-sim-and-real-co-training-a-simple-recipe-for-vis.md) | 仿真数据与真实数据混合训练 | 可与 DART 采集的仿真数据配合使用 |
| 真实世界大规模数据集 | 真机遥操作 | 论文强调 DART 是补充而非替代 |

## 结论

**DART 赌的是采集侧的经济学：把遥操作从「真机 + 本地硬件」搬到「云仿真 + AR 众包」，再用 DexHub 把众包出来的数据沉淀成公共仓库。**

- 真正起作用的不是新算法而是吞吐与门槛：用户研究显示相比真机遥操作采集吞吐更高、体力疲劳更低，且不依赖本地硬件，因而才谈得上"互联网规模"。
- 这条路径能否成立取决于仿真采集的数据是否可用——本页给出的关键证据正是 sim-to-real 迁移成功与对视觉扰动的鲁棒性，这也是该类方案最容易被质疑的地方。
- DexHub 是"意在成为"公共仓库，属于目标而非既成事实：数据库的价值随社区参与规模放大，因此本工作的成败与其说在方法，不如说在生态。
- 对人形尤其友好的原因是硬件稀缺：绕开本体即可采集，把瓶颈从机器人数量转移到云与人力。
- 定位对照：与 ARMADA（AR 无机器人采集）思路相通、规模更大，两者同属"降低采集硬件依赖"这条线。
- 适用边界：Sim2Real 证据只有两个任务，但差距很明显——换到公共厨房时真实数据策略 0%，DART 策略 50% / 35%；受限因素在于任务能否被物理引擎仿真。

## 局限与风险

- **受限于物理引擎**：切洋葱等无法仿真的任务不能在 DART 中演示，可变形物体仍难仿真。
- **需要把场景导入仿真**：真实场景需先扫描 / 建模成仿真资产。
- **评测规模有限**：Sim2Real 只有 2 个任务，每种设置下的试验数较少。
- **依赖 Apple Vision Pro 与云仿真**：采集端硬件门槛从机器人转移到 AR 头显与云资源。
- **DexHub 仍是愿景**：作为公共数据仓库的价值取决于社区参与规模。
- **开源边界**：未能核实代码；源码运行时序图 **不适用**。

## 与其他页面的关系

- 分类父节点：[paper-notebook-category-06-manipulation](../overview/paper-notebook-category-06-manipulation.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- 任务语境：遥操作：[teleoperation](../tasks/teleoperation.md)
- Sim2Real：[sim2real](../concepts/sim2real.md)
- 仿真侧视觉增广：[domain-randomization](../concepts/domain-randomization.md)
- 下游策略 ACT：[action-chunking](../methods/action-chunking.md)
- AR 无机器人采集的对照：[paper-notebook-armada-augmented-reality-for-robot-manipulation](./paper-notebook-armada-augmented-reality-for-robot-manipulation.md)

## 参考来源

- [humanoid_pnb_dexhub-and-dart.md](../../sources/papers/humanoid_pnb_dexhub-and-dart.md)
- 深读笔记：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/DexHub_and_DART__Towards_Internet_Scale_Robot_Data_Collection/DexHub_and_DART__Towards_Internet_Scale_Robot_Data_Collection.html>
- 论文：<https://arxiv.org/abs/2411.02214>
- 论文正文（用户研究、Table IV、讨论节）：<https://arxiv.org/html/2411.02214>

## 推荐继续阅读

- [机器人论文阅读笔记：DexHub and DART](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/06_Manipulation/DexHub_and_DART__Towards_Internet_Scale_Robot_Data_Collection/DexHub_and_DART__Towards_Internet_Scale_Robot_Data_Collection.html)
- DexHub：<https://dexhub.ai/project>
