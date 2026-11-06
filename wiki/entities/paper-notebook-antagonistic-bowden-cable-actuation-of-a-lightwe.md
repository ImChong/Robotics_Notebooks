---
type: entity
tags: [paper, humanoid-paper-notebooks, paper-notebook-stub]
status: complete
updated: 2026-11-06
arxiv: "2512.24657"
related:
  - ../overview/paper-notebook-category-12-hardware-design.md
  - ../overview/humanoid-paper-notebooks-index.md
  - ../concepts/dexterous-hand-payload-core-metrics.md
  - ../concepts/dexterous-hand-industry-core-players.md
  - ../concepts/dexterous-kinematics.md
  - ../entities/paper-phd-hand-printable-humanoid-dexterous-hand.md
  - ../entities/humanoid-actuator-102-gear-reflected-inertia.md
sources:
  - ../../sources/papers/humanoid_pnb_antagonistic-bowden-cable-actuation-of-a-lightwe.md
summary: "用 拮抗式 Bowden 缆绳 + 滚动接触关节优化 把\"驱动电机\"全部搬到躯干，手部远端只剩 236 g 结构件却仍能输出 >18 N 指尖力、抓起 >100 倍自重 的负载 —— 给\"手臂载荷不够、手却必须像人手\"的人形机器人一条可工程化的路。"
---

# Antagonistic Bowden-Cable Actuation of a Lightweight Robotic Hand

**Antagonistic Bowden-Cable Actuation of a Lightweight Robotic Hand**（arXiv:2512.24657，Min、Kim、Shim，KAIST，RA-L 风格的 letter，2025-12-31 提交）收录于 [Robot Learning Paper Notebooks](https://imchong.github.io/Robot_Learning_Paper_Notebooks/index.html)（分类：12_Hardware_Design）。本页基于 **论文原文全文**（6 页，含参考文献，无独立附录）编写；Paper Notebooks 深读笔记未作为依据。

## 一句话定义

用 拮抗式 Bowden 缆绳 + 滚动接触关节优化 把"驱动电机"全部搬到躯干，手部远端只剩 236 g 结构件却仍能输出 >18 N 指尖力、抓起 >100 倍自重 的负载 —— 给"手臂载荷不够、手却必须像人手"的人形机器人一条可工程化的路。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| DOF | Degree of Freedom | 自由度；本手共 20 DOF（5 指，拇指含 CMC/MCP/IP 等） |
| RCJ | Rolling Contact Joint | 滚动接触关节：两个曲面滚动啮合形成转动，无需轴承或销轴 |
| ROM | Range of Motion | 关节活动范围；论文 Table II 给出各关节 0–100° 等区间 |
| MCP | Metacarpophalangeal (joint) | 掌指关节；四指各有 2 DOF（屈伸 + 桡尺偏） |
| CMC | Carpometacarpal (joint) | 腕掌关节；拇指基部，与 MCP 共同决定对掌能力 |
| PTFE | Polytetrafluoroethylene | 聚四氟乙烯；Bowden 线的内衬套管，降低摩擦 |
| PLA | Polylactic Acid | 聚乳酸；手部主体 3D 打印材料，摩擦系数低，需局部贴橡胶带 |
| FDM | Fused Deposition Modeling | 熔融沉积成型；手指与执行器模块的打印工艺 |
| RMSE | Root-Mean-Square Error | 均方根误差；用于比较扰动前后指尖轨迹一致性 |

## 为什么重要

- 人形机器人的手要同时满足大抓握力、高速度、多自由度与轻量化，但这些指标通常需要更重的执行器与更粗的传动，反过来压缩手臂的有效负载（论文 §I）。
- 把驱动器放到躯干、用 Bowden 线远程传力，是对"远端质量 ↔ 驱动能力"这一对冲突的一个折中：远端质量下降，手臂的末端惯量与臂端载荷占用随之下降。
- 论文同时指出两个具体结构性痛点：手掌内塞满驱动器会加长到腕部的力臂、恶化逆运动学；高减速比会放大反射惯量与摩擦，损害反向驱动性与顺应性（§I）。本工作的价值在于把这两点都当作设计约束处理。
- 它是 Paper Notebooks 硬件类清单中的一条"可装到人形上的手"路线，适合与 [灵巧手负载与核心指标](../concepts/dexterous-hand-payload-core-metrics.md) 对照阅读。

## 核心原理

### 输入、机制、输出

- **输入**：每个关节由一对拮抗的 Bowden 线（屈 / 伸，或桡侧 / 尺侧）驱动；每对线由 **一个电机** 通过绕线盘（bobbin）反向缠绕驱动。
- **关键机制 1：滚动接触关节（RCJ）**。关节由两段曲面以虚拟枢轴 $P_i$ 对称滚动形成，屈伸孔距 $\kappa_i$、桡尺孔间距 $\gamma_i$ 决定力矩臂。论文用运动学仿真优化滚动半径 $r_i$，使一对拮抗线的长度变化近似抵消：

$$
\max\left|\Delta c_f + \Delta c_e\right| \approx 0 \quad \text{（全 ROM 内）}
$$

  在多组 $\kappa$、$\beta$ 组合下，残余长度偏差约 0.03 mm（仿真结果，见 Fig. 5），相对各关节线长可忽略。这是"一个电机驱动一对拮抗线而不需两台电机同步"的数学前提。
- **关键机制 2：Bowden 远程传动**。0.75 mm 镀尼龙不锈钢丝穿在内径 1 mm、外径 2 mm 的 PTFE 套管中，套管外加紧绷弹簧（内径 2 mm、外径 3.2 mm）以抗压溃；沿途设导线器防止急弯。布线时尽量保持大弯曲半径，因为紧弯处的法向力会增大摩擦损耗（§III-B）。
- **关键机制 3：拇指与指列布局**。四指的屈伸线锚定在近节指骨，拇指的 MC 相关线经掌骨固定，形成 CMC 与 MCP 的欠驱动耦合；两组平行线共同驱动同一自由度，使基底关节的扭矩约为单线方案的两倍（§II）。拇指与小指之间的斜向弓形（oblique arch）通过额外几何分析放置，并用对掌指数 $J$ 量化：

$$
J = \frac{1}{d^{3}}\sum_{i=1}^{k} w_i v_i
$$

  其中 $d$ 为拇指长度，$w_i$ 取 1，$v_i$ 为拇指与第 $i$ 指的共同可达工作空间（§IV-A）。论文强调 $J$ 仅用于量化，并未用来优化指位。
- **输出**：指尖力、指尖速度、Cutkosky 抓取类型覆盖，以及在躯干-腕部扰动下的指尖轨迹一致性。

### 流程总览

```mermaid
flowchart LR
    subgraph ACT[远端执行器模块: 可装于躯干]
        M[Dynamixel XC330 舵机<br/>每指 3 个, 5 层堆叠共 15 个] --> B[绕线盘 bobbin<br/>拮抗线对反向缠绕, 装配时设预紧]
    end
    B --> BC[30 根 Bowden 线 = 15 对拮抗线对<br/>0.75 mm 钢丝 + PTFE 套管 + 加强弹簧 + 导线器]
    BC --> RCJ
    subgraph RCJ[滚动接触关节 RCJ, 3D 打印手指]
        K[运动学优化滚动半径 r_i<br/>使屈肌与伸肌线长变化之和 ΔcF + ΔcE 约为 0] --> J[单电机驱动一个关节的屈伸 / 桡尺偏]
    end
    J --> F[5 指 20 DOF 仿人尺度手: 手部 236 g]
    F --> E[指尖力 / 速度 / Cutkosky 抓取 / 拇指对掌指数 评测]
```

图依据：论文 §II、§III-A（Fig. 5）、§III-B（Fig. 6）、§III-D。

## 工程实践

| 项目 | 论文数值 / 做法 | 备注 |
|------|----------------|------|
| 手部质量 | 236 g（不含远程执行器与 Bowden 套管） | Table I |
| 连同 0.9 m Bowden 线 | 1262 g | Table I |
| 执行器模块 | 690 g，可放置于躯干 | §II |
| 电机 | Dynamixel XC330-T288-T，23 g，堵转扭矩 0.92 N·m，编码器 4096 counts/rev | Table V |
| 手指驱动 | 每指 3 对拮抗线（3 台电机），5 层堆叠模块共 15 台 | §III-D |
| 线缆参数 | 0.75 mm 钢丝（尼龙涂层）+ PTFE 套管 1/2 mm + 加强弹簧；30 根 = 15 对 | §III-B |
| 预紧 | 绕线盘分两段，通过八边形键槽调整拮抗对线长以设定预紧 | §III-D |
| 线缆布置 | 屈线缠绕在下盘以减小电机轴弯矩 | §III-D |
| 手部尺寸 | 122 × 189 × 198 mm³（解剖姿态），与人手 1:1 | Table I |
| 材料与工艺 | PLA，FDM 打印；手指与掌面局部贴橡胶带提高摩擦 | §II |
| 指尖力（拇指 / 食-小指） | 21.9 ± 0.6 N / 18.0 ± 0.3 N（单一姿态、最大屈曲命令） | Table VI |
| 指尖速度 | 拇指 193 mm/s；食-小指 208 mm/s | Table VI |
| 载荷 | 25 kg 哑铃，power-wrap 抓取，连续 10 次成功 | §IV-C |
| 源码运行时序图 | **不适用**（未在 arXiv 页或论文中找到可运行的官方代码仓库，见下方开源状态） | 步骤 2.5 核查 |

**开源状态（步骤 2.5，入库日核查）**：arXiv 摘要页的 "Code, Data, Media" 区块未提供可用的代码或数据链接（仅有第三方 CatalyzeX 占位）；论文仅给出补充视频 `youtu.be/bZ6SG_xJHlY`。未发现项目主页。因此 **截至 2026-11-06 未找到官方代码、数据或权重**，标注为"未开源 / 未列链接"，不构成可复现的软件入口。

**调试与复现要点**：
- 拮抗线预紧决定零位，应在装配时通过绕线盘键槽调节，而不是只靠软件偏置。
- 线缆沿 phalange 与 knuckle 内部走平滑曲线，并保持弯曲半径大于推荐最小值（§II）。
- 在 0.9 m 级 Bowden 布线中，执行器与手之间的相对运动会改变线长，论文用扰动实验（见下节）检验这一效应。

## 实验与评测

- **指尖力与速度**：力传感器放在完全伸直的指尖附近，下发最大屈曲指令；速度则在无传感器时测量峰值（Fig. 8，Table VI）。
- **对掌指数**：拇指与各指共同工作空间的仿真结果（Fig. 9，Table VII）；论文报告本手 $J = 0.172$，高于其引用的若干既有设计（同表中 Casia 为 0.049）。
- **驱动一致性**：执行器模块被刻意晃动（相对手部的刚体变换扰动），用 AprilTag 标记和外部视觉系统跟踪指尖轨迹，经时间对齐、平滑后比较 RMSE；指尖路径偏差约 5 mm，论文将其归因于欠驱动耦合与指令相同但路径不同的固有差异（Fig. 10）。
- **抓取分类**：复现 Cutkosky 分类的全部 16 类抓取（Fig. 12a–p，Table VIII），对象从 0.012 kg 的尺子到 7.56 kg 的书堆（Platform push）。
- **大负载**：25 kg 哑铃，power-wrap 抓取，连续 10 次未失败（Fig. 11）。

## 与其他工作对比

- 论文 Table III 将本手与 Biomimetic Hand [7]、ACT Hand [8]、SeoulTech Hand [9]、CasiaHand [10] 对比（手部重量、自由度、驱动器数、指尖力、欠驱动方式、对掌指数等）。本页仅引用其中与本手直接相关的数字：手部 236 g、20 DOF、15 台驱动器、指尖力 18 N 量级、对掌指数 0.172。其余对比项在原表中有 "N/A" 与重排的版式，**具体数值（未核实，以原表为准）**。
- 与 Casia / Biomimetic 等 **腱驱动或手内驱动** 方案相比，本工作的差异在于 **把驱动器整体外移** 并用 **单电机拮抗 RCJ** 消除两台电机的同步需求；代价是 Bowden 线的摩擦与线长一致性成为新的主要误差源。
- 同样采用 3D 打印的手可参考 [paper-phd-hand-printable-humanoid-dexterous-hand](../entities/paper-phd-hand-printable-humanoid-dexterous-hand.md)（本页未对其细节做逐项核对，仅作并列阅读入口）。

## 结论

**这只手的取舍非常明确：用拮抗式 Bowden 缆绳把驱动电机整体外移到躯干，换取远端极低的质量——重量不是消失了，而是被转嫁到传动路径与躯干上。**

- 真正起作用的是两件事的组合：**拮抗式 Bowden 缆绳驱动 + 滚动接触关节优化**，使手部远端只剩 236 g 结构件，仍能输出 >18 N 指尖力、抓起 >100 倍自重的负载。
- 它瞄准的是一个具体的人形工程约束——「手臂载荷不够、手却必须像人手」：减掉的是远端质量与对手臂载荷的占用，而不是系统总重（整机连同执行器与线缆约 1.95 kg，由表 I 与 §II 数值推算，未在论文中直接给出）。
- 定位是 **硬件设计（12_Hardware_Design）** 而非学习方法，评价维度是机构学与驱动指标，与策略泛化类工作不可直接比较。
- 论文只给出单一样机、单一评测流程的结果：无耐久性、疲劳、线缆寿命、效率或反向驱动性的量化数据，不宜仅凭上述数字下工程可用性结论。

## 局限与风险

- **摩擦与效率未量化**：论文承认紧弯会增加 Bowden 线摩擦（§III-B），但未给出效率、迟滞或摩擦补偿结果；与早期 tendon-sheath 研究 [11] 所述的非线性问题相比，本文没有相应的建模与补偿。
- **线长一致性仅在仿真中验证**：约 0.03 mm 的拮抗长度偏差来自运动学仿真（Fig. 5），实机只有扰动下的指尖轨迹 RMSE（约 5 mm）。
- **材料与寿命风险**：PLA 3D 打印件的摩擦系数低，依赖局部橡胶带（§II）；RCJ 接触面磨损、PLA 蠕变与线缆疲劳均未报告。
- **测试面窄**：力测试为单一姿态的最大屈曲，速度测试为无负载，25 kg 抓取为单一 power-wrap 姿态；没有连续作业、冲击或温升数据。
- **拇指与指间协同未完成**：作者自述 CMC 的滚动几何（尺桡向运动、被动滚动）与类似 first dorsal interosseous 的拮抗布线仍是未来工作（§V）。
- **未接入学习或遥操作**：论文不含策略学习、遥操作或灵巧操作基准；把它用于 Sim2Real 或 VLA 数据采集需要额外验证。
- **人类手自由度口径**：论文正文写"人手约 23 DOF"，而 Table I 与本手的比较均采用 20 DOF。两者口径不同，比较时需注意。

## 关联页面

- 分类父节点：[paper-notebook-category-12-hardware-design](../overview/paper-notebook-category-12-hardware-design.md)
- 总索引：[humanoid-paper-notebooks-index.md](../overview/humanoid-paper-notebooks-index.md)
- 灵巧手负载与指标框架：[灵巧手负载与核心指标](../concepts/dexterous-hand-payload-core-metrics.md)
- 灵巧手运动学背景：[Dexterous Kinematics](../concepts/dexterous-kinematics.md)
- 同类 3D 打印腱驱动手：[paper-phd-hand-printable-humanoid-dexterous-hand](../entities/paper-phd-hand-printable-humanoid-dexterous-hand.md)
- 远端执行器的反射惯量与减速比权衡：[humanoid-actuator-102-gear-reflected-inertia](../entities/humanoid-actuator-102-gear-reflected-inertia.md)
- 灵巧手产业格局：[灵巧手产业核心玩家](../concepts/dexterous-hand-industry-core-players.md)

## 参考来源

- [humanoid_pnb_antagonistic-bowden-cable-actuation-of-a-lightwe.md](../../sources/papers/humanoid_pnb_antagonistic-bowden-cable-actuation-of-a-lightwe.md)
- 深读笔记（策展，未作为本页依据）：<https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/Antagonistic_Bowden-Cable_Actuation_of_a_Lightweight_Robotic_Hand/Antagonistic_Bowden-Cable_Actuation_of_a_Lightweight_Robotic_Hand.html>
- 论文：<https://arxiv.org/abs/2512.24657>
- [arXiv PDF](https://arxiv.org/pdf/2512.24657)：2026-11-06 已通读全文（6 页，含方法 §II–III、实验 §IV、结论 §V 与参考文献 [1]–[30]；无独立附录）。
- 补充视频（论文给出）：<https://youtu.be/bZ6SG_xJHlY>

## 推荐继续阅读

- [论文 arXiv 页](https://arxiv.org/abs/2512.24657)（版本 v1，2025-12-31）
- [机器人论文阅读笔记：Antagonistic Bowden-Cable Actuation of a Lightweight Robotic Hand](https://imchong.github.io/Robot_Learning_Paper_Notebooks/papers/12_Hardware_Design/Antagonistic_Bowden-Cable_Actuation_of_a_Lightweight_Robotic_Hand/Antagonistic_Bowden-Cable_Actuation_of_a_Lightweight_Robotic_Hand.html)
- Deshpande et al., ACT Hand 与 Xu & Todorov, Biomimetic Hand（论文参考 [7][8]，用于对照腱驱动手的历史设计）
