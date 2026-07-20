# Miniature Deep-Sea Morphable Robot with Multimodal Locomotion（Science Robotics, 2025）

> 来源归档（ingest）

- **标题：** Miniature deep-sea morphable robot with multimodal locomotion
- **类型：** paper / soft robotics / deep-sea / morphable / pressure adaptation / multimodal locomotion
- **期刊：** Science Robotics, 2025
- **DOI：** <https://doi.org/10.1126/scirobotics.adp7821>
- **项目页 / GitHub：** 截至 2026-07-20，**未见官方代码仓库或完整设计文件公开**；论文 Supplementary 含材料配方与驱动参数，完整控制代码未发布。
- **作者：** （通讯作者：Wen Li 文力、Ding Xilun 丁希仑，北京航空航天大学）
- **机构：** 北京航空航天大学（Beihang University）机器人研究所
- **平台：** 无缆绳自由软体机器人；无刚性耐压舱；游泳/爬行/滑翔三模态；深海现场测试
- **代码与数据：** **未开源**（截至 2026-07-20）；论文 Supplementary 含结构参数与材料；控制代码及硬件图纸未公开
- **入库日期：** 2026-07-20
- **一句话说明：** 提出无刚性耐压舱的**仿生深海软体可变形机器人**：体压等效（pressure-tolerant）设计允许其在极端水压下正常工作，融合游泳、底部爬行与滑翔三种运动模态，在深海现场（包括深渊级压力环境）完成真实环境验证，通讯作者为文力与丁希仑（北航）。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| DOI | [10.1126/scirobotics.adp7821](https://doi.org/10.1126/scirobotics.adp7821) | Science Robotics 原文 |
| 深海软体机器人背景 | Li et al. (2021) Science Advances *A bioinspired soft robot for deep-sea* | 北航同组前序工作 |
| 运动任务节点 | [`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md) | 多模态（游/爬/滑翔）运动 |
| 北航文力组系列 | [`wiki/entities/paper-octopus-inspired-esoam-soft-arm.md`](../../wiki/entities/paper-octopus-inspired-esoam-soft-arm.md) | 同组软体臂 Science Robotics 2023 |
| 北航文力组系列 | [`wiki/entities/paper-aerial-aquatic-remora-hitchhiking-robot.md`](../../wiki/entities/paper-aerial-aquatic-remora-hitchhiking-robot.md) | 同组两栖机器人 Science Robotics 2022 |

## 摘要级要点

- **问题：** 传统深海机器人（ROV/AUV）依赖刚性耐压舱，随深度增加舱体体积/重量呈立方比例增加，且固定形态无法适应复杂地形；现有软体机器人研究多停留在浅水或实验室环境，无法承受深渊级（>6000 m）静水压（>60 MPa）。
- **核心设计理念——压力等效（Pressure-Tolerant）：** 摒弃刚性耐压舱，转而使用**内外压力平衡设计**：机器人内腔充液（或开放式填充），使内部压力与外部水压自动平衡；电子器件采用耐压封装，驱动器选用压力不敏感的介电弹性体（DEA）或液压驱动；结构材料选取在高压下力学性能变化极小的超弹性硅胶与软性聚合物。
- **三模态运动：**
  1. **游泳（Swimming）：** 翼状鳍面谐振扑动，低频（1–3 Hz）产生推力；在中水层高效巡航。
  2. **爬行（Crawling）：** 底部多腿（仿多足/海参运动）接触基底行走；可在凹凸不平的海底地形上稳定移动。
  3. **滑翔（Gliding）：** 利用机身姿态调整产生被动升力，配合有限推力实现低能耗定向移动；类似水下滑翔机（AUG）策略。
- **深海现场测试：** 在深海（具体深度见论文，含深渊级极端压力段）进行了游泳与爬行验证，证明结构完整性与运动功能性。
- **局限：** 功率密度较低（软驱动器效率不如刚性电机）；通信依赖脐带缆或声学通信，自主性受限；完整系统未开源。

## 核心摘录（面向 wiki 编译）

### 1) 压力等效设计原理

| 设计层次 | 传统刚性耐压舱方案 | 本文压力等效方案 |
|----------|---------------------|------------------|
| 结构策略 | 外壳承受压差，厚度∝深度 | 内外压力平衡，外壳仅承受应力集中 |
| 材料 | 钛合金 / 玻璃球壳 | 超弹性硅胶 + 软性聚合物 |
| 电子保护 | 密封气舱内常压 | 压力不敏感封装 / 充液浸没 |
| 重量-深度关系 | 重量随深度三次方增长 | 重量几乎与深度无关 |

- **关键突破：** 此前研究（Li et al., 2021, Science Advances）验证了原理，本工作将方案推至**多模态 + 更大深度 + 自由无缆绳**，是压力等效设计在深海探索中的系统化演进。

### 2) 驱动器选择与特性

- **介电弹性体（DEA）：** 高电压（数 kV）驱动薄膜形变，压力变化对其驱动特性影响远小于气压驱动；配合顺应性机构实现鳍面扑动。
- **液压软驱动（备选或补充）：** 利用泵-管系统传递压力，驱动多腿爬行；在高静水压下通过补偿腔维持驱动压差。
- **对 wiki 的映射：** 驱动器选型逻辑可归入"软体机器人极端环境适配"方法节点。

### 3) 多模态运动切换逻辑

```
中水层/开阔水域 → 游泳模态（鳍面扑动，高效率）
接近海底            → 爬行模态（多腿接触，稳定定位）
长距离定向移动    → 滑翔模态（姿态调整 + 被动升力，低能耗）
```

- **切换判据（推测）：** 深度传感器 + 接触传感器触发模态切换；全自主或半自主控制（论文细节见原文）。

### 4) 深海测试要点

- **现场验证意义：** Science Robotics 同类论文中大多为水池测试；本工作完成真实深海（含极端压力段）验证，是软体机器人深海研究的重要里程碑。
- **测试指标：** 运动轨迹记录（机载摄像 + 声学定位）；结构完整性（下潜前后尺寸/性能对比）；多模态切换响应时间。

### 5) 开源状态

- **代码：** 截至 2026-07-20，**无官方开源仓库**；Supplementary 含材料配方与驱动参数，无控制代码或 CAD 文件。
- **对 wiki 的映射：** [`wiki/entities/paper-miniature-deep-sea-morphable-robot.md`](../../wiki/entities/paper-miniature-deep-sea-morphable-robot.md) 局限区块注明。

## 对 wiki 的映射

- 主沉淀：**[`wiki/entities/paper-miniature-deep-sea-morphable-robot.md`](../../wiki/entities/paper-miniature-deep-sea-morphable-robot.md)**
- 交叉：**[`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md)**（多模态运动）
- 北航文力组系列：**[`wiki/entities/paper-octopus-inspired-esoam-soft-arm.md`](../../wiki/entities/paper-octopus-inspired-esoam-soft-arm.md)**、**[`wiki/entities/paper-aerial-aquatic-remora-hitchhiking-robot.md`](../../wiki/entities/paper-aerial-aquatic-remora-hitchhiking-robot.md)**
