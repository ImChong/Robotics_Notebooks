# Bistable Soft Jumper Capable of Fast Response and High Takeoff Velocity（Science Robotics, 2024）

> 来源归档（ingest）

- **标题：** Bistable soft jumper capable of fast response and high takeoff velocity
- **类型：** paper / soft robotics / magnetic actuation / bistable mechanism / jumping locomotion
- **期刊：** Science Robotics, August 21, 2024（Volume 9, Issue 93, eadm8484）
- **DOI：** <https://doi.org/10.1126/scirobotics.adm8484>
- **数据集：** <https://doi.org/10.5061/dryad.hdr7sqvsg>（Dryad）
- **作者：** Daofan Tang、Chengqian Zhang、Chengfeng Pan、Hao Hu、Haonan Sun、Huangzhe Dai、Jianzhong Fu、Carmel Majidi*（CMU）、Peng Zhao*（ZJU）（* 通讯作者）
- **机构：** 浙江大学机械工程学院（ZJU）、卡内基梅隆大学（CMU，Carmel Majidi 软体机器人实验室）
- **入库日期：** 2026-07-20
- **一句话说明：** 报告一种 **磁场驱动的双稳态软跳跃机器人**：3D 折叠结构 + 磁化组合，通过磁场脉冲触发两稳态之间的 snap-through 转换，爆发式释放弹性势能，实现 **>2 m/s 起跳速度、<15 ms 响应时间、>108 倍体高跳跃**，并演示两栖管道环境中的全向多模态运动。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 数据 | [Dryad hdr7sqvsg](https://doi.org/10.5061/dryad.hdr7sqvsg) | 论文配套数据 |
| ZJU 报道 | [Dialogue@ZJU](https://www.zju.edu.cn/english/2024/1028/c75130a2980568/page.htm) | 团队访谈，解释设计思路 |
| 媒体报道 | [TechXplore 2024-09](https://techxplore.com/news/2024-09-magnetically-driven-soft-robot-high.html) | 英文二次报道，含图例 |
| 软体 locomotion 对照 | [SoftMimic（顺从全身控制）](../../wiki/entities/paper-notebook-softmimic-learning-compliant-whole-body-control.md) | 刚-软混合控制视角对照 |
| Locomotion 任务 | [Locomotion 任务页](../../wiki/tasks/locomotion.md) | 宏观跳跃/运动分类 |

## 摘要级要点

- **问题：** 现有软跳跃机器人在响应速度（>100 ms）、起跳速度（<1 m/s）和跳跃高度上远落后于刚性机器人；柔性材料虽有抗冲击、顺从性优势，但能量释放机制成为性能瓶颈。
- **核心创新——双稳态结构：**
  - **3D 折叠 + 磁化**：四个磁性材料薄板按特定磁化方向设计，外部脉冲磁场施加磁力矩后触发折叠结构快速翻转（snap-through）。
  - **双稳态弹性储能**：结构在两个稳定几何构型（State 0 和 State 1）之间跳变，每次跳变中储存的弹性势能在几毫秒内爆发释放。
  - **可重复循环**：snap-through 转换形成可重复的能量存储–释放闭环，支持连续跳跃。
- **两种运动模态：**
  - **Interwell 模式（跳跃 Jumping）**：跨越两稳态之间的势垒，高能量释放，对应大跳；
  - **Intrawell 模式（弹跳 Hopping）**：在单一稳态内小幅振荡，小幅快速弹跳。
  - 通过调整外部磁场的 **强度** 和 **持续时间** 切换两种模态。
- **性能指标（相较于已有软跳跃机器人）：**
  - 起跳速度：**>2 m/s**（前沿约 0.5–1 m/s）
  - 响应时间：**<15 ms**（前沿约 100–500 ms）
  - 跳跃高度：**>108 倍体高**（108× body height）
  - 全向跳跃；高度与距离可调
- **演示场景：** 在模拟两栖地形管道（含水面）中完成 **污水清洁任务模拟**，验证复杂环境适应性。
- **尺寸与气动影响：** 测试不同尺寸原型，发现较小机体受空气阻力影响更大（跳跃高度更低），但各尺寸起跳速度基本一致；说明速度性能由弹性机制主导，而非尺寸。

## 核心摘录（面向 wiki 编译）

### 1) 双稳态机制与磁驱

| 要素 | 说明 |
|------|------|
| 结构 | 四块磁性面板 + 3D 折叠铰链，两个稳定几何构型（展开 / 折叠） |
| 磁化方向 | 设计与外部脉冲磁场对齐，施加时产生磁力矩驱动翻转 |
| Snap-through | 穿越势垒的非线性快速转变；弹性应变能在 <15 ms 内释放 |
| 可重复性 | 翻转后恢复原态再充能，形成闭环；可持续跳跃 |

### 2) 模态控制逻辑

- **Interwell（跳跃）**：磁场强度 & 持续时间足以越过势垒 → 大位移跳；
- **Intrawell（弹跳）**：磁场参数不足以翻转 → 在单稳态内弹跳（类昆虫步态）；
- **全向跳跃**：调整磁场方向（偏转角）→ 调整起跳方向，实现全向运动。

### 3) 开源状态

- **无 GitHub 代码仓库**（截至 2026-07-20），设计侧重物理硬件与材料制造；
- 论文配套数据已 deposit 至 **Dryad**（doi:10.5061/dryad.hdr7sqvsg）；
- 制造方法（材料选型、磁化工艺、折叠模具）详见论文 Materials and Methods 节。
- **对 wiki 的映射**：`局限与风险`节写明「未开源制造文件，复现需参照 M&M 节」。

## 源码开放核查（步骤 2.5）

| 类别 | 状态 | 说明 |
|------|------|------|
| 代码 / 控制软件 | **未开源** | 无 GitHub；磁驱软体机器人为硬件为主工作 |
| 制造/设计文件 | **未公开** | 材料与制造工艺在论文 M&M 节描述，但无 CAD/模板文件 |
| 数据集 | **已开源** | Dryad doi:10.5061/dryad.hdr7sqvsg |

## 对 wiki 的映射

- 主沉淀：**[`wiki/entities/paper-bistable-soft-jumper-magnetic.md`](../../wiki/entities/paper-bistable-soft-jumper-magnetic.md)**
- 交叉：**[`wiki/tasks/locomotion.md`](../../wiki/tasks/locomotion.md)**（跳跃运动模式）
- 谱系：**[`wiki/entities/paper-notebook-softmimic-learning-compliant-whole-body-control.md`](../../wiki/entities/paper-notebook-softmimic-learning-compliant-whole-body-control.md)**（软体/顺从控制对照）
