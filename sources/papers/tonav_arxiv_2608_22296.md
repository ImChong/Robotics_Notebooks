# TONAV（任务导向四足移动操作）

> 来源归档（ingest）

- **标题：** TONAV: Task-Oriented Navigation and Action-Velocity Chunk Learning for Articulated Object Quadrupedal Mobile Manipulation
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.22296>
- **机构：** 湖南大学（Hunan University）
- **作者：** Haoran Lin、Mingyu Yang、Pengfei Qi、Kehan Chen、Qiang Diao、Liangji Zeng、Wenrui Chen、Yaonan Wang、Kailun Yang
- **项目页：** <https://haochen611.github.io/TONAV>
- **入库日期：** 2026-08-30
- **一句话说明：** 把任务导向导航与位置–速度动作块统一起来，填补「到达目标」与「稳定接触操作」之间的空档。

## 核心摘录（MVP）

### 1) 可达 ≠ 可操作

- **摘录要点：** 现有方法常在靠近目标后结束导航，留下可达但不可操作的构型；跟踪滞后、抖动和接触不稳限制持续交互。
- **对 wiki 的映射：**
  - [TONAV](../../wiki/entities/paper-tonav.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)

### 2) 三段管线

- **摘录要点：** (1) 位置–速度耦合遥操作采集平滑示范；(2) 视觉语言推理拆子目标并持续把底座调到操作就绪位姿；(3) 动作–速度块联合建模关节位置及其时间变化，用速度监督改善持续接触。
- **对 wiki 的映射：**
  - [TONAV](../../wiki/entities/paper-tonav.md)
  - [Action Chunking](../../wiki/methods/action-chunking.md)

### 3) 真机任务

- **摘录要点：** 关抽屉、放下马桶盖、开灯等铰接物体。项目页对比 TONAV / DP / ACT，并做有无 P–V 控制与不同 LLM（Doubao-Seed-2.1-Pro vs Qwen-3.7-Max）导航消融。
- **对 wiki 的映射：**
  - [TONAV](../../wiki/entities/paper-tonav.md)

### 4) 开源状态（截至 2026-08-30）

- **摘录要点：** **部分开源 / 待发布**。项目页写 Code (Learning, Coming Soon) 与 Code (Teleop)。`haochen611/TONAV` 仅为 GitHub Pages（`index.html` / `static`），无训练脚本。

## 当前提炼状态

- [x] 项目页与 arXiv 摘要对齐
- [x] wiki 映射：`wiki/entities/paper-tonav.md` 新建
