# TravExplorer: Cross-Floor Embodied Exploration via Traversability-Aware 3-D Planning（arXiv:2605.19958）

> 来源归档（ingest）

- **标题：** TravExplorer: Cross-Floor Embodied Exploration via Traversability-Aware 3-D Planning
- **缩写 / 系统：** **TravExplorer**
- **类型：** paper / zero-shot-object-navigation / quadruped / traversability-mapping
- **arXiv：** <https://arxiv.org/abs/2605.19958>（HTML：<https://arxiv.org/html/2605.19958v1>）
- **项目页：** <https://wuyi2121.github.io/TravExplorer/> — 归档见 [`sources/sites/wuyi2121-travexplorer.md`](../sites/wuyi2121-travexplorer.md)
- **代码：** <https://github.com/wuyi2121/TravExplorer>（Apache-2.0；截至入库日为占位仓）— 归档见 [`sources/repos/travexplorer.md`](../repos/travexplorer.md)
- **作者：** Han Zheng、Zhe Chen、Yudong Huang、Haoran Liu、Jinghao Wang、Ming Yang、Tong Qin
- **机构：** 上海交通大学（Shanghai Jiao Tong University）
- **入库日期：** 2026-08-04
- **一句话说明：** 面向四足的跨楼层零样本 ObjectNav：可通行感知 3D 体积图 + 轻量开放词汇语义引导 + 分层跨楼层规划；HM3D/MP3D 4195 episodes，Unitree Go2 真机 50 次试验。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-04）：** [wuyi2121.github.io/TravExplorer](https://wuyi2121.github.io/TravExplorer/) 列 Code → GitHub，并展示系统/真机视频。
- **仓库核查：** [wuyi2121/TravExplorer](https://github.com/wuyi2121/TravExplorer) 含 `README.md`、`LICENSE`（Apache-2.0）、`assets/`；README 写明 **「Code will be released upon acceptance」**，无可运行训练/部署入口。
- **结论：** **宣称将开源 / 占位仓**（许可已声明，实现待发布）。局部规划依赖 [SCAN-Planner](https://github.com/wuyi2121/SCAN-Planner)；定位依赖 [Elevator-LIO](https://github.com/xiaofan4122/Elevator-LIO)。

## 摘录 1：问题设定与贡献（§Abstract–§1）

- **缺口：** 既有 ZSON 多绑平面图与单楼层假设；真实建筑含楼梯、平台与竖直叠层。
- **贡献：** (1) 可通行 frontier 的 3D 探索范式；(2) 概率实例图 + 图像–文本匹配的空间语义引导；(3) TSP 全局巡游 + foothold 引导 3D 搜索 + 竖直约束局部规划；(4) 4195 仿真 + 50 真机验证。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-travexplorer.md`](../../wiki/entities/paper-travexplorer.md)；任务坐标见 [`wiki/tasks/zero-shot-object-navigation.md`](../../wiki/tasks/zero-shot-object-navigation.md)。

## 摘录 2：闭环架构（§System Overview）

- **建图：** 位姿 RGB-D → 体积占据图 + 语义可通行层（地板/楼梯/平台统一度量帧）。
- **语义：** 在线开放词汇分割 → 概率实例记忆；轻量 image-text matching → 空间价值图。
- **规划：** 目标感知 frontier 巡游 → foothold 引导 3D 图搜索 → 在线 review/replan + 竖直约束局部轨迹优化。

**对 wiki 的映射：** 实体页画 flowchart；与 [hierarchical-quadruped-navigation-stack](../../wiki/concepts/hierarchical-quadruped-navigation-stack.md)、[embodied-semantic-cognitive-map](../../wiki/concepts/embodied-semantic-cognitive-map.md) 互链。

## 摘录 3：评测与真机（§Experiments）

- **仿真：** HM3D / MP3D，4195 episodes；相对 ASCENT / VLFM 等报告多楼层 SR 优势（项目页叙述多楼层约 +15.4% 量级）。
- **真机：** Unitree Go2，无先验地图；单楼层 + 跨楼层开放词汇寻物；项目页称整体 SR 约 64%。

**对 wiki 的映射：** 与 [ZONDA](../../wiki/entities/paper-zonda.md)（多楼层动态 ObjectNav）对照：TravExplorer 强调 **可通行 3D 规划 + 四足**，ZONDA 强调 **行人动态 + 轮腿双足**。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-travexplorer.md`**（`## 源码运行时序图` 标注不适用）。
- 新建 `sources/repos/travexplorer.md`、`sources/sites/wuyi2121-travexplorer.md`。
- 任务页 [`zero-shot-object-navigation`](../../wiki/tasks/zero-shot-object-navigation.md) 收入主线实例。
