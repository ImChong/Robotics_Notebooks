# 9篇开源具身智能论文看懂VLA、预测控制与双臂抓取

> 来源归档（blog / 微信公众号）

- **标题：** 9篇开源具身智能论文看懂VLA、预测控制与双臂抓取
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/e0yXB8Rz4ma3CCPX8HN2CQ
- **发表日期：** 2026-08-24
- **入库日期：** 2026-08-24
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + `wechat-article-for-ai`（Camoufox）；`--no-images`；Jina Reader 对该链接触发微信 CAPTCHA，未采用
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_9_papers_vla_predict_grasp_2026-08-24.md`](../raw/wechat_embodied_station_9_papers_vla_predict_grasp_2026-08-24.md)
- **一句话说明：** 汇总 9 篇近期机器人/具身论文，主线覆盖 VLA 跨本体适配、动态反应、层级/视频世界模型、双臂与灵巧抓取、平面物体基准、无障碍 CPS 与低成本机械臂；**9/9 均有独立 `paper-*` 详情节点**（本 ingest **新建 6**、**复用 3 既有 complete**；同一 arXiv **不重复造页**）。

## 核心摘录（归纳，非全文）

文内判断：研究重心从「扩大模型与数据」转向「补齐部署闭环」——自生成数据缓解本体差异，未来预测与系统加速应对动态环境，几何与逻辑约束提高可执行性，再以基准、传感器融合与任务工程验证真实价值。

### 9 篇 → 本库节点

| # | 论文 | arXiv | 开源结论（入库日） | wiki |
|---|------|-------|-------------------|------|
| 01 | Self-Demonstrated Generative Control | [2608.19490](https://arxiv.org/abs/2608.19490) | **确认未开源**：项目页无 GitHub/权重链 | [paper-self-supervised-control](../../wiki/entities/paper-self-supervised-control.md) |
| 02 | CPS4All（无障碍赛博物理系统） | [2608.19422](https://arxiv.org/abs/2608.19422) | **不适用**：UIST 工作坊提案，无算法仓 | [paper-cps4all](../../wiki/entities/paper-cps4all.md) |
| 03 | PartialBiGrasp | [2608.19188](https://arxiv.org/abs/2608.19188) | **部分开源**：架构仓已建，权重/训练 **TODO** | [paper-partialbigrasp](../../wiki/entities/paper-partialbigrasp.md)（既有 complete） |
| 04 | ReflexVLA | [2608.14379](https://arxiv.org/abs/2608.14379) | **宣称录用后开源**：项目页 Code After acceptance | [paper-reflexvla](../../wiki/entities/paper-reflexvla.md)（既有 complete） |
| 05 | FlatLab | [2608.14049](https://arxiv.org/abs/2608.14049) | **待发布**：摘要/项目页写明 code 将公开，尚无 URL | [paper-flatlab](../../wiki/entities/paper-flatlab.md) |
| 06 | hint² | [2608.13678](https://arxiv.org/abs/2608.13678) | **待发布**：匿名项目页，无 GitHub 链 | [paper-hint2](../../wiki/entities/paper-hint2.md) |
| 07 | DreamX-Phi 1.0 | [2608.13489](https://arxiv.org/abs/2608.13489) | **部分开源**：占位 README，权重待赛后 | [paper-dreamx-phi](../../wiki/entities/paper-dreamx-phi.md)（既有 complete） |
| 08 | Arm-Aware DexGrasp | [2608.16351](https://arxiv.org/abs/2608.16351) | **待发布**：匿名 RA-L 页，无代码链 | [paper-arm-aware-dexgrasp](../../wiki/entities/paper-arm-aware-dexgrasp.md) |
| 09 | 4-DoF 视觉笔具分拣 | [2608.15968](https://arxiv.org/abs/2608.15968) | **已开源** GitHub 全栈 | [paper-4dof-pen-sorting](../../wiki/entities/paper-4dof-pen-sorting.md) |

### 文内要点速记

1. **Self-Demonstrated Control** — 零样本 VLA 在线 rollout 作自监督微调数据，兼顾旧能力与新手势；ALOHA + RoboTwin 新基准。
2. **CPS4All** — 可穿戴/机器人/XR/智能环境统一无障碍与能力增强框架；社区议题非算法 SOTA。
3. **PartialBiGrasp** — 局部点云隐式补隐藏几何 → 力闭合双臂抓取对。
4. **ReflexVLA** — ReflexBench 延迟感知动态任务 + 未来预测/时序融合/CUDA Graph 加速。
5. **FlatLab** — 策略生成器 + 动作原语执行；刚/可变形平面物体仿真基准。
6. **hint²** — 高低层世界模型在推理时引导 LTL 进度与安全；CALVIN + UR5e。
7. **DreamX-Phi** — 动作条件视频 WM；PRoPE 式 SE(3) + depth/SAM3/V-JEPA。
8. **Arm-Aware DexGrasp** — 机械臂无关扩散抓取模型 + 推理时臂/环境约束引导。
9. **4-DoF 笔具分拣** — YOLO11n-OBB + 纠偏扫动补偿缺失腕部自由度；约 200 美元机械臂。

## 对 wiki 的映射

- **9/9 独立详情节点**：每篇对应唯一 `wiki/entities/paper-*.md`；静态站 `detail.html?id=entity-paper-…` 均可直达。
- **本 ingest 新建 6** 个实体；**3 篇**在先前 ingest 已有 complete 页 → **只回链博客，不新建第二节点**。
- 阅读坐标：[VLA·预测·抓取 9 篇技术地图](../../wiki/overview/vla-predict-grasp-9-papers-technology-map.md)（横切面索引，非详情替代）。
- 交叉：[VLA](../../wiki/methods/vla.md)、[双臂操作](../../wiki/tasks/bimanual-manipulation.md)、[Manipulation](../../wiki/tasks/manipulation.md)、[生成式世界模型](../../wiki/methods/generative-world-models.md)。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 9 篇独立节点核查（6 新建 / 3 既有 complete / **0 重复 arXiv 节点**）
- [x] 项目页与仓库开源状态核查（步骤 2.5）
