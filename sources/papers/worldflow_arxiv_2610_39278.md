# WorldFlow: Joint Video and 3D Motion Prediction via Large-Scale Pretraining

> 论文一手来源归档（paper；原入库条目在 2026-11-20 按最新 v2 原文更新）

- **标题：** WorldFlow: Joint Video and 3D Motion Prediction via Large-Scale Pretraining
- **中文：** WorldFlow：通过大规模预训练联合预测视频与 3D 运动
- **类型：** paper
- **作者 / 机构：** Yuhao Lu、Jiongxu Chen 等 12 人；RoboScience / National University of Singapore
- **arXiv：** <https://arxiv.org/abs/2610.39278>（2026-10-29 v1；2026-11-17 v2）
- **HTML 全文：** <https://arxiv.org/html/2610.39278v2>
- **PDF：** <https://arxiv.org/pdf/2610.39278v2>
- **项目页：** <https://worldflow-rbs.github.io/>
- **官方资源仓：** <https://github.com/roboscience-ai/worldflow>（截至 2026-11-20 无可运行模型代码）
- **首次入库：** 2026-10-30
- **本次复核：** 2026-11-20
- **核对范围：** arXiv v2 HTML 正文 §1–5，Table 1–3，附录 A 数据处理/统计、B 课程采样与训练、C 辅助损失、D 视频-运动一致性；核对项目页与官方 README Release Plan。未运行论文实验。
- **通读：** 已完成

## 核心原文摘录（结构化转述，不转载全文）

1. **任务与输入输出（§3.1）：** 单帧 RGB `I0`、语言 `c`、初始 3D 查询点及图像坐标，联合生成未来视频 `I1:T` 与逐帧三维位移 `f1:T`。全部 3D 点以首帧相机坐标系表示；位移累加得到完整轨迹。
2. **方法（§3.2–3.4）：** 从 Cosmos-Nano 预训练 reasoner/视频生成框架出发，用初始点绝对/相对几何编码、对齐视频 latent 网格的时空 RoPE 构造连续位移 token；MoT 全注意力联合预测视频/flow，rectified-flow 监督并添加位移、方向和终点损失。
3. **数据（§4.1、附录 A）：** 17 个来源形成约 1.44M **object-level clips**，49 帧@16FPS，64 个点/物体；Sim7 七仿真来源，Real3 三真实来源，SpatialCog 自采桌面空间关系，以及 MolmoMotion-1M 六来源。自动标注来自 SAM 2/Track4World 等；样本统计不等于去重原视频小时数。
4. **训练（§4.2、附录 B/C）：** 49 帧、288×480；32k SpatialCog warm-start + 51.1k 混合课程 step；64 张 H800；推理 40 步、CFG=2。点轨迹按域固定归一化。
5. **基准（Table 1）：** WorldFlowBench 1,678 样本：Sim7 396（ADE/FDE 0.051/0.094 m）、Real3 882（0.081/0.113 m）、SpatialCog 400（0.060/0.078 m）。PWT=0.564/0.311/0.449；Real3 PWT 不是最佳。相较 flow-only，SpatialCog ADE 0.136→0.060 m、FDE 0.248→0.078 m。
6. **真实任务验证（§4.4、Table 2/3）：** RoboLab Simple 共享操作管线中，WorldFlow-Orch 总成功率 222/576（38.54%），μ₀-Orch 22/576（3.82%）；真机 xArm+XHand 36 次试验中，WorldFlow 轨迹合理 35/36、任务成功 24/36，对照 μ₀ 为 9/36 和 4/36。只替换 flow 预测器，不能把不同协议的榜单策略作为严格控制实验。
7. **一致性与限制（§5、附录 D）：** 需要预设查询点，自动标注有噪声。未来视频和轨迹投影之间在高置信筛选时更一致，检测误差含跟踪/相机估计误差，不能视作物理一致性证明。未来工作包括自动查询点、动作联合生成。

## 源码/复现边界

[官方仓库 README 的 Release Plan](https://github.com/roboscience-ai/worldflow) 当前只完成论文、项目介绍、展示素材；模型实现、推理案例、权重、训练配置、数据准备与 WorldFlowBench 工具均未发布。仓库可访问不等于模型代码已开源。

## 对 wiki 的映射

- [WorldFlow 唯一论文实体](../../wiki/entities/paper-worldflow.md)
- [WAM 世界侧纵深路线 Stage 1](../../roadmap/depth-wam.md)
- [具身智能小站 10-30 期索引](../../wiki/overview/embodied-station-2026-10-30-28-papers-technology-map.md)

## 历史线索

- [具身智能小站 28 篇论文线索](../blogs/wechat_embodied_station_28_papers_event2reward_progress_rewards_2026-10-30.md)，旧稿仅根据二次转述记录趋势、未逐项核验原文表格；以本次 v2 核对为准。
