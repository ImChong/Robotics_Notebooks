# ViTacWorld: Scaling Visuo-Tactile World Models for Contact-Rich Robot Manipulation

> 来源归档（ingest）

- **标题：** ViTacWorld: Scaling Visuo-Tactile World Models for Contact-Rich Robot Manipulation
- **类型：** paper
- **来源：** arXiv abs / PDF；项目页交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2607.22530>
  - <https://ar5iv.labs.arxiv.org/html/2607.22530>
  - <https://vitacworld.github.io/>
- **作者：** Yunao Huang, Shiyu Sang, Haotao Lu, Suting Ni, Shijie Wu, Ziyang Guo, Ye Shi*, Jingya Wang*
- **机构：** 上海科技大学（ShanghaiTech）；InstAdapt
- **入库日期：** 2026-07-27
- **一句话说明：** 动作条件 **视触觉世界模型**：把触觉当作与主摄/腕摄并列的生成视图；公开真实触觉数据 + 任务对齐仿真预训，再以真机 demo/策略 rollout 微调；生成对齐的视触觉–动作轨迹，用于 **dream 数据增强** 与 **策略评估**。Franka + Xense 四项接触任务上，π₀.₅+触觉平均成功率 **42.5%→67.5%**（Round-2 可达 **80.0%**）。

## 开源核查（2026-07-27）

| 项 | 状态 |
|----|------|
| 项目页 | <https://vitacworld.github.io/> — 方法图、结果表、视频；**Code** 按钮标注 *coming soon* |
| GitHub 实现仓 | **未列 URL**（仅有站点仓 `ViTacWorld/vitacworld.github.io`） |
| 可运行代码 / 权重 | **否** |
| 结论 | **宣称将开源 / 待发布** |

## 核心论文摘录（MVP）

### 1) 问题：接触丰富操作的触觉数据难规模化

- **链接：** <https://arxiv.org/abs/2607.22530> §1
- **摘录要点：** 插装/削皮/插接等任务中关键接触状态常对相机不可见；真机视触觉采集贵、寿命有限；纯视觉 sim-to-real 大，而 **图像式触觉** 在传感器几何对齐时模态间隙更小。目标：用世界模型生成 **策略可用的视触觉–动作 rollout**，而非仅作策略内预测头。
- **对 wiki 的映射：**
  - [ViTacWorld（论文实体）](../../wiki/entities/paper-vitacworld.md)
  - [视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)
  - [Contact-Rich Manipulation](../../wiki/concepts/contact-rich-manipulation.md)

### 2) 方法：view-aware DiT + 两阶段缩放

- **链接：** arXiv §3
- **摘录要点：**
  - 观测 \(o_t=\{I_t^v\}_{v\in\{\mathrm{main},\mathrm{wrist},\mathrm{tactile}\}}\)；动作块为相对末端 + 夹爪；view-presence mask 支持异构数据。
  - 在动作条件机器人视频先验上扩展：流身份嵌入 → AdaLN；流内 SelfAttn + **CrossViewAttn**；触觉为独立生成流。
  - 预训：大规模公开真实视触觉轨迹 + Isaac Sim / Xense 任务对齐仿真（3DGS 场景对齐）；微调：真机专家 demo + 策略 rollout（含失败）。
- **对 wiki 的映射：**
  - [ViTacWorld](../../wiki/entities/paper-vitacworld.md)
  - [VT-WAM](../../wiki/entities/paper-vt-wam-visuotactile-contact-rich.md) — 视触觉预测但主攻 WAM/策略耦合

### 3) 真机四任务与主结果

- **链接：** arXiv §4；项目页；Appendix A
- **摘录要点：**
  - 平台：Franka Panda + Robotiq 2F-85；指端 Xense；外视 RealSense D435 + 腕部 ZED Mini；FACTR 遥操作。
  - 任务：Charger Plugging / Cucumber Peeling / U-Block Insertion / Cuboid Insertion；300 专家 + 每任务 50 策略 rollout 微调 WM；筛 200 成功 dream。
  - Expert only → +Round-1 dream：π₀.₅+触觉平均 **42.5→67.5**；Round-2 进一步到 **80.0**。
  - 策略评估：同一策略实机 Avg **67.5** vs ViTacWorld 多数票想象 **57.5**（偏保守）。
- **对 wiki 的映射：**
  - [ViTacWorld](../../wiki/entities/paper-vitacworld.md)
  - [Manipulation](../../wiki/tasks/manipulation.md)
  - [Ctrl-World](../../wiki/entities/paper-ctrl-world.md) — 视觉 WM 做策略评估对照

### 4) 局限

- **链接：** arXiv §6
- **摘录要点：** 成功 dream 筛选仍部分依赖人工；未来拟用 VLM 自动过滤。**代码截至入库日未发布。**
- **对 wiki 的映射：**
  - [ViTacWorld](../../wiki/entities/paper-vitacworld.md)
