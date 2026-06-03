---
type: comparison
tags: [real2sim, sim2real, 3dgs, monocular-video, contact, humanoid, visual-rl, selection]
status: complete
updated: 2026-06-02
related:
  - ../methods/crisp-real2sim.md
  - ../entities/gs-playground.md
  - ../concepts/sim2real.md
  - ../concepts/whole-body-control.md
sources:
  - ../../sources/papers/crisp_real2sim_iclr2026.md
  - ../../sources/sites/crisp-real2sim-project-github-io.md
  - ../../sources/repos/gs_playground.md
summary: "CRISP 与 GS-Playground 是两条互补的 Real2Sim 路线：前者从单目视频用凸平面原语 + 接触补全 + RL 人形闭环还原『接触动力学可仿真』的运动与场景，后者用批量 3DGS 把真实外观渲成『光真实感视觉观测』供视觉 RL；选型取决于瓶颈在接触物理还是视觉 domain gap。"
---

# CRISP vs GS-Playground：Real2Sim 路线选型（接触动力学 vs 光真实感）

两者都把**真实世界**变成**能在仿真里训练策略**的资产，但优化的「真实」是不同维度：[CRISP](../methods/crisp-real2sim.md) 追求**接触动力学可信**（脚–地、臀–椅不穿透、可 rollout），[GS-Playground](../entities/gs-playground.md) 追求**外观光真实感**（缩小 visual domain gap）。它们不是竞品，而是 Real2Sim 资产链上常被**串联**的两段。

## 对比表

| 维度 | [CRISP](../methods/crisp-real2sim.md) | [GS-Playground](../entities/gs-playground.md) |
|------|----------------------------------------|-----------------------------------------------|
| 一句话定位 | 单目视频 → 可仿真的人–场景**接触几何 + 参考运动** | 真实场景 → 批量 3DGS **光真实感视觉观测** |
| 发表 | Wang et al., ICLR 2026 | discoverse-dev, RSS 2026 |
| 输入 | 互联网风格**单目 RGB 视频** | 真实场景采集（**相机矩阵** + 多视角，3DGS 重建） |
| 场景表示 | **凸平面原语**（depth/normal/flow 聚类拟合） | 数百万**各向异性高斯椭球**（splat） |
| 「真实」的维度 | **接触动力学**：可碰撞、不穿透、可站坐 | **外观**：RGB 光真实感、视觉分布对齐 |
| 遮挡处理 | **人–场景接触**推断被挡支撑面（如椅面） | 不补几何，受重建视角覆盖限制 |
| 物理/控制耦合 | **RL 人形控制器**做物理一致性闭环 | 速度冲量求解器 + Rigid-Link Gaussian Kinematics（视觉随刚体） |
| 主要观测产物 | 物理可行的运动 + 可仿真场景几何 | RGB + Depth（峰值约 **10^4 FPS**） |
| 招牌指标 | 跟踪失败率 **55.2% → 6.9%**，RL 吞吐约 **+43%** | 渲染 **10,000 FPS** @ 640×480 |
| 成熟度 | 论文 + 项目页/仓库索引 | 早期预览（核心 Simulator API / Real2Sim 工具链规划中） |

## 何时优先 CRISP

- 瓶颈在**接触与几何**：稠密 mesh / 噪声深度在脚–地、臀–椅等接触处产生伪碰撞，导致跟踪/模仿策略大量失败。
- 数据来源是**单目互联网视频**，没有多视角采集条件，但需要**物理一致的参考运动**。
- 下游是 [全身控制](../concepts/whole-body-control.md) 意义上的人形**接触丰富技能**（坐、起、上下台阶），需要「能 rollout」而非「渲染好看」。

## 何时优先 GS-Playground

- 瓶颈在**视觉 domain gap**：策略吃 RGB 观测，sim 外观失真导致 zero-shot 迁移掉点。
- 有条件做**真实场景 3DGS 重建**，希望视觉分布对齐**内建**、减少 domain randomization 负担。
- 需要**高吞吐视觉 RL** 训练（批量 3DGS 渲染 + 并行物理步进）。

## 互补而非互斥

两条路线天然可以**串联**成一条 Real2Sim 资产链：

- **CRISP 管几何与接触，GS-Playground 管外观**：用 CRISP 得到接触可信的场景原语与参考运动，再叠加 3DGS 外观做光真实感视觉观测，既「跑得动」又「看得真」。
- 都服务 [Sim2Real](../concepts/sim2real.md) 上游：先把真实世界变成**动力学/外观双重一致**的仿真，再在同一物理里训练策略。
- 项目页常把 CRISP 与 VideoMimic 并排做交互对比，把 GS-Playground 与 [Spark](../entities/spark-3dgs-renderer.md) / [Aholo Viewer](../entities/aholo-viewer.md) 等 Web 3DGS 栈对照——评估口径不同，数字只能在各自设定内解读。

## 共同局限

- **都不是端到端「真实世界进、策略出」**：仍依赖重建质量、视角覆盖与下游训练设定。
- **指标不可直接横比**：CRISP 报跟踪失败率/仿真吞吐，GS-Playground 报渲染 FPS，关注维度正交。
- **成熟度均偏早期**：CRISP 为论文阶段，GS-Playground 核心 API 与 Real2Sim 工具链尚未发布。

## 关联页面

- [CRISP（Contact-guided Real2Sim）](../methods/crisp-real2sim.md)
- [GS-Playground](../entities/gs-playground.md)
- [Sim2Real](../concepts/sim2real.md)
- [Whole-Body Control](../concepts/whole-body-control.md)
- [Spark vs Aholo Viewer：Web 大场景 3DGS 渲染选型](./spark-vs-aholo-web-3dgs-renderers.md)

## 参考来源

- [CRISP（ICLR 2026）论文摘录](../../sources/papers/crisp_real2sim_iclr2026.md)
- [CRISP 项目页归档](../../sources/sites/crisp-real2sim-project-github-io.md)
- [GS-Playground 仓库归档](../../sources/repos/gs_playground.md)

## 推荐继续阅读

- [OpenReview：CRISP 论文页](https://openreview.net/forum?id=xlr3NqxUqY)
- [VideoMimic 项目页](https://videomimic.github.io/)
