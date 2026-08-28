# Video2DoorTraversal: Push Door Traversal via Simulated Door Twins

> 来源归档（ingest）

- **标题：** Video2DoorTraversal: Push Door Traversal via Simulated Door Twins
- **短名：** Video2DoorTraversal
- **类型：** paper
- **作者：** Xincheng Tang、Yiji Chen、Youhan Xie、Wanyu Li、Zhengjie Shu、Lai Jiang、Wenkang Hu、Yitong Li、Jinchuang Zhang、Xibin Song、Ruigang Yang
- **机构：** 上海交通大学（SJTU）；山东大学（SDU）；纽娲机器人（NeoWa Robotics）
- **arXiv：** <https://arxiv.org/abs/2608.20251>（v1：<https://arxiv.org/abs/2608.20251v1>）
- **PDF：** <https://arxiv.org/pdf/2608.20251>
- **项目页：** <https://video2doortraversal.github.io/>
- **代码：** 无公开仓库链（见项目页）
- **入库日期：** 2026-08-22
- **加深日期：** 2026-08-28（对照 arXiv v1 全文与项目页重核开源）
- **索引来源：** [具身智能小站 10 篇盘点](../../sources/blogs/wechat_embodied_station_video_contact_control_10_papers_2026-08-22.md)（<https://mp.weixin.qq.com/s/EmC4gNgcQdPX34vxy-qSVQ>）
- **一句话说明：** 单段 RGB 视频重建实例对齐可仿真门孪生（DoorTwin），仿真闭环 agent 生成可执行穿门演示，再训 ArticuACT 双深度策略，在轮足移动操作平台上完成推门穿越。

## 开源状态（步骤 2.5）

- **待发布 / 宣称将开源：** 项目页头部按钮仍为 **Code Coming soon**（复核日 **2026-08-28**）；Footer / Resources 未列 GitHub、Hugging Face 或权重链接。
- **结论：** 以项目页实际链接为准，**无可运行官方训练/推理仓**。勿把论文里的 Isaac Gym / Articraft / ACT 上游栈误写成「本工作已开源」。
- **交叉：** [项目页归档](../sites/video2door-traversal.md) ↔ 本文 ↔ [wiki 实体](../../wiki/entities/paper-video2door-traversal.md)。

## 核心摘录（面向 wiki 编译）

### 摘录 1：任务定位与系统边界

- **链接：** Abstract；§I；Table I
- **摘录要点：** 开门并穿越是长视野、接触丰富的 loco-manipulation：接近 → 解锁把手 → 底盘–臂协同推门 → 窄门洞通过。Video2DoorTraversal 把任务写成 **关节条件化策略学习**：单 RGB 视频给出实例级门孪生，作为重建、专家生成与真机执行的公共任务表示。相对 Human2Sim2Robot / X-SIM / Video2Sim2Real（固定基座刚体 + 额外扫描）、UniDoorManip（开门而非穿越）、Teacher–student 腿式穿门（程序化门 / 给定把手位），本文输入是 **单 RGB 视频**，部署感知是 **机载深度**，覆盖完整穿越。
- **对 wiki 的映射：**
  - [Video2DoorTraversal 实体](../../wiki/entities/paper-video2door-traversal.md) — 任务定义与对照表。
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md) — 轮足移动操作穿门样本。
  - [DoorMan](../../wiki/entities/paper-doorman-opening-sim2real-door.md) — 人形 RGB 开门对照：程序化资产 + 特权教师 vs 单视频孪生 + IL。

### 摘录 2：DoorTwin（度量接地 + Articraft + 参考视角 critic）

- **链接：** §III-A；Eq. (1)–(2)；Table II；Fig. 3–4
- **摘录要点：** DAGE 从无标定 RGB 视频估计度量几何与相机位姿；玻璃/反光处再用 LingBot-Depth 修深度。SAM 3 分割整门后把多帧点云变到门坐标系，PCA 定宽高与法向。VLM 选参考帧，并分割门板/把手得到板相对把手位置 \(\Delta\mathbf{p}_h\)。约束写入 Articraft：框固定、门板 revolute、把手贴在接地位置；先大门体再修小把手。程序化校验只保证可加载，实例相似靠 **参考视角渲染 + visual critic**（轮廓、把手类型、铰链侧、比例、把手位置），不通过就回写生成 agent。几何锁定后 GPT 去光，Tripo 3D 贴外观。20 扇真门上 DoorTwin：SS **94.95**、mIoU **0.972**、VLM Score **56.74**，高于 Articraft / Articulate-Anything / PhysX-Omni。
- **对 wiki 的映射：**
  - [Video2DoorTraversal 实体](../../wiki/entities/paper-video2door-traversal.md) — DoorTwin 机制。
  - [Articraft](../../wiki/entities/articraft.md) — 上游程序化关节资产生成；本文补度量接地与实例 critic。
  - [PhysX-Omni](../../wiki/entities/physx-omni.md) — 资产生成基线之一。

### 摘录 3：仿真闭环 agentic 专家

- **链接：** §III-B；Eq. (3)–(5)；Table IV
- **摘录要点：** 门孪生 \(\Theta_D\) 实例化参数化技能程序 \(\Pi_D=\{(\sigma_j,\eta_j)\}\)，技能集含 BaseMoveTo、EE_Approach、Close_Gripper、Rotate_Handle、Push_Door、Pass、ReleaseAndRetract。Isaac Gym **50 Hz** 并行 rollout；失败时把进度/可行性信号与多视角关键帧交给 agent，诊断后有界改 \(\eta_j\)，局部仿真搜索只接受满足任务/碰撞/运动学约束的参数。成功轨迹以 **25 Hz** 记录。随后在保持孪生名义几何下随机化初位姿、铰链/把手摩擦阻尼、开门阻力、相机外参，深度裁到 \([0.2,1.5]\) m 并加噪声。每扇门 **200** 条成功演示。完整 generate–execute–diagnose–refine 成功率 **85.63%** / 平均 **2.5** 轮，去掉仿真 rollout 掉到 **48.13%**。
- **对 wiki 的映射：**
  - [Video2DoorTraversal 实体](../../wiki/entities/paper-video2door-traversal.md) — 专家数据机。
  - [Isaac Gym / Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md) — 并行仿真后端。
  - [Agentic Real2Sim](../../wiki/entities/paper-agentic-real2sim.md) — 同属 simulator-in-the-loop agent，单位不同（门孪生 vs DROID episode twin）。

### 摘录 4：ArticuACT（双深度 ACT + Plücker + 交互进度）

- **链接：** §III-C；Eq. (6)–(10)；Fig. 5
- **摘录要点：** 基于 ACT。输入前视/腕部深度 \(D_t^f,D_t^w\) 与 9 维本体状态，预测 chunk \(H=100\)、每步 9 维：底盘前向速度、偏航速度、六臂关节、夹爪。相对原 ACT 增加：（1）双相机像素对齐 Plücker 射线写到 **机器人基座系**，轻量 CNN 编码后与 ResNet-18 深度特征晚融合；（2）每个未来 token 额外预测接触 / 把手进度 / 开门进度，仅作辅助监督，不回灌动作解码、不改 9 维控制接口。损失 \(\mathcal{L}_{\mathrm{act}}+\beta\mathcal{L}_{\mathrm{KL}}+\lambda_c\mathcal{L}_{\mathrm{contact}}+\lambda_h\mathcal{L}_{\mathrm{handle}}+\lambda_d\mathcal{L}_{\mathrm{door}}\)。关节命令空间比末端命令更稳；两模块相对 vanilla ACT 在关节空间把穿越成功率抬约 **26.18%**。数据从 50→200 条，仿真穿越 **59.38% → 97.27%**。
- **对 wiki 的映射：**
  - [Video2DoorTraversal 实体](../../wiki/entities/paper-video2door-traversal.md) — 策略设计。
  - [Action Chunking](../../wiki/methods/action-chunking.md) — ArticuACT 是 ACT chunk 在轮足穿门上的几何/交互条件化。

### 摘录 5：仿真、真机与 zero-shot 读法

- **链接：** §IV-C–E；Table III、V；Conclusion
- **摘录要点：** 硬件为 Unitree **A2-W** 轮足 + **Z1** 臂；头/腕各一枚 RealSense D435；低层在 A2-W 机载，视觉与策略在 Jetson Orin NX。仿真 20 扇门 × 256 trial：本文开门 **98.44%** / 穿越 **97.27%**，vanilla ACT **67.58% / 64.84%**，UniDoorManip 开门尚可（74.22%）但穿越掉到 **50.78%**。真机五扇训练门 169/175 = **96.57%**，全程约 **13 s**；结构相近未见三扇 zero-shot **80.95%**（25/35、31/35、29/35），无额外轨迹生成或真机微调。未来工作写明扩展到 **拉门**、更多把手与更多样门型。
- **对 wiki 的映射：**
  - [Video2DoorTraversal 实体](../../wiki/entities/paper-video2door-traversal.md) — 评测与结论。
  - [Sim2Real](../../wiki/concepts/sim2real.md) — 单视频 R2S2R 部署口径。

## BibTeX

```bibtex
@article{tang2026video2doortraversal,
  title   = {Video2DoorTraversal: Push Door Traversal via Simulated Door Twins},
  author  = {Tang, Xincheng and Chen, Yiji and Xie, Youhan and
             Li, Wanyu and Shu, Zhengjie and Jiang, Lai and
             Hu, Wenkang and Li, Yitong and Zhang, Jinchuang and
             Song, Xibin and Yang, Ruigang},
  journal = {arXiv preprint arXiv:2608.20251},
  year    = {2026}
}
```

## 对 wiki 的映射

- 升格并加深 [`wiki/entities/paper-video2door-traversal.md`](../../wiki/entities/paper-video2door-traversal.md)

## 当前提炼状态

- [x] 方法要点与开源核查（2026-08-28 复核项目页仍为 Coming soon）
- [x] wiki 实体与技术地图回链
- [x] 对照 Articraft / PhysX-Omni / DoorMan / ACT / Agentic Real2Sim
