# Fast Generative Grasping via Lie Group-Constrained MeanFlow（arXiv:2608.26076）

> 来源归档（ingest）

- **标题：** Fast Generative Grasping via Lie Group-Constrained MeanFlow
- **短名：** GraspMF
- **类型：** paper / manipulation / grasp-synthesis / generative-model / meanflow / lie-group / flow-matching
- **arXiv：** <https://arxiv.org/abs/2608.26076>
- **PDF：** <https://arxiv.org/pdf/2608.26076>
- **HTML：** <https://arxiv.org/html/2608.26076v1>
- **项目页：** 无
- **代码：** 论文与 arXiv 页未列 GitHub / Hugging Face；检索未见官方仓库 → **确认未开源**
- **作者：** S. Talha Bukhari、Yi Wei、Ruiqi Ni、Zachary Kingston、Aniket Bera
- **机构：** 普渡大学（Purdue University）计算机科学系，West Lafayette, IN
- **版本：** arXiv:2608.26076v1（2026-08-26）
- **入库日期：** 2026-09-13
- **一句话说明：** 在乘积李群 $\mathcal{G}=\mathrm{SO}(3)\times\mathbb{R}^3$ 上做 **Lie Group-constrained MeanFlow（GraspMF）**：**半群一致性 + 黎曼 CFM 锚定** 双目标，$\leq 5$ NFE 达到 ACRONYM 上扩散/流基线级 SR/EMD，毫秒延迟（最高 **39×** 加速）；Franka 真机零微调部署。

## 摘要级要点

- **问题：** 抓取是 **SE(3) 上的多模态分布**；扩散与 flow 多步采样质量高但 NFE 大，难满足闭环重规划延迟。
- **方法（GraspMF）：** 抓取位姿参数化为 $(R,p)\in\mathcal{G}=\mathrm{SO}(3)\times\mathbb{R}^3$（非完整 SE(3) 群元素，但分离旋转/平移积群）；**端点预测器** $X_\theta(H,s,t)$ 经群指数/对数映射导出 **平均速度** $\bar u_\theta$ 与 **流映射** $\Phi_\theta$；训练 = **(i) 纯代数半群（flow-map）一致性** + **(ii) 对角 $t=s$ 处黎曼 Conditional Flow Matching 锚定** + 辅助 **SDF 回归**（SE3Dif 骨干）；推理 **T=1–5** 步。
- **仿真（ACRONYM · Isaac Gym）：** open→approach→close→lift + 5 s shake 计 **SR**；相对 GT 分布 **EMD** 衡量覆盖。ID/OOD split；10 类物体形状（Book/Bottle/Bowl/…）。
- **主结果（T=5，RTX 5080，每物体并行 100 抓取 batch）：** **ID SR 87.40% / OOD 71.73%**（均为最高）；**OOD EMD 0.4191（最佳）**、ID EMD 0.3702；**15.5 ms** 延迟。**T=1：** ID SR 81.11% / OOD 66.34%，**6.3 ms**。对照 **SE3Dif / VSIGD** 各 **140 NFE**，**EGF 80 NFE**；GraspMF **5 或 1 NFE**。
- **采样预算曲线：** OOD SR 在 $T\geq 5$ 平均 **70.15%** 且各预算下延迟最低；SR–latency Pareto 前沿。
- **消融（T=5）：** 去 SDF、恒定半群权重、Gram–Schmidt 替 SVD、分解目标（Zhong et al.）均伤 SR；完整 GraspMF 设计必要。
- **部分观测：** 单视角 raycast 残缺几何下仍稳健（相对基线）。
- **真机（零微调）：** **Franka Research 3 + Franka Hand + 腕部 Orbbec Femto Mega RGB-D**；Black/Red Mug、Gray Bowl 各 10 次。**GraspMF T=5：** **9/10、9/10、10/10**（Table V）。对照 SE3Dif(T=70) 3/6/2，BRIDGE(T=40) 10/8/7，EGF(T=20) 10/8/10，VSIGD(T=70) 6/9/9。
- **开源（截至 2026-09-13）：** 无项目页、无 GitHub/HF；未建 `sources/repos/` / `sources/sites/`。

## 核心摘录（面向 wiki 编译）

### 方法命名与流形

- 论文称完整框架 **Lie Group-constrained MeanFlow**，实现短名 **GraspMF**。
- 积群 $\mathcal{G}=\mathrm{SO}(3)\times\mathbb{R}^3$ 上 bi-invariant 度量；左平凡化把切向量映到李代数 $\mathfrak g$ 做残差。
- 半群约束为 **纯前向代数**（群 exp/log + 网络前向），无 $\mathrm d\exp^{-1}$ 等高方差微分项（继承 Woo et al. 黎曼 MeanFlow 思路）。

### Table I 量级（仿真 · T=5 · GraspMF）

| 指标 | ID | OOD |
|------|----|-----|
| SR | **87.40%** | **71.73%** |
| EMD | 0.3702 | **0.4191** |
| 延迟 | 15.5 ms | （同设置） |

### 基线（仿真对照）

| 方法 | 范式 | 典型 NFE | 备注 |
|------|------|----------|------|
| SE3Dif | SE(3) 扩散场 / DSM | 140 | 每步 predictor–corrector 2 NFE |
| VSIGD | 形状推断 + 扩散 | 140 | 延迟 ~1124 ms |
| EGF (EquiGraspFlow) | SE(3) 等变 flow | 80 | RK4 4 NFE/步；~188 ms |
| BRIDGE | 少步加速扩散 | 变化 | $T<10$ 不稳定 |
| **GraspMF** | **积群 MeanFlow** | **1–5** | SE3Dif 骨干 + 少步 |

## 开源核查（步骤 2.5）

无独立项目页。arXiv 摘要页、HTML 全文与用户指定检索均未发现 GitHub / Hugging Face / 模型权重链接。论文未写 "code will be released"。→ **确认未开源**。

## 对 wiki 的映射

- 升格 [GraspMF 论文实体](../../wiki/entities/paper-graspmf.md)
- 交叉：[抓取位姿估计](../../wiki/methods/grasp-pose-estimation.md)、[李群/刚体旋转](../../wiki/formalizations/lie-group-rigid-body-motions.md)、[黎曼流形与切空间](../../wiki/formalizations/riemannian-manifold-tangent-space.md)、[RoamFlow](../../wiki/entities/paper-roamflow.md)（MeanFlow 导航对照）、[Manipulation 任务](../../wiki/tasks/manipulation.md)

## 当前提炼状态

- [x] 方法、主表数量级、真机 Table V、开源结论
- [x] wiki 实体与交叉引用
