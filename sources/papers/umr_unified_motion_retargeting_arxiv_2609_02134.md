# umr_unified_motion_retargeting_arxiv_2609_02134

> 来源归档（ingest）

- **标题：** Unified Motion Retargeting for Humanoids with Learned Point Cloud Correspondence
- **短名：** UMR（Unified Motion Retargeting）
- **类型：** paper
- **来源：** arXiv abs / HTML
- **原始链接：**
  - <https://arxiv.org/abs/2609.02134>
  - <https://arxiv.org/html/2609.02134>
  - <https://arxiv.org/pdf/2609.02134>
- **作者：** Hanyang Cao、Yuetong Fang（共同一作，HKUST-GZ / Noitom）；Taesoo Kwon（共同一作，Hanyang University）；Runyi Yu（Noitom / HKUST）；Ji Ma（HKU）；Jing Tan、Yangchen Zhou、Baoze Du、Yi Gu、Yukang Gao、Ruoli Dai；通讯 Lei Han（Noitom）、Renjing Xu（HKUST-GZ）
- **机构：** 香港科技大学广州校区（HKUST-GZ）；诺亦腾机器人（Noitom Robotics）；汉阳大学（Hanyang University）；香港科技大学（HKUST）；香港大学（HKU）
- **版本：** arXiv:2609.02134（cs.RO，2026-09）
- **入库日期：** 2026-09-04
- **一句话说明：** 在规范 T-pose 上学稠密人–机表面点云对应，再作为约束优化锚点做位姿匹配与接触图直传；不手写骨架关键点映射。定量评测在 Unitree G1。

## 核心摘录

### 1) 问题
- 现有人形重定向多是骨架中心：手工稀疏关键点 / 肢体对。换机要重画映射、重调拟合；稀疏约束也难保细姿态与接触。
- 人体侧越来越多用 SMPL-X / MHR / SOMA 等网格；网格能给出表面几何与接触，但人–机关节不必一一对应。

### 2) 方法
两阶段（Fig. 2）：

1. **点云对应学习（一次、可复用）**  
   对齐 T-pose 上：有序人体点云 \(\mathbf{X}^{h}\) + 无序机器人点云 \(\mathbf{X}^{r}\)。PointNet 式 \(E_\theta\) 编码 \(\mathbf{X}^{r}\)，MLP \(D_\theta\) 对每个人体点预测变形：
   \(\hat{\mathbf{X}}^{r}=\mathbf{X}^{h}+D_\theta(E_\theta(\mathbf{X}^{r}))\)。  
   损失 \(\mathcal{L}_{\mathrm{corr}}=\lambda_c\mathcal{L}_c+\lambda_r\mathcal{L}_r+\lambda_e\mathcal{L}_e\)：对称 Chamfer、排斥、人体测地图边上的变形平滑。索引继承自人体模板，故人体分段权重可跨机复用。

2. **对应引导重定向（逐帧）**  
   对应点绑到网格：人体侧重心坐标随姿态网格走，机器人侧随 FK。最小化位姿残差 \(\mathbf{r}_p\)（位置 + 相对 T-pose 法向）与接触残差 \(\mathbf{r}_c\)（BimArt 式：点到环境最近点的方向向量；自接触用非邻段点代替环境）。  
   约束 Gauss-Newton QP（Clarabel）：关节限位、信任域、地面净空线性不等式。实现挂 [MuJoCo](https://mujoco.org/)。

输入接口：任意「规范 T-pose 网格 + 随时间姿态表面」——SMPL 族、SOMA、绑定角色、扫描人体网格。

### 3) 实验（定量均在 G1）
**统一性（定性 Fig. 5）：** 四种源 × 五台人形（身高 0.75–1.83 m）：MimicKit 角色、BONES-SEED 的 SOMA、LAFAN1 的 SMPL-X、扫描网格 + 自采 MoCap。对应学习与优化公式不变，只改分段/权重可用性。

**吞吐（Table I，LAFAN1 均值；RTX 4070 Ti SUPER + Ultra 7 265KF）：**

| 阶段 | 成本 |
|------|------|
| Stage I 对应（采样 + 测地 + 训练） | **25.79 s**（一次） |
| Stage II 重定向 | **121.26 FPS**；含预处理总吞吐 **65.29 FPS** |

**LAFAN1 跟踪（BeyondMimic，40 条有 Unitree 参考的动作，每格 4096 trial）：**

| 读法 | UMR vs GMR |
|------|------------|
| Fight 成功（Sim 无 DR） | **99.941%** vs 87.603% |
| Fall and GetUp（无 DR） | **96.981%** vs 84.717% |
| Fight Sim2Sim | **95.298%** vs 79.492% |
| 均值 \(E_{\mathrm{g\text{-}mpbpe}}\)（无 DR） | **89.61** vs 198.89 mm |
| 均值 \(E_{\mathrm{mpjpe}}\)（无 DR） | **610.19** vs 758.52（\(10^{-3}\) rad） |

相对 Unitree 官方参考：成功率与体段误差大体持平，**关节角误差均值更低**。

**SONIC + BONES-SEED：** 官方 G1 参考是 SOMA-Uniform→GMR；UMR 直接吃演员特异的 SOMA-Proportional。关 SMPL encoder 时，相对官方参考约 **+10%**（训练末总奖励 / 锚点误差 / 关节误差）。开 encoder 时两边接近（人侧潜空间对齐补了一截）。

**接触任务（Table IV，UMR / OmniRetarget）：**

| 任务 | 成功 | 关节误差 | 物体误差 |
|------|------|----------|----------|
| Carry（OmniContact） | **99.28** / 82.89 | **0.570** / 1.030 | **0.081** / 0.252 |
| Kick | **99.98** / 83.13 | **0.619** / 1.396 | **0.038** / 0.215 |
| Push | 99.99 / 99.87 | **0.630** / 1.043 | 0.048 / 0.049 |
| Stair（GRAIL） | **43.53** / 11.01 | **0.229** / 0.327 | — |
| Slope | **58.32** / 52.37 | **0.152** / 0.179 | — |
| Chair | 75.24 / **79.67** | 0.315 / **0.308** | — |

作者注：OmniContact 终止较松，成功率对跟踪保真不敏感；关节误差降 **40–56%** 才是交互跟踪质量。Chair 上 OmniRetarget 略好，归因其 stance 启发式 / 粘脚硬约束与该分布更合。

**真机（Fig. 3）：** 旋踢；捡球后倒走转弯（动捕）；爬楼后跳下。

### 4) 开源核查（步骤 2.5）
- **用户给出的项目链接即 arXiv：** <https://arxiv.org/abs/2609.02134>。HTML/PDF **未列** GitHub、HF 或独立 `*.github.io`。
- 论文未写 “code will be released”。
- 同团队 [AdaPT 项目页](https://humanoidtennis.github.io/AdaPT/) 写：MoCap「retargeted using **UMR (Unified Motion Retargeting, coming soon)**」。
- **结论（2026-09-04）：待发布。** 无独立项目页、无可运行官方仓。勿把 AdaPT 仓当成 UMR 实现。

### 5) 局限（原文）
需要网格源 + 规范模板。展望：非结构化观测、灵巧手、多智能体。

## 对 wiki 的映射

- 升格 [UMR 论文实体](../../wiki/entities/paper-umr-unified-motion-retargeting.md)
- 更新 [Motion Retargeting](../../wiki/concepts/motion-retargeting.md)、[hub](../../wiki/overview/hub-motion-retargeting.md)、[GMR](../../wiki/methods/motion-retargeting-gmr.md)、[OmniRetarget](../../wiki/entities/paper-hrl-stack-03-omniretarget.md)、[OmniContact](../../wiki/entities/paper-omnicontact-humanoid-loco-manipulation.md)、[BeyondMimic](../../wiki/methods/beyondmimic.md)、[SONIC](../../wiki/methods/sonic-motion-tracking.md)、[AdaPT](../../wiki/entities/paper-adapt.md)
- **勿与** AdaMorph（arXiv:2601.07284，embodiment-aware Transformer「统一重定向」）或 PALUM（arXiv:2601.07272）混页

## 当前提炼状态

- [x] 摘要 + 两阶段公式 + Table I–IV 关键格 + 真机三项
- [x] 开源核查（arXiv-only + AdaPT coming soon）
- [x] wiki 实体与交叉引用
