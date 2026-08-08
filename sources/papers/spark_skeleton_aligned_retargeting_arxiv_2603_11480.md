# spark_skeleton_aligned_retargeting_arxiv_2603_11480

> 来源归档（ingest）

- **标题：** SPARK: Skeleton-Parameter Aligned Retargeting on Humanoid Robots with Kinodynamic Trajectory Optimization
- **短名：** SPARK（Skeleton-Parameter Aligned Retargeting；**勿与** Paper Notebooks 中 *SPARK: A Toolbox for Safe Humanoid Autonomy and Teleoperation* 混淆）
- **类型：** paper
- **来源：** arXiv abs / PDF / Legged AI Lab 项目页
- **原始链接：**
  - <https://arxiv.org/abs/2603.11480>
  - <https://arxiv.org/pdf/2603.11480>
  - 项目页：<https://www.leggedai.com/publication/2026_spark/>
  - 项目页 PDF（作者站）：<https://www.leggedai.com/publication/2026_submission3/SPARK_2026.pdf>
- **作者：** Hanwen Wang, Kunzhao Ren（UW–Madison）；Qiayuan Liao, Bike Zhang, Koushil Sreenath（UC Berkeley）；Xiaobin Xiong（原 UW–Madison，现上海创智学院 SII）
- **版本：** arXiv:2603.11480（Submitted 2026-03-12）；站点标注 Preprint / 拟投 IROS 2026
- **入库日期：** 2026-08-08
- **一句话说明：** 两阶段管线——先把任务空间人体运动建成 **可校准 human URDF** 并与目标人形尺寸对齐以降低 IK 误差，再经 **KTO → ID → KDTO** 渐进 kinodynamic TO 产出动力学一致轨迹与关节力矩参考，供 BeyondMimic/IsaacLab 等高动态跟踪（含 side flip）。

## 核心摘录

### 1) 问题与动机
- GMR 类 **root–keyframe 缩放 + 局部偏移** 与底层骨架不一致，跨体型/AMASS 角落易失败，且需大量 IK 权重与朝向偏移调参。
- PHC 等用 SMPL shape 校准到机器人时，机器人尺寸常落在人体分布外，易出不对称肢体/脊柱扭曲。
- SPARK：直接校准 **从任务空间生成的 human URDF**（可解释、可推广到非 SMPL 隐式骨架），再用渐进 TO 恢复动力学可行性与力矩监督。

### 2) 方法要点
1. **Human URDF + 广义坐标（§III-A）：** 骨帧 → 链节；对齐到与机器人根同向的对齐帧 \(A\)；球关节欧拉角序列。
2. **URDF 校准（§III-B）：** 分末端/臂长缩放/腿绝对平移/躯干仿射（含 xz 剪切）/根位置缩放五组；固定机器人与人体格式后，**同一校准可复用、无需 per-motion IK 重调**。
3. **IK（§III-C）：** 在校准 URDF 上回放得任务空间目标；位置全关键点 + 末端朝向；关节位/速限与上一帧正则。
4. **Progressive kinodynamic TO（§IV）：**
   - **KTO：** 双积分动力学；摆动脚离地、首次触地位姿、stance 无滑；点对点自碰；加速度平滑。
   - **ID：** 固定 \((q,v)\) 的单步 QP，全阶 Lagrangian + CWC；为力矩/接触暖启动。
   - **KDTO：** 联合运动学+动力学约束；可选对 \(\tau^*_{\mathrm{KDTO}}\) 加指数力矩跟踪奖励（KDTO+T）。

### 3) 实验
**Table I — IK Empbpe (cm)，AMASS ACCAD，相对 GMR：**

| 方法 | G1 | H1 | T1 | PM01 | Kuavo 4Pro |
|------|----|----|----|------|------------|
| GMR | 9.37 | 12.53 | 6.59 | 9.51 | 19.50 |
| URDF Calibration | **1.60** | **4.40** | **1.85** | **2.47** | **4.71** |
| 相对改善 | 82.9% | 64.9% | 71.9% | 74.0% | 75.8% |

- **运动编辑：** 跳高（CoM 高 ×1.4 → 时间 \(\sqrt{1.4}\) 插值）时 KDTO 最终 Empbpe 显著低于 raw edit / KTO；前后加站立段的速度间断编辑上 KTO≈KDTO。
- **Side flip：** KTO 在倒立段长时间 plateau；KDTO 缩短 plateau；**KDTO+T**（力矩奖励）收敛更快（Fig. 8）。下游 BeyondMimic + IsaacLab @ G1。

### 4) 开源核查（步骤 2.5）
- **项目页：** <https://www.leggedai.com/publication/2026_spark/>（Legged AI Lab / 足智实验室）— 有 Abstract、PDF、Cite、Video；页内可见作者 GitHub 个人页链接，**无论文代码仓 / HF 数据集链接**。
- **论文：** 未给出官方 GitHub；未写 “code will be released” 明确承诺（相对 KDMR）。
- **结论：** **截至 2026-08-08 项目页未列可运行源码 → 未开源**；归档见 [`sources/sites/spark-leggedai.md`](../sites/spark-leggedai.md)。

## 对 wiki 的映射

- 升格 [SPARK（骨架对齐重定向）实体](../../wiki/entities/paper-spark-skeleton-aligned-retargeting.md)
- 项目页 [`sources/sites/spark-leggedai.md`](../sites/spark-leggedai.md)
- 更新 [Motion Retargeting](../../wiki/concepts/motion-retargeting.md)、[Pipeline](../../wiki/concepts/motion-retargeting-pipeline.md)、[hub](../../wiki/overview/hub-motion-retargeting.md)、[GMR](../../wiki/methods/motion-retargeting-gmr.md)、[BeyondMimic](../../wiki/methods/beyondmimic.md)、[KDMR](../../wiki/entities/paper-kdmr.md)
- **勿覆盖** [paper-notebook-spark（安全自主工具箱占位）](../../wiki/entities/paper-notebook-spark.md)

## 当前提炼状态

- [x] 摘要 + URDF 校准 + KTO/ID/KDTO + Table I + side flip
- [x] 项目页核查与站点归档
- [x] wiki 实体页与交叉引用
