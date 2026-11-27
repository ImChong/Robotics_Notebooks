# CoDimRecon: Agentic Reconstruction of Sim-Ready 3D Scenes with Deformable Curves, Surfaces, and Volumes

- **作者：** Shuzhao Xie, Lelin Wang, Guying Lin, Zhi Wang, Minchen Li
- **机构：** SIGS, Tsinghua University; Carnegie Mellon University; Genesis AI
- **版本 / 载体：** arXiv:2609.36024v3 [cs.CV]（2026-09-28 提交 v1，2026-10-03 更新 v3）
- **原文：** [arXiv HTML v3](https://arxiv.org/html/2609.36024) / [arXiv Abstract](https://arxiv.org/abs/2609.36024)
- **项目页 / 代码：** [项目页](https://shuzhaoxie.github.io/CoDimRecon/) — **宣称将开源或未列链接**（项目页按钮标注 `Code Soon`，截至 2026-11-27 暂未发布 GitHub 仓库或公开源码；项目页已开放交互式 3D 场景检视器与可变形仿真视频演示）
- **核对范围：** 通读 arXiv HTML 全文（Abstract、§§1–5、Table 1–5、Fig. 1–7、References 与 Appendix A–G 含 Table 6 及 Fig. 8–15，共 1261 行）以及项目页文本（376 行）；未见公开源码，未进行本地复现。
- **通读：** 已完成

---

## 原文核对笔记

- **对 wiki 的映射：** [CoDimRecon](../../wiki/entities/paper-agentic-2609-36024.md)

### 1. 研究动机与问题定义

- 现有组合式三维场景重建方法（优化式如 HoloScene、零样本式如 ReplicateAnyScene / SimRecon、以及智能体式如 VIGA / Lucida / Lumera / LiteReality-Agent）普遍将场景物体视为刚体，忽略了两类真实物理特征：
  1. **余维可变形几何（Codimensional Deformables）：** 线缆、衣物、纸张等细长或薄壁物体在物理仿真中不宜直接离散为普通 3D 四面体实体（厚向网格极密且低阶单元易发生 locking），需要按几何维度分别重建：**1D 曲线**（中心线 + 半径 → 离散弹性杆 DER）、**2D 曲面**（带厚度的流形中面 → 薄壳 Shell）、**3D 体积**（水密闭合体 → 四面体实体 Solid）。
  2. **非弹性/不可逆行为：** 如折纸（paper folding）在卸载后需保持折痕，仅靠弹性本构会回弹，必须引入塑性铰链弯曲（plastic hinge bending）等模型；且静态多视角 RGB 图像无法直接确定真实动力学参数，需在仿真交互中暴露行为失配。

### 2. 三阶段核心流程（Sec. 3 & Appendix B）

1. **几何与生成双先验引导的 Blender 重建（Sec. 3.1, Appendix B.1–B.3）：**
   - **几何上下文：** 使用 VGGT-Omega 估计相机内参、camera-to-world 位姿、逐帧深度图 $D_v$ 与反投影点云，统一全局度量尺度与布局。
   - **网格参考构建：** 使用 CropFormer 提取实体掩码，按 MaskClustering 的 3D 重叠聚类（配合 InstaScene 欠分割过滤）生成跨视角 3D 轨迹；对每条轨迹选取提升后三维表面积 $A_v^o$ 最大的视角 $v^\star$，用 SAM3D 生成规范网格并借助 MASt3R 稠密对应与 Umeyama 迭代对齐（按渲染-跟踪掩码平均 IoU 选最优位姿 $\Theta^\star$）；容器内部遗漏的小物体通过 REST3D 结合 SAM3D 补全。
   - **可编辑图元重建：** 智能体将参考场景导入 Blender，不直接保留高面数生成网格，而是以生成网格为参考、用 Blender 基础图元（cuboid/cylinder/frustum 等）配合 bevel、subdivision 与 lattice 修改器重建紧凑可编辑几何。在 Appendix B.2 的办公椅压缩统计中，SAM3D 参考网格含 178,172 顶点 / 356,084 三角面（22.9 MiB），智能体重建仅用 43 个图元、2,192 顶点 / 1,183 多边形面（610 KiB），实现 $81\times$ 顶点数与 $301\times$ 面数压缩，单侧表面距离均值 0.59% 物高、95 分位 1.53%、五正交视角轮廓 IoU 81.1%–93.7%。
2. **场景细化（Sec. 3.2）：**
   - **渲染-评估-细化闭环（RR）：** 使用迭代重加权最小二乘（IRLS）拟合二次色彩对齐映射 $c' = a c^2 + b c + d$ 作为诊断信号——若对齐后误差大幅下降，说明主要是光照/材质差异；若改善有限，则指向几何或位姿偏差。显式修改 Blender 材质与光照，不烘焙 3DGS 外观。
   - **铰接物体拆解：** 提供涵盖 revolute、prismatic、screw、cylindrical、universal、spherical 六类关节的建模指南，拆解抽屉、柜门、台灯乃至键盘按键与电话按键。
   - **刚体重力稳定：** 导入 MuJoCo 暂时全作刚体在重力下沉降，并将稳定位姿写回规范摆放。
3. **分维度可变形重建与行为验证闭环（Sec. 3.3, Appendix B.4）：**
   - **分会话重建（Category-wise Sessions）：** 分别在独立智能体会话中处理曲线（路径补全 + 半径估计）、曲面（补洞 + 片段合并为带厚度流形壳）、体（水密封闭，高面数物体绑定到低面数闭合代理后四面体化）。混合三类指令在同一会话会明显退化线缆几何精度（Fig. 7 消融）。
   - **可复用仿真技能初始化：** 物理后端采用 IPC 家族无穿透连续碰撞接触求解器（具体提及 QIPC v0.0.1.dev895）；可复用技能（reusable simulator skills）封装可用本构模型、可运行示例场景、脚本与候选初值参数。
   - **静息态沉降（Rest-Shape Settling）：** 导入重力场沉降并写回 Blender 作为平衡静息态（如豆袋沙发的稳定 Neo-Hookean 沉降在 $D=1.75\text{ m}$ 下 2.26 s 后最大顶点偏移 $<10^{-4}D$，最小单元 Jacobian $>0.91$，体积变化 $<0.75\%$）。
   - **智能体引导的行为验证与固定序排障：** 智能体规划预设诊断性机器人操作（固定基座 Franka 7 自由度机械臂 + 2 指夹爪，或 70 mm 圆头压头），在离线仿真中运行并核查物理有效性（拉伸率、单元翻转 Jacobian）与任务验收标准。失败时严格按 **Motion（运动/工具/接触点）→ Geometry（几何与边界条件）→ Numerics（数值与接触参数）→ Material model（仅在前三者无法复现目标行为时修改本构模型）** 顺序修订。

### 3. 关键实验数据与参数核对

- **定量基准（Table 3，3 个 Replica 场景 + 3 个 ScanNet++ 场景 `67d702f2e8`、`7831862f02`、`acd69a1746`）：**
  - **Replica：** CoDimRecon 取得 CD **6.63 cm**（最低，HoloScene* 6.79 / GPT-6 Astra 11.79 / ReplicateAnyScene 41.88）、F1@5cm **52.24%**（低于 HoloScene* 53.48 与 GPT-6 Astra 52.91）、NC **79.76%**（最高）、PSNR 15.06 / SSIM 0.5695 / LPIPS 0.4817（均次于 HoloScene* 的 21.22 / 0.6881 / 0.4171）、V-JEPA 2.1 Layout **95.14** / Motion **96.90**、刚体稳定性 Stable(Ground) **100.00%** / Stable(All) **97.54%**（HoloScene* 仅 77.77% / 49.41%）。
  - **ScanNet++：** CoDimRecon 取得 CD **21.66 cm**（最低）、F1@5cm **38.28%**（最高）、NC **74.91%**（最高）、PSNR 16.68 / SSIM 0.6440 / LPIPS **0.3728**（LPIPS 最低）、Layout **96.30** / Motion **97.04**（均最高）、Stable(Ground) **97.05%** / Stable(All) **96.31%**（均最高或并列最高）。
  - 注：文中标注 `HoloScene*` 为关闭法向监督、改用本文 MaskClustering 掩码与 VGGT-Omega 位姿/深度的修改版官方配置；`ReplicateAnyScene` 因官方未开源位姿对齐与关系推理模块而由作者复现；`SimRecon` 因关键模块未公开仅列于 Tab. 1 能力对比。
- **可变形参数与演示（Table 2, Table 4/5, Appendix B.4）：**
  - **电话线（1D DER）：** $\rho=1600\text{ kg/m}^3, E=40\text{ MPa}, \nu=0.35, r=1.3\text{ mm}, \delta=0.15\text{ mm}, \mu=0.60$；原始 3200 节点重采样为 576 段无自穿透折线（长 1.664 m，最大偏差 0.422 mm）；两次重复抓取抬升峰值分别为 197.8 mm 和 197.7 mm，最小非邻接间隙 $22.0\text{ }\mu\text{m}$。
  - **纸张折纸（2D StVK 膜 + 塑性铰链）：** $\rho=761.9\text{ kg/m}^3$（面密度 $80\text{ g/m}^2$）, $E=2250\text{ MPa}, \nu=0.15, h=0.105\text{ mm}, \kappa_Y=25\text{ m}^{-1}, \delta=0.50\text{ mm}, \mu=0.40$。同一 $140^\circ$ 机械臂折叠轨迹对比：纯弹性峰值 $121.66^\circ$、松开 1 s 后回弹至 $4.48^\circ$（保持率 **3.7%**，0 个铰链屈服）；启用塑性铰链峰值 $124.24^\circ$、松开 1 s 后保持 $114.97^\circ$、结束时 $114.15^\circ$（保持率 **92.5%**，372 个铰链自然角变化 $>0.1\text{ rad}$）。
  - **塑料袋（2D 薄壳）：** $\rho=920\text{ kg/m}^3, E=60\text{ MPa}, \nu=0.45, h=0.03\text{ mm}, \mu=0.45$，与独立刚性垃圾桶分离建模，无固定顶点，仅靠接触抓取。
  - **椅垫（3D StVK–Hencky 实体）：** $\rho=100\text{ kg/m}^3, E=0.03\text{ MPa}, \nu=0.20, \delta=0.50\text{ mm}, \mu=0.40$；70 mm 圆头压头三点下压 25 mm，实测下陷 22.9 / 22.6 / 22.9 mm 并可恢复。
  - **豆袋沙发（3D Stable Neo-Hookean 实体）：** $\rho=120\text{ kg/m}^3, E=0.02\text{ MPa}, \nu=0.30, \delta=0.50\text{ mm}, \mu=0.40$。
- **消融实验（Table 5 / 文中称 Tab. 7）：**
  - ScanNet++ `67d702f2e8` 几何消融：Full (CD 7.24, F1 44.98, NC 75.93)；去几何参考 CD 升至 11.94（+64.9%），NC 降至 69.70（-8.2%）；去生成参考 F1 降至 39.12（-13.0%），NC 降至 68.14（-10.3%）；去渲染细化 RR 仅使 CD 微增至 7.36（+1.7%），但在全部 ScanNet++ 场景均值上，去 RR 使 PSNR 从 16.68 降至 15.82（-5.2%）、SSIM 从 0.6440 降至 0.6343（-1.5%）、LPIPS 从 0.3728 升至 0.3938（+5.6%）。
- **核心局限：**
  - 缺乏真实动力学测量校准，Tab. 2 中接受的物理参数仅为视觉与行为合理下的**有效仿真参数（effective simulation parameters）**，而非实测材料参数。
  - 几何消融仅在单个 ScanNet++ 场景完成；定量基准仅在 6 个场景（3 Replica + 3 ScanNet++）评估；可变形仿真目前仅做能力演示，尚无同类重建基线可横向定量对比。
  - 源码尚未公开发布，流程依赖闭源大模型骨干（文中基线提及共享 GPT-6 Astra 智能体骨干）及 VGGT-Omega / SAM3D / QIPC 等多工具链协同，端到端复现门槛较高。
