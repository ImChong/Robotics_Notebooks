# MeRoPE: Metric Rotary Position Embedding for Camera-Controlled Video Generation（arXiv:2609.01252）

> 来源归档（ingest）

- **标题：** MeRoPE: Metric Rotary Position Embedding for Camera-Controlled Video Generation
- **类型：** paper / camera-controlled video generation / geometry-aware positional encoding
- **arXiv abs：** <https://arxiv.org/abs/2609.01252>
- **PDF：** <https://arxiv.org/pdf/2609.01252>
- **项目页：** <https://qiaozhijian.github.io/merope/>
- **发表日期：** 2026-09（arXiv v1）
- **机构：** 香港科技大学（HKUST）；卓驭科技（Zhuoyu Technology）
- **通讯作者：** Shaojie Shen（eeshaojie@ust.hk）
- **入库日期：** 2026-09-06
- **代码：** 论文与项目页宣称将公开；**截至入库日项目页未列 GitHub / Hugging Face 链接**（见 [`sources/sites/merope.md`](../sites/merope.md)）
- **一句话说明：** **MeRoPE** 针对相机可控视频生成中 **齐次射影相机位置编码** 在大基线 metric 轨迹上的 **尺度失控**（attention logit 与特征范数随物理平移无界增长），提出 **范数保持** 的相对相机编码：正交块编码射线朝向、多频 RoPE 编码 query 相机系 metric 位移，并以 **视差锚定球面旋转** 沿极线弧给出对应先验；在 **nuScenes** 大基线与 **PanShot** 多样光学上相对 UCPE / PRoPE / GTA 等取得最佳 **旋转–平移一致性**。

## 摘要级要点

- **问题定义：** 几何感知位置编码（PE）让 attention 对相机外参与 per-token 视线射线敏感，但现有 **齐次射影相对位姿**（GTA、PRoPE、UCPE）把 metric 平移写成 **未归一化内积**，随 $\|\mathbf{o}_j-\mathbf{o}_i\|$ 线性放大，大基线时压制视觉内容并破坏轨迹控制。
- **三元困境（Theorem 1）：** 深度无关的 **全 metric 相对位姿**（A）、**严格 per-token 分解**（B）、**范数保持 attention**（C）三者不可同时满足；相机控制需要 **A+C**，MeRoPE 通过 **query-camera 分组** 放松 B。
- **MeRoPE 四块结构：** $\mathcal{U}_{ab}=\mathcal{U}^{\mathrm{disp}}\oplus\mathcal{U}^{\mathrm{rot}}\oplus\mathcal{U}^{\mathrm{trans}}\oplus\mathcal{U}^{\mathrm{native}}$，各块为正交旋转，$\mathcal{U}_{ab}^\top\mathcal{U}_{ab}=I$。
  - $\mathcal{U}^{\mathrm{rot}}$：UCPE 式 **MinRot 射线局部坐标系** 间相对旋转；
  - $\mathcal{U}^{\mathrm{trans}}$：query 相机系位移 $\Delta\mathbf{o}_{b\mid a}=C_a^\top(\mathbf{o}_b-\mathbf{o}_a)$ 各坐标 **多频 2D RoPE**（改相位不改范数）；
  - $\mathcal{U}^{\mathrm{disp}}$：沿 **极线球面弧** 采样视差锚点，其相对 query 射线帧的旋转提供 **静态场景对应** 先验（无需 VGGT / Depth Anything 等 3D 重建前处理）；
  - $\mathcal{U}^{\mathrm{native}}$：保留骨干 **时间 + 图像坐标 RoPE**。
- **实验骨干：** **Wan2.2 TI2V-5B**（nuScenes 相机控制）；**Wan2.1 T2V-1.3B**（PanShot 跨光学泛化）；另展示 **History SA** 检索图像的 **Real-to-Sim** 长 rollout（跨天气/跨遍历）。
- **主要结果：** nuScenes 与 PanShot 上 **生成相机运动与条件位姿一致性**（旋转与平移联合）优于 UCPE 等 prior；消融验证各几何块贡献；attention 可视化显示 UCPE 在大基线下向最远帧异常分配质量，MeRoPE 保持局部时序 profile。
- **开源承诺：** 摘要写 "Code will be made publicly available"；项目页截至入库日 **无代码链接**。
- **局限：** 仍依赖预训练视频扩散骨干与校准射线输入；不做显式 3D 重建管线；query-camera 分组牺牲严格 per-token 分解；机器人操纵 WM 需另接动作条件而非仅相机轨迹。

## 核心论文摘录

### 1) 齐次射影编码的 metric-scale 失效

- **链接：** §3.1；Eq. (2)–(4)；Fig. 1
- **摘录要点：** $\mathcal{U}^{\mathrm{cam}}_{ij}$ 的平移项 $h_k(\mathbf{u}^q)^\top\Delta\mathbf{o}_{j\mid i}$ 随基线线性增长；非正交块亦使 value 特征范数无界放大；UCPE 训练模型在直线前进轨迹上，末帧中心 query 对最早 key 帧 attention 质量随基线增大。
- **对 wiki 的映射：**
  - [paper-merope.md](../../wiki/entities/paper-merope.md) — 问题动机与与 UCPE/PRoPE 对比表。

### 2) 正交 metric 位姿 + 视差锚定对应

- **链接：** §3.2–3.3；Fig. 2–3
- **摘录要点：** 保留 $R_{ab}$ 正交旋转；用 RoPE 替代齐次平移块；沿极线弧采样 anchor frame 旋转构成 $\mathcal{U}^{\mathrm{disp}}$，给跨视角静态对应提供有界先验。
- **对 wiki 的映射：**
  - [paper-merope.md](../../wiki/entities/paper-merope.md) — 核心机制与流程总览。

### 3) nuScenes / PanShot 评测与 Real-to-Sim

- **链接：** §4–5；项目页 demo
- **摘录要点：** nuScenes 上命令/恢复相机路径与生成帧一致；PanShot 覆盖 118°–177° FoV、室内/户外/低光；检索历史帧 + History SA 在跨天气静态布局保持上优于 History CA。
- **对 wiki 的映射：**
  - [Generative World Models](../../wiki/methods/generative-world-models.md) — 相机几何 PE 路线实例。

## 对 wiki 的映射

- 主实体页：[paper-merope.md](../../wiki/entities/paper-merope.md)
- 互链：[Generative World Models](../../wiki/methods/generative-world-models.md)、[Video-as-Simulation](../../wiki/concepts/video-as-simulation.md)、[Wan 视频基础模型](../../wiki/entities/paper-wan-video.md)、[PanoWorld](../../wiki/entities/paper-panoworld-real-world-panoramic-generation.md)（同为射线/PRoPE 几何条件视频生成）

## 参考来源（原始）

- arXiv:2609.01252
- 项目页：<https://qiaozhijian.github.io/merope/>
- 相关 prior：CaPE、GTA、PRoPE、UCPE、RayNova、URoPE、RayRoPE、CameraCtrl、ReCamMaster、CamCo、Wan2.1/2.2
