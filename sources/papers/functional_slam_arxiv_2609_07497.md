# Functional-SLAM（arXiv:2609.07497）

> 来源归档（ingest）

- **标题：** Functional-SLAM: Interaction-Aware Mapping with Online Functional Scene Graphs
- **短名：** Functional-SLAM
- **类型：** paper / slam / functional-scene-graph / open-vocabulary / interaction-aware-mapping
- **arXiv：** <https://arxiv.org/abs/2609.07497>
- **PDF：** <https://arxiv.org/pdf/2609.07497>
- **HTML：** <https://arxiv.org/html/2609.07497v1>
- **代码：** <https://github.com/Hbelief1998/Functional-SLAM-CoRL_2026> — [`sources/repos/functional_slam_corl_2026.md`](../repos/functional_slam_corl_2026.md)
- **数据集：** <https://huggingface.co/datasets/xg-123/Functional-SLAM-dataset>
- **Demo：** <https://www.bilibili.com/video/BV13xbw6NECB/>
- **会议：** CoRL 2026（已接收）
- **作者：** Xinggang Hu、Chenyangguang Zhang、Zihan Zhu、Ruida Zhang、Xiangkui Zhang、Xiangyang Ji
- **机构：** 清华大学；大连理工大学；苏黎世联邦理工学院（ETH Zurich）
- **版本：** arXiv:2609.07497v1（2026-09-07）
- **入库日期：** 2026-09-09
- **一句话说明：** 首个在 SLAM 过程中**在线递归维护**功能 3D 场景图（O/C/U 节点 + 功能边）的框架；MASt3R-SLAM 几何跟踪 + RAM++/DeepSeek/SAM3 开放词汇感知 + anchor-keyframe 节点稳定 + 时序关系后验 + 功能拓扑辅助回环。

## 项目页核查（步骤 2.5）

用户给出的 `https://cosmoh2g.github.io` **不是** Functional-SLAM 项目页（该域名为 **CosmoH2G** 手–夹爪迁移工作，Code 标注 Coming Soon）。Functional-SLAM **无独立 `*.github.io` 落地页**；截至入库日，公开入口为：

| 入口 | URL | 说明 |
|------|-----|------|
| GitHub README | <https://github.com/Hbelief1998/Functional-SLAM-CoRL_2026> | 安装、配置、运行、输出说明 |
| arXiv | <https://arxiv.org/abs/2609.07497> | 论文 |
| Hugging Face 数据集 | <https://huggingface.co/datasets/xg-123/Functional-SLAM-dataset> | 18×FunGraph3D + 18×SceneFun3D RGB 序列 |
| Bilibili | <https://www.bilibili.com/video/BV13xbw6NECB/> | Demo 视频 |

**开源结论：已开源**（推理/建图代码 + 评测数据 + 运行文档）。依赖 MASt3R、RAM++、SAM3（gated）等外部 checkpoint；DeepSeek API 需环境变量，非硬编码密钥。

## 摘要级要点

- **问题：** 几何/语义 SLAM 能建图与定位，但缺少**可操作交互单元**（把手、旋钮、按钮）及其**功能关系**（如「拎水壶把手才能倒水」）。既有功能 3D 场景图（OpenFunGraph、FunGraph、KeySG）多依赖**已知位姿或离线重建**，无法支撑机器人实时探索与交互。
- **表示：** 在线功能图 \(G_t=(U_t,O_t,R_t)\)：\(O\) 物体、\(U\) 机器人可操作交互单元、\(R\) 功能关系；沿用 OpenFunGraph 开放词汇定义。
- **三阶段管线：** (1) **跟踪 + 功能感知** — MASt3R-SLAM 估计位姿/点图/关键帧；RAM++ 标签 + DeepSeek 推理 + SAM3 分割产生帧级 O/U 观测与候选关系；(2) **在线功能图建图** — anchor-keyframe 局部几何同步位姿优化；角色分组的节点关联；时序关系后验提交稳定功能边；(3) **功能拓扑辅助回环** — 从在线图提取 graphlet 候选，补充纯视觉检索在重复外观/弱纹理场景的漏检。
- **定位（FunGraph3D）：** ATE RMSE 相对最强基线 MASt3R-SLAM **降低 16.3%**（论文）；消融完整系统 ATE **16.5 mm**（FunGraph3D）。
- **功能图（Ours pose，FunGraph3D）：** Overall Nodes R@3 **64.55%**、Triplets R@5 **41.10%**，显著高于同设定下的 OpenFunGraph / FunGraph / KeySG。
- **效率：** **0.38 FPS** vs 离线功能图方法 0.02–0.06 FPS（Table 4）；关键帧模式可进一步加速。
- **硬件：** 论文实验单卡 **NVIDIA RTX 3090**；官方 README 开发环境 Ubuntu + Python 3.11 + PyTorch 2.5.1 + CUDA 12.4。

## 核心摘录（面向 wiki 编译）

### 架构

1. **帧级观测：** 检测 \(\ell,s,b,m\)，反投影得 3D support \(P\)；关系证据 \(\eta_{ou}^t\) 含功能结构门控与 mask 重叠。
2. **节点关联：** anchor keyframe 维护 canonical 几何；匹配分 \(S_{ij}=\phi_{geo}+\phi_{sem}+\phi_{ctx}\)，匈牙利一对一。
3. **时序边提交：** 多帧 support \(r_t\) 累积后才 commit；每个交互单元只保留最可靠对象归属。
4. **回环：** 对象拓扑 \(S_G\) 与功能拓扑 \(S_{FT}\) 与 MASt3R 视觉候选合并，几何验证后进入位姿图优化。

### 数字读法

| 设定 | Functional-SLAM | 对照 |
|------|-----------------|------|
| FunGraph3D ATE vs MASt3R-SLAM | **−16.3%** RMSE | 功能拓扑回环补视觉漏检 |
| SceneFun3D 平均 ATE | **33.25 mm** | 与基线可比（纹理丰富、回环少） |
| FunGraph3D Nodes R@3（Ours pose） | **64.55** | OpenFunGraph 34.04；FunGraph 39.44 |
| FunGraph3D Triplets R@5（Ours pose） | **41.10** | OpenFunGraph 11.64；FunGraph 27.05 |
| 运行时 FPS | **0.38** | OpenFunGraph 0.02；FunGraph 0.06 |
| 消融 ATE（FunGraph3D） | w/o 节点稳定 19.0 → w/o 时序边 17.9 → w/o 功能回环 19.7 → **Ours 16.5** | 三模块均贡献定位或建图 |

### 开源核查（步骤 2.5）

见 [`sources/repos/functional_slam_corl_2026.md`](../repos/functional_slam_corl_2026.md)：**已开源、可运行**（`main.py` + 配置 + HF 数据 + checkpoint 下载说明）。License：**CC BY-NC-SA 4.0**。SAM3 需 HF gated 权限；DeepSeek 走 `DEEPSEEK_API_KEY` 环境变量。

## 对 wiki 的映射

- 升格 [Functional-SLAM 论文实体](../../wiki/entities/paper-functional-slam.md)
- 交叉：[vS-Graphs](../../wiki/entities/paper-vs-graphs-visual-slam-scene-graph.md)、[SLAMFormer-∞](../../wiki/entities/paper-slamformer-infinity.md)、[导航·SLAM 栈](../../wiki/overview/navigation-slam-autonomy-stack.md)、[State Estimation](../../wiki/concepts/state-estimation.md)、[FunRec 策展索引](../../wiki/entities/paper-sa-2604-05621-funrec-reconstructing-functional-3d-scenes-from.md)

## 当前提炼状态

- [x] 方法 + 仿真数字 + 开源入口
- [x] wiki 实体、时序图与交叉引用
- [x] `sources/repos/`
