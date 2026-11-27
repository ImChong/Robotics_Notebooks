# Hierarchical Floorplan-Guided Vision-Language Exploration for Embodied Question Answering

- **作者：** Albert Gassol Puigjaner, Kostas Alexis
- **机构：** Autonomous Robots Lab, Norwegian University of Science and Technology (NTNU)
- **版本 / 载体：** arXiv:2609.26360v1 [cs.RO]（2026-09-22 提交），arXiv HTML 全文
- **原文：** https://arxiv.org/abs/2609.26360 （正文阅读：https://arxiv.org/html/2609.26360）
- **项目页 / 代码：** 已开源 — 主规划与评测仓库 <https://github.com/ntnu-arl/hflex_eqa>（BSD-3-Clause），配套 ROS 2 接口与仿真编排仓库 <https://github.com/ntnu-arl/hflex_eqa_ros>（BSD-3-Clause），项目主页 <https://ntnu-arl.github.io/hflexeqa/>，Zenodo 归档 DOI `10.5281/zenodo.22919691`
- **核对范围：** 通读 arXiv HTML 全文 Section I–VI（Abstract、Introduction、Related Work、Problem Statement、Method IV-A 至 IV-D 含公式 (1)–(14)、Experiments V-A 至 V-C 含 Table I–IV 与 Fig. 1–5 图注、Conclusions），并核对 `ntnu-arl/hflex_eqa` 与 `ntnu-arl/hflex_eqa_ros` 官方仓库 README 与文件树。
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [Hierarchical Floorplan-Guided Vision-Language Exploration for Embodied Question Answering](../../wiki/entities/paper-agentic-2609-26360.md)
- **核心问题与动机：** 现有基于 VLM 的主动具身问答（EQA）方法（如 ExploreEQA、GraphEQA）通常仅基于局部观测与单一规划策略探索环境，未利用建筑平面图等全局结构先验来推理“尚未观测到的相关房间类型”，也未区分“探索已知房间”“精细检查已见物体”“寻找未发现房间”三类不同的信息缺失模式。
- **核心架构（HFLEX-EQA）：**
  1. **在线分层场景图（Section IV-A）：** 基于 Hydra 增量构建含房间 $L_r$、语义边界前沿 $L_f$、导航图 $L_n$、物体 $L_o$ 与度量-语义网格 $L_m$ 五层节点 $\mathcal{N}_t$；结合 YOLOe 目标检测与 LSeg 像素级 CLIP 特征投影至 TSDF 形成开放词汇二维占据栅格 $\mathcal{O}_t$，由广义 Voronoi 图（GVD）提取导航图，波前边界检测（WFD）提取语义前沿。
  2. **弱拓扑平面图先验（Section IV-B）：** $\mathcal{G}^{\mathcal{F}} = (\mathcal{N}^{\mathcal{F}}, \mathcal{E}^{\mathcal{F}})$ 仅保留带语义标签的房间节点与连通边，不保留度量几何或机器人在平面图内的精确全局定位，允许缺失节点/边或错误标签。
  3. **高层 VLM 规划器（Section IV-C）：** 输入 $\mathcal{X}_t = (q, \mathcal{A}, \mathcal{G}_t, \mathcal{I}_t, \mathcal{H}_t, \mathcal{G}^{\mathcal{F}})$；通过 CLIP 相似度检索每房间最多 $M=3$ 张关键帧构建紧凑视觉记忆 $\mathcal{I}_t$，并约束房间标签预测落入平面图词表；输出 $(m_t, g_t, a_t, \kappa_t)$，当置信度 $\kappa_t \ge \kappa_{\min}=0.8$ 时直接回答，否则在 `explore_room`、`go_to_objects`、`find_room` 三种探索模式中选择。
  4. **底层专用规划器（Section IV-D）：**
     - `explore_room`：任务解析 VLM 预测 $N=6$ 个相关物体，按式 (7) 计算目标房间内前沿最大余弦相似度。
     - `go_to_objects`：围绕目标物体包围盒以最小视距 $r_{\min}$ 采样无碰撞视点，按式 (8)（$\lambda_{\text{vis}}=0.7, \lambda_{\text{img}}=0.3$）最大化遮挡后可见顶点比例与图像投影覆盖率。
     - `find_room`：将当前房间标签匹配至平面图内所有同类节点作为定位候选，按式 (9)–(11) 计算邻居房间类别到目标房间类别 $c^\star$ 的最短图距离进度权重 $w(c)$，并结合入口模板匹配分 $S_{\text{tr}}$、平面图进度引导分 $S_{\text{fp}}$ 与归一化前沿尺度 $S_{\text{free}}$（式 (12)–(14)，$(\alpha, \beta, \gamma) = (0.35, 0.55, 0.1)$）选择最佳前沿。
- **关键实验数据核对：**
  - **仿真评测（Table I）：** 在 HM3D 单层带语义标注子集上评测 OpenEQA（108 episodes）与 ExploreEQA（114 episodes），最大规划步数 50，评测 GPT-4o、GPT-5.5、Gemini-3.5 Pro、Claude Opus 4.8、Qwen 3.5-9B；有选项设置下 Gemini-3.5 Pro 在 OpenEQA 达 75.0% SR（ExploreEQA 63.2%），GPT-5.5 在 OpenEQA 同达 75.0% SR（ExploreEQA 57.9%），显著高于 GraphEQA 与 ExploreEQA；无选项自由回答设置下 Gemini-3.5 Pro 在 OpenEQA 达 56.5% SR、Claude Opus 4.8 在 ExploreEQA 达 57.9% SR。
  - **消融实验（Table II）与真值语义对比（Table III）：** 去掉平面图先验（w/o FP）使 Gemini-3.5 SR 在 OpenEQA 从 75.0% 降至 64.8%、ExploreEQA 从 63.2% 降至 53.5%；去掉视点选择（w/o VP）或语义前沿（w/o SF）均明显降分；直接向 GraphEQA 提示词注入平面图仅在 ExploreEQA 略升而在 OpenEQA 下降（46.3% → 43.5%）。使用真值语义（+GT）可进一步提升至 OpenEQA 79.6%、ExploreEQA 64.0%。
  - **实机验证（Section V-C, Table IV, Fig. 5）：** 部署于 ANYmal 四足机器人（Ouster OS0 LiDAR、VectorNav VN100 IMU、Intel RealSense D455、NVIDIA Jetson Thor），所有感知建图与规划机载实时运行，VLM（GPT-5.5）经外部 API 调用。在 7 类真实室内问题上全部答对（置信度 $\kappa \in [0.84, 0.95]$，步数 $S \in [4, 11]$，路径 $P \in [14.6, 50.9]\text{ m}$）；无选项回放中 47/50 步高阶模式与有选项一致；高斯模糊/光照衰减/Gamma 扰动回放中模式一致性保持稳定，仅强模糊下答案准确率下降。
