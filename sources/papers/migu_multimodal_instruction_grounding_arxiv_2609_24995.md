# MIGU: Multimodal Instruction Grounding under Uncertainty（arXiv:2609.24995）

- **来源：** <https://arxiv.org/abs/2609.24995>（arXiv:2609.24995v1，2026-09-21）
- **作者与机构：** Mingke Lu\*（NUS, UCLA）、Anxing Xiao\*（NUS）、David Hsu（NUS）
- **项目主页：** <https://multimodal-instruction.github.io/>
- **代码仓库：** <https://github.com/MingkeLu/MIGU>
- **开源状态：** 部分开源（核心多模态 grounding 管线、100 帧真实 RGB-D 基准数据集与 ROS Noetic 在线接口已发布；单轮 POMDP 澄清对话、ASR 与下游 Fetch/Autolife S2 规划器未包含在仓库中）
- **通读：** 已完成
- **核对范围：** 已通读论文原文 `papers/2609.24995.txt` 全文（L1–1125，§I–VII 及参考文献 [1]–[51]），重点核对 §III 问题定义、§IV 几何/语义不确定性建模与单轮 POMDP 澄清策略、§V 100 场景基准（Table I、Fig. 7）、Autolife S2 真机（Table II）与 Fetch TAMP 演示，以及 §VI 局限性。

---

## 核心问题与动机

在模糊的人机交互（HRI）场景中，人类常使用带指示代词的多模态指代指令（如“把**这个**放到**那里**”+ 指向手势）。纯语言指令在存在多个同类物体或未知放置区时存在语义歧义，而 3D 指向手势受距离、视角噪声、传感器深度误差以及人体关键点估计抖动影响，越远越容易发生角度与深度偏移；同时将低质量分割或 VLM（视觉语言模型）误检引入规划器也会引发错误执行。现有研究多将语言与手势视为确定性输出或仅输出 Top-1 候选，缺少面向下游任务规划（TAMP）的概率校准信念分布（belief distribution）。

---

## 方法要点（§III–§IV）

### 1. 3D 几何不确定性建模（§IV-B）
- 从 RGB-D 与人体姿态估计提取视线方向 $u_i$、手部食指位置 $F$（关键点 5–8）、眼睛位置 $E$ 与像素 $x_i$ 对应 3D 深度 $d_i$，构成名义指向向量 $\bar{y}_i = x_i - F$。
- 考虑两类传感误差：视角单位向量在切平面 $T_{u_i}\mathbb{S}^2$ 上的零均值高斯噪声 $\delta_u \sim \mathcal{N}(0, \sigma_u^2 P_i)$，以及与深度线性相关的深度噪声 $\sigma_{d,i} = \alpha_d d_i$（论文 §IV-B 引用的文献先验分别为 $\sigma_u = 0.044\text{ rad}$ [35] 与 $\alpha_d = 0.04$ [36]）。
- 通过向量映射 $y = E + d u - F$ 的雅可比 $J_i$ 将像素位置协方差传播到指针-目标向量空间，并投影至方向切平面 $T_{\mu_i}\mathbb{S}^2$，再叠加手部关键点拟合方向协方差 $B_i^\top \Sigma_{\mathrm{fit}} B_i$，得到总方向协方差 $C_i = C_i^{\mathrm{geo}} + B_i^\top \Sigma_{\mathrm{fit}} B_i$。
- 利用球面 $\mathbb{S}^2$ 对数映射（log-map）残差在高斯近似下评估几何似然 $p(g^{\mathrm{obs}} \mid x_i)$，从而保留距离越近角度容忍越宽、距离越远波束越窄的物理结构。

### 2. 语义似然与逐像素贝叶斯式融合（§IV-A, §IV-C）
- VLM 输出至多 $k$ 个候选目标边界框及其未校准相对置信度 $c_h$；SAM 根据边界框生成像素级实例掩码 $X_h$，构建空间语义分布 $s_{\mathrm{sem}}(x_i, h) \propto c_h \mathbb{I}[x_i \in X_h]$。
- 逐像素融合未归一化得分 $q_h(x_i) \propto p(g^{\mathrm{obs}} \mid x_i) \cdot c_h \mathbb{I}[x_i \in X_h]$；对每个掩码区域 $X_h$ 采用空间最大池化（max pooling）聚合后归一化，得到候选假设的概率信念 $b(h)$。

### 3. 单轮 POMDP 行为规划（§IV-D）
- 对候选信念排序并截断至最小 90% 累积概率集合再归一化。
- 从固定预定义视觉/空间二元属性词表 $\mathcal{C}$ 中选择最优澄清问题 $q^*$，比较直接执行期望收益 $V_{\mathrm{exec}}(b) = b(h^*) R_{\mathrm{success}}$ 与提问后条件信念收益 $V_{\mathrm{query}}(b, q) = -C_{\mathrm{query}}(q) + \sum_y p(y \mid b, q) V_{\mathrm{exec}}(b_{q,y})$；当且仅当 $V_{\mathrm{query}} > V_{\mathrm{exec}}$ 时触发**单轮二元属性澄清**，否则直接将 $b(h)$ 交给下游规划器。

---

## 实验与关键结果（§V）

### 1. 100 场景真实 RGB-D 基准（Table I & Fig. 7）
- 覆盖 0.2–0.5 m 近距桌面与 0.5–3 m 中距场景（含同类别多干扰物），共 100 帧真实采集样本。
- **Table I 消融实验（Acc. ↑ / Top-2 ↑ / Scaled 2D Dev ↓ / NLL ↓ / Brier ↓）：**
  - `MIGU (full)`：**83% / 95% / 0.0592 / 1.17 / 0.25**
  - `w/o semantics`：51% / 76% / 0.0869 / 1.37 / 0.49
  - `w/o geometry`：46% / 64% / 0.1746 / 1.76 / 0.57
  - `Geo. -> ray-distance`：76% / 89% / 0.0805 / 2.91 / 0.40
  - `Geo. -> ray-angle`：58% / 83% / 0.1441 / 1.63 / 0.51
  - `Geo. -> distance-threshold`：74% / 90% / 0.0851 / 2.88 / 0.36
  - `Sem. -> VLM+detector`：60% / 76% / 0.0984 / 5.08 / 0.47
  - `Sem. -> VLM-SoM`：77% / 85% / 0.0907 / 3.49 / 0.41
  - `Sem. -> VLM-Grid`：40% / 45% / 0.1287 / 17.66 / 1.02
  - `Sem. -> VLM-PIVOT`：42% / 49% / 0.1237 / 14.67 / 0.93
- **Fig. 7 基线对比（正文报告）：** MIGU 取得最低 NLL 与 Brier 分数，相比 `LEGS-POMDP` 将 NLL 降低 **22.5%**、Brier 降低 **60.3%**，并全面优于 Geometric、Touch-Line Transformer、GIRAF、VLM-SoM、VLM-Grid 与 VLM-PIVOT。

### 2. 真机系统集成验证（Table II & Fetch TAMP）
- **Autolife S2 轮式人形机器人（Table II，8 个任务 × 3 次试验 = 24 次）：** 集成 Whisper ASR + MIGU + cuRoboV2 / ContactGraspNet / VAMP 运动规划。成功率：`Semantic Only` **45.8%**、`Geometry Only` **41.7%**、`MIGU Without Clarification` **83.3%**、`MIGU With Clarification` **91.7%**（+8.4 pp）。
- **Fetch 移动操作平台（定性演示）：** 集成 PDDLStream + OWLv2 + SAM，演示带障碍移开的取水瓶任务与基于主动二元澄清的苹果抓取放置任务。

---

## 原文声明的局限性（§VI）
1. 语义先验直接依赖 VLM 输出的未校准置信度分数作为相对加权；若 VLM 对错误目标给出过高分数或漏检真值，融合仍受上游偏差影响。
2. 行为规划仅实现单轮二元属性澄清，尚未扩展到多轮对话或将语义/几何信念在下游 TAMP 长程规划步间动态传播。
