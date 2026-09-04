# CrossTracer（arXiv:2608.06688）

> 来源归档（ingest）

- **标题：** CrossTracer: Cross-Embodiment Navigation via VLA Model Reasoning and Trace Residuals Adapting
- **类型：** paper / VLA / navigation / VLN / cross-embodiment / pixel-space trace
- **arXiv：** <https://arxiv.org/abs/2608.06688>
- **PDF：** <https://arxiv.org/pdf/2608.06688>
- **HTML：** <https://arxiv.org/html/2608.06688>
- **项目页：** <https://lilduckkk.github.io/CrossTracer-Nav/>
- **作者：** Yao Wang（鹏城实验室 / 南方科技大学）、Siyuan Wang（南方科技大学）、Zhirui Sun（创新投资研究院）、Wenzheng Chi（苏州大学）、Liang Lin（鹏城实验室）、Jiankun Wang（南方科技大学，通讯）、Wenjun Xu（鹏城实验室，通讯）
- **机构：** 鹏城实验室（Peng Cheng Laboratory）；南方科技大学（SUSTech）；创新投资研究院（Innovation Investment Research Institute）；苏州大学（Soochow University）
- **入库日期：** 2026-09-04
- **一句话说明：** 用归一化像素轨迹当语义提案与本体可通行性之间的接口：VL-Tracer（OmniVLA + LoRA）出无本体语义轨迹，CE-Adapter 用残差改到轮式/腿式可走路径；CE-RRT* 从全景分割自动造监督。NaviTrace 总分 45.68（相对 Gemini-2.5-Pro +28.1%）；真机相对 OmniVLA 抬 SR。项目页截至入库日无代码仓。

## 摘要级要点

- **问题：** 导航 VLA 有开放词汇语义先验，但不编码本体可通行性。同一指令与同一 egocentric 图，腿式可上台阶/粗糙地面，轮式必须绕行。端到端 VLA 把语义、本体、控制缠在同一策略里，换机就要重训。
- **接口：** 导航计划写成 \(N=8\) 个归一化图像平面航点 \(T_e=\{\mathbf{w}_t\}\subset[-1,1]^2\)，与 NaviTrace 的 pixel-space trace 对齐；不是低层速度，也不是完整 3D 轨迹。
- **两段：** **VL-Tracer** 基于 OmniVLA，从 RGB + 语言 / 像素目标 / 二者预测无本体初始轨迹 \(T_{init}\)；**CE-Adapter** 只在第二段看机器人 ID，用 FiLM + 轨迹–视觉交叉注意力预测残差 \(\Delta T_e\)，\(T_e=T_{init}+\Delta T_e\)。
- **监督：** **CE-RRT\*** 用 Mask2Former 全景分割 → 按本体写代价图 → 图像平面 RRT*，给 CE-Adapter 自动造 \(T_e^*\)。推理时不要分割、不要代价图。
- **NaviTrace：** 总分 **45.68** vs Gemini-2.5-Pro **35.67**（+10.01 / +28.1%），vs Robobrain-2.5-8B **27.96**。去掉 CE-Adapter 掉到 **22.56**（−23.12）。加像素目标后 **63.91**（输入不同，不能直接比语言-only）。
- **真机：** 轮式 / 腿式共用 Jetson Orin 采集 + 4090 工作站推理。相对 OmniVLA：轮式平均 SR **0.40→0.65**，腿式 **0.45→0.70**（四任务 × 五次）。
- **开源（截至 2026-09-04）：** 对照表把 CrossTracer-8B 标成 Open-Source ✓，但项目页 **未列 GitHub / Hugging Face**；作者公开仓无 CrossTracer → **宣称开源 / 待核实**。勿写「已开源」。

## 核心摘录（面向 wiki 编译）

### 相对候选重排序：连续残差，不是从有限提案里挑

VAMOS 用 VLM 出一组图像空间候选，再用本体 affordance 打分重排。CrossTracer 把像素轨迹当成 **连续 refinement 目标**：提案全错局部段时，残差仍可把航点挪出高代价区，而不是在全错集合里选「最不坏的一条」。

### VL-Tracer 训练设定

- 骨干：OmniVLA（视觉编码 + Llama 7B 级语言推理）；输出头从低层动作改成 \(N\) 个 \(\tanh\) 约束的 2D 航点。
- 数据：VAMOS 图像空间路径标注；LoRA + 轨迹头，骨干冻结。
- 目标：MSE + \(\lambda_{smooth}=0.01\) 一阶平滑；训练时语言/位姿以 \(p_{drop}=0.3\) 随机丢。
- 算力：8×A100，约 48 小时。训完冻结，梯度不回传到 proposer。

### CE-Adapter 与 CE-RRT*

- 视觉：预训练 ResNet；机器人 embedding 经各层 FiLM（\(\gamma\approx 1,\beta\approx 0\) 近恒等初始化）。
- 轨迹 query 对 embodiment-conditioned visual tokens 做交叉注意力；残差头 \(\Delta T_e=\delta_{max}\tanh(\cdot)\) 限制偏离语义提案。
- 辅助头：可通行代价图重建；Sensitivity Head 出 \(\alpha_e\) 加权物理代价损失。
- 损失：\(\mathcal{L}_{trace}+\lambda_{trav}\mathcal{L}_{trav}+\lambda_{cost}\mathcal{L}_{cost}+\lambda_{smooth}\mathcal{L}_{smooth}\)，权重 \((1,1,1,0.05)\)。
- 数据：62k 导航图，CE-RRT* 按本体标 \(T_e^*\) 与 \(\mathcal{C}_e\)；单卡 4090 约 3 小时。输入与代价图缩到 \(64\times 64\)。
- 规划器：Mask2Former-R50；goal-bias 0.15、步长 25 px、半径 60 px、最多 10k 次迭代。

### NaviTrace 分项读法（语言 + 本体，无额外像素目标）

| 轴 | CrossTracer | w/o CE-Adapter | 读法 |
|----|-------------|----------------|------|
| Total | 45.68 | 22.56 | 残差段贡献约一半分数 |
| Bicycle / Human / Legged / Wheeled | 42.16 / 46.26 / 46.40 / 46.28 | 23.94 / 22.64 / 22.47 / 21.41 | 四本体差距小，不是只拟合一种机 |
| Accessibility | 33.79 | −3.18 | 物理接地类涨幅最大 |
| Social norms | 37.87 | 1.28 | 同上 |
| Stationary obstacle | 46.11 | 25.23 | 局部绕障靠残差，不只平滑 |
| Dynamic obstacle / Visibility | 52.93 / 52.49 | 31.68 / 32.27 | 场景类也稳定高于基线 |

### 真机协议边界

四指令：柜台后白桌、柜台后米色沙发、二层平台「Caution Wet Floor」牌、户外左转后黑垃圾桶。每任务每机每方法 5 次。成功=到目标距离阈值内、无碰撞、无人工干预。SPL / STT 的参考来自人类专家路径/时间。低层控制器跨方法固定。WiFi 远程推理，不是机上闭环 4090。

### 开源核查（步骤 2.5，2026-09-04）

见 [项目页归档](../sites/crosstracer-nav-github-io.md)。**结论：宣称开源 / 待核实。** 勿建 `sources/repos/`。

## 对 wiki 的映射

- 沉淀实体页：[CrossTracer](../../wiki/entities/paper-crosstracer.md)
- 交叉补强：[VLA](../../wiki/methods/vla.md)、[视觉–语言导航](../../wiki/tasks/vision-language-navigation.md)、[VLN 四范式开源复现](../../wiki/overview/vln-open-source-repro-paradigms.md)、[跨具身迁移知识链](../../wiki/overview/hub-cross-embodiment.md)、[Green for Go](../../wiki/entities/paper-green-for-go-vla-nav-grounding.md)、[HumanoidVLN](../../wiki/entities/paper-humanoidvln.md)、[NaVILA](../../wiki/entities/paper-notebook-navila-legged-robot-vision-language-action-model.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2608.06688>
- 项目页核查：[crosstracer-nav-github-io.md](../sites/crosstracer-nav-github-io.md)
