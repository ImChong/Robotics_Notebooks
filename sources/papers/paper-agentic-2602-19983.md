# Contextual Safety Reasoning and Grounding for Open-World Robots

- **作者：** Zachary Ravichandran, David Snyder, Alexander Robey, Hamed Hassani, Vijay Kumar, George J. Pappas
- **机构：** University of Pennsylvania, Carnegie Mellon University
- **版本 / 载体：** arXiv:2602.19983v2 [cs.RO]（2026-02-23 提交 v1，2026-02-24 修订 v2）；项目页抬头另标注 ICRA 2025，BibTeX 标为 arXiv 2602.19983（2026）
- **原文：** <https://arxiv.org/html/2602.19983> ；<https://arxiv.org/abs/2602.19983> ；<https://zacravichandran.github.io/CORE>
- **项目页 / 代码：** 项目页 <https://zacravichandran.github.io/CORE> 可访问；官方 GitHub 仓库 <https://github.com/KumarRobotics/CORE> 当前仅包含占位 `README.md`（标注 `Coming soon.`），无实际代码文件（开源状态：宣称将开源，截至 2026-11-27 未确认可运行官方源码实现）
- **核对范围：** 通读 arXiv HTML 全文（§ I Introduction 至 § VII Conclusion、参考文献及附录 § A-A–§ B-D3、表 I–IV、表 A.I–A.II、系统提示词 Listing B-D3），并核查项目页文本与 `KumarRobotics/CORE` 仓库文件清单快照
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[Contextual Safety Reasoning and Grounding for Open-World Robots](../../wiki/entities/paper-agentic-2602-19983.md)

### 1. 问题设定与动机
- 针对控制仿射（control-affine）动力学移动机器人 $\dot{x} = f(x) + g(x)u$，配备 RGB-D 相机与里程计，给定名义控制输入 $u_{\text{nom}}$ 与高层安全准则 $\mathcal{G}$（如遵循行人社交规范），在无先验地图、无预定义环境特定安全规范的未知开放世界运行。
- 目标：在线从流式视觉观测 $\{I_t\}$ 推理上下文安全约束 $\{s_1, \dots, s_K\}$，落地为空间安全集合 $\mathcal{S} = \bigcap_k \mathcal{S}_k \subset \mathbb{R}^n$，并在感知不确定性下以至少 $1-\delta$ 概率保持安全同时最小化 $\|u_{\text{safe}} - u_{\text{nom}}\|$。

### 2. CORE 三阶段架构（§ IV）
1. **上下文安全推理（Contextual Safety Reasoning，§ IV-A）**：
   - 将 VLM 输出结构化为空间谓词集合 $\mathcal{P}_{\text{safe}}$ 与 $\mathcal{P}_{\text{unsafe}}$，结合 4 个空间算子：`ON(class)`（可通行表面）、`NEAR(class)`（实体碰撞缓冲）、`AROUND(class)`（危险指示物/社交禁区周边 50 像素膨胀）、`BETWEEN(class)`（多实例排列围成的禁止区域凸包）。
   - 系统提示词要求先输出 `safety_logic` 思维链（CoT）推理，再输出扁平 JSON 字符串字段 `classes`、`unsafe_regions`、`safe_regions`。
2. **语义落地（Semantic Grounding，§ IV-B）**：
   - 使用开放词汇分割模型 SAM3 提取各语义类别像素掩码，按算子计算安全像素集 $\mathcal{I}_{\text{safe}} = (\bigcup_j \mathcal{I}_j^{\text{safe}}) \setminus (\bigcup_k \mathcal{I}_k^{\text{unsafe}})$。
   - 结合配准深度与针孔相机模型投影至 3D 点云，累计到分辨率 0.2 m 的二维网格 $\mathcal{C}_{ij}$；统计安全/不安全点数得到 $\mathbb{P}[x\text{ is safe}] = n_{ij}^{\text{safe}} / (n_{ij}^{\text{safe}} + n_{ij}^{\text{unsafe}})$，按阈值 $\tau = 0.5$ 构造安全集 $\mathcal{S}$，再通过符号距离场（SDF）构造障碍函数 $h(x)$ 与解析梯度 $\nabla \hat{h}(x)$。
3. **上下文安全执行与感知不确定性保障（§ IV-C、§ IV-D）**：
   - 求解二次规划（QP）安全滤波器：$\min_u \|u - u_{\text{nom}}\|_2^2$，满足 $\langle \nabla h(x), f(x)+g(x)u \rangle \ge -\alpha(h(x))$（实验中 $\alpha(x) = 0.25x$，CVXPY 求解）。
   - 针对外环感知（~4–5 s）与内环控制（$\Delta t = 0.1\text{ s}$）的尺度分离，引入单调递减、非边界、有限探测半径 $D$ 的保守漏检模型 $\underline{m}(\delta x)$，基于逆正则化距离 $d^{-1}(r; c, \ell) = c/(r+\ell)$、非负上鞅（NSM）与 Ville 不等式证明 **Theorem 1**：当每次移动收集足够观测满足期望上界 $c(\delta-\gamma)/\ell$ 时，全程离开安全集概率 $\mathbb{P}[\exists t: x_t \notin \mathcal{S}] \le \delta$。附录 § B-D 给出 NOPM 与最大通行速度 MTS 两种等价优化形式，并在 $D=4\text{ m}$、最大速度 $0.35\text{ m/s}$、延迟约 $3\text{ s}$（$\kappa_{\text{MTS}}^*=3$）与 75% 探测率下界条件下求得 $\delta=0.1$（90% 安全穿越证书）。

### 3. 实验配置与关键数据（§ V & Appendix B）
- **仿真（NVIDIA Isaac Sim，60 次评估）**：Warehouse、Hospital、Outdoor Residential 3 个环境，每环境 2 个安全任务 + 2 个不安全任务（共 12 个任务各重复 5 次）。使用 4-bit 量化 Gemma 3 27B（Ollama 部署于 L40 GPU）、SAM3、OpenCV、CVXPY、ROS2，丢弃 <3 m 与 >7 m 深度。
  - 总成功率：Oracle 98.3%（安全 100%、不安全 96.6%）；**CORE 95.0%**（安全 96.6%、不安全 93.3%；上下文失败 3.3%、落地失败 1.7%）；No Context 55.0%（安全 93.3%、不安全 16.6%；上下文失败 43.3%）；Geometric 50.0%（安全 100%、不安全 0%）。
- **VLM 对比与结构化推理消融（100 张实验采样图像，4-bit 量化，L40 GPU）**：
  - Gemma 3 27B：总成功 88.0%（安全 91.0%、不安全 85.0%），延迟 4.1 s，显存 20.4 GB；Gemma 3 12B：75.0%（83.0% / 67.0%），2.2 s，10.4 GB；Qwen3-VL 30B：75.0%（77.0% / 73.0%），26.6 s，24.3 GB；Llava 34B：52.5%（49.0% / 56.0%），6.0 s，21.4 GB；Llava 13B：0.0%（无法输出合法 JSON 谓词），2.7 s，11.3 GB。
  - 提示词消融（基于 Gemma 3 27B）：Full 91.0% / 85.0%（4.1 s）；No Safety Logic 90.0% / 80.0%（4.1 s）；No CoT Reasoning 61.0% / 77.0%（2.8 s）。
- **模块延迟（表 A.I）**：Safety Reasoning 4.37 ± 0.45 s；Semantic Grounding 0.75 ± 1.14 s；Safety Enforcement 0.04 ± 0.15 s。
- **硬件部署（Boston Dynamics Spot，实验室 3 安全 + 3 不安全任务各 5 次，共 30 次）**：因 Spot 真实 RGB-D 深度稀疏且带噪声，用 DepthAnythingV3 预测平滑深度并以中值拟合尺度；**CORE 总成功率 86.6%**（安全 86.6%、不安全 86.6%；上下文失败 3.9%、落地失败 9.0%），Geometric 总成功率 50.0%（安全 100%、不安全 0.0%）。

### 4. 局限与开放问题（§ VI）
- 感知不确定性建模基于平均检测错误率，尚未支持逐帧预测不确定度估计；VLM 约 4.1–4.37 s 延迟制约高速运动场景（文中设定最大速度 $0.35\text{ m/s}$、有效深度窗 3–7 m）。
- 当前安全集为纯空间（spatial）定义，未覆盖时变动态约束（如 STL 时序规范）；动力学局限于控制仿射模型；安全滤波器与上层规划器解耦，尚未联合改写名义路径规划。
