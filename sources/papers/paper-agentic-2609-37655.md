# Exemplar2VQA: A Scalable Exemplar-Driven Visual Question Answering Generation Framework via Multi-Agent Coding

- **作者：** Jiayu Ying、Qijian Tian、Ruijie Xu、Xinnan Zhu、Daoguo Dong、Jiachen Xu、Xin Tan
- **机构：** 华东师范大学（East China Normal University）、上海交通大学（Shanghai Jiao Tong University）、复旦大学（Fudan University）、腾讯（Tencent）
- **版本 / 载体：** arXiv:2609.37655v1 [cs.CV]（2026-09-29 提交，NeurIPS 2026 录用）
- **原文：** [arXiv HTML v1](https://arxiv.org/html/2609.37655)、[arXiv 摘要页](https://arxiv.org/abs/2609.37655)
- **项目页 / 代码：** <https://github.com/yingjiayu12/Exemplar2VQA>（**已开源**：官方公开代码仓 `yingjiayu12/Exemplar2VQA`，默认分支 `main`，协议 `CC-BY-4.0`；包含 Track 1 AI2-THOR 相机采集多智能体流水线 `Track1/Track1_Pipeline/`、Track 2 QA 代码生成与几何工具流水线 `Track2/Track2_pipeline/`、跨轨道数据桥接脚本 `prepare_data.py`、统一模型后端封装 `shared/llm_backend.py` 及 3 个示例 Holodeck 场景；完整大规模生成需另行配置 Holodeck/Objaverse 资产、AI2-THOR 与本地模型权重或 API）
- **核对范围：** 全文通读 arXiv HTML v1（`paper_text.txt` 第 1–1430 行，110,931 字符），覆盖 Abstract、§1–5、Code 1、Table 1–6、Figure 1–2 题注与正文分析、References [1]–[56] 及 Appendix A.1–A.8（含 Table A1–A5、NeurIPS Paper Checklist）；并核对官方仓库快照 `project_page.txt`（第 1–325 行）及 `candidate_repos` 中的 `README.md` 与 `files_sample`。
- **通读：** 已完成

## 原文核对笔记

### 1. 研究背景与核心问题（§1, §2）

1. **现有 3D 空间 VQA 数据获取的三大瓶颈：**
   - **人工标注难扩展：** 多视角 3D 空间关系标注成本极高（如 MMSI-Bench 仅 1,000 题）。
   - **LLM/VLM 直接标注易幻觉：** 模型缺乏确定性 3D 几何计算能力，在遮挡判断、跨视角一致性和异中心（allocentric）视角推理上容易产生逻辑矛盾。
   - **传统规则流水线僵化：** 硬编码模板难以根据用户自定义查询结构或跨基准形式动态扩展。
2. **Exemplar2VQA 的基本范式：**
   - 形式化定义为确定性映射 $\mathcal{D} = \Phi_{\mathrm{E2V}}(\mathcal{S}, \mathcal{E}) = \{(\mathcal{V}_i, Q_i, A_i)\}_{i=1}^N$，输入仿真环境 $\mathcal{S}$（AI2-THOR / ProcTHOR / Holodeck）与用户提供的静态空间查询示例（exemplar）$\mathcal{E}$，输出多视角视觉观测 $\mathcal{V}_i$、实例化问题 $Q_i$ 与程序计算真值 $A_i$。
   - 将 LLM 的角色从“直接猜空间答案”转变为“编排确定性几何工具并编写可执行 Python 脚本的编码智能体”。

### 2. 双轨四角色多智能体编码与几何 API（§3）

1. **统一的同构四角色拓扑（§3.2）：**
   - 两个轨道均使用同一开源基座模型 **`Qwen3-Coder-30B-A3B-Instruct`** 实例化 4 个独立角色（Architect、Coder、Reviewer、Refiner），通过独立系统提示、隔离上下文窗口与轨道专用工具解耦语义规划、几何编码与错误诊断。
2. **Track 1：相机生成轨道（Camera Generation Track）：**
   - **Stage 1 语义解析（CameraArchitect）：** 将自然语言采集指令 $I_{\mathrm{NL}}$ 解析为结构化 JSON 轨迹任务 $\mathcal{T}_{\mathrm{json}}$（目标物体过滤 $\mathcal{F}$、轨迹类型、视角 $\theta$、最小可见像素等视觉约束 $\mathcal{C}$），并按目标包围盒最大尺寸 $D_o \in \mathbb{R}^3$ 计算自适应相机半径：$r(o) = r_{\mathrm{base}} + \frac{1}{2}\max(D_o)$（Eq. 1）。
   - **Stage 2 相机代码生成（CameraCoder）：** 将 $\mathcal{T}_{\mathrm{json}}$ 编译为可执行的 Python 导航脚本 $z_{\mathrm{cam}}$，计算相机位姿序列 $P \in SE(3)$ 并调用 AI2-THOR 接口采集图像 $\mathcal{V}_i$ 与去重场景元数据。
   - **Stage 3 物理反馈闭环修复（CameraReviewer + CameraRefiner）：** 在 AI2-THOR 物理引擎中运行 $z_{\mathrm{cam}}$，遇到碰撞、穿墙或越界渲染回溯 $e_{\mathrm{sim}}^{(k)}$ 时由 Reviewer 诊断、Refiner 迭代修补：$z_{\mathrm{cam}}^{(k+1)} = \mathcal{R}_{\mathrm{cam}}(z_{\mathrm{cam}}^{(k)}, e_{\mathrm{sim}}^{(k)})$（Eq. 2）。
3. **Track 2：QA 生成轨道（QA Generation Track）：**
   - **Stage 1 语义解析（QA Architect）：** 将具体示例 $\mathcal{E}$ 逆向抽象为带占位符 $\mathbf{V} = \{v_{\mathrm{cat}}, v_{\mathrm{count}}, v_{\mathrm{view}}, \dots\}$ 的参数化模式 $\Phi_{\mathcal{E}}$。
   - **Stage 2 QA 代码生成（QA Coder）：** 调用几何 API 库生成面向元数据 $\mathcal{M}$（包围盒与可见视角记录）的 Python 脚本 $z_{\mathrm{qa}}$。
   - **Stage 3 解释器自纠错与批量实例化（QA Reviewer + QA Refiner）：** 根据 Python 运行异常 $e_{\mathrm{py}}^{(k)}$ 迭代修复：$z_{\mathrm{qa}}^{(k+1)} = \mathcal{R}_{\mathrm{qa}}(z_{\mathrm{qa}}^{(k)}, e_{\mathrm{py}}^{(k)})$（Eq. 3），编译通过的 $z_{\mathrm{qa}}^*$ 枚举场景元数据组合生成 $\mathcal{D}_{\mathrm{qa}}$。
4. **几何 API 编排（§3.3, Code 1）：**
   - 暴露 6 个核心确定性接口：`get_object_xz_points`（3D 映射 2D `[X, Z]` 足迹）、`calculate_planar_angles`、`is_spatial_relation_satisfied(dir, ref, tgt, th=0.15)`、`calculate_projected_distance`、`is_angle_ambiguous(angles, threshold=10)`（过滤靠近坐标轴边界的模糊方向查询）及 `cal_3d_bbox_distance_between_categories`。

### 3. 实验设置、主评测、泛化与消融结论（§4, Appendix A.1–A.6）

1. **合成与微调配置（§4.1, Appendix A.1）：**
   - 双轨数据生成运行于 2× NVIDIA RTX A6000；微调使用 swift 框架 + DeepSpeed ZeRO-2 + NEFTune（$\alpha=5$）于 4× H200；评测于单卡 H200。
   - **MMSI-Bench（1,000 多选题）：** 合成约 **6K** 条多图 QA，LoRA（$r=16, \alpha=32$，lr `1e-4`，3 epochs）与 Full SFT 微调 `Qwen2.5-VL-7B`：Base **25.9%** → LoRA **28.2%**（Cam.-Cam. 由 24.7% 升至 **33.3%**，零样本 Appr.* 由 18.2% 升至 **27.3%**；但静态训练使零样本动态追踪 Obj.* 从 39.5% 降至 25.0%）；Full SFT 达 **28.0%**；3 组随机种子平均 **29.33 ± 0.99%**（Table A4）。
   - **VSI-Bench（选择题子集 + 数值子集）：** 合成约 **10K** 条并拼接为连续旋转视频，全参 SFT 微调 `Qwen2.5-VL-3B`（lr `1e-5`，3 epochs）：选择题平均分由 **35.3%** 升至 **42.9%**（+7.6%，超越 GPT-4o 34.6%、Gemini-1.5 Pro 42.1% 与 LLaVA-Video-72B 40.7%），零样本 Route Plan* 由 28.9% 升至 **35.1%**；跨架构迁移至 `InternVL2-2B`（28.2% → **35.6%**）与 `InternVL2-8B`（36.7% → **46.1%**）（Table A3）；4 项数值任务均值由 22.5% 升至 **37.4%**（Table A5）。
2. **零样本跨基准与室外泛化（§4.3, Appendix A.3）：**
   - 直接复用仅在室内合成数据上微调的 `Qwen2.5-VL-7B`（MMSI-Bench 6K LoRA）：
     - **SpaCE-10（Table 3）：** Base **33.3%** → **42.2%**（+8.9%），其中 Size Assessment +18.2%、Function Reasoning +16.1%，Scene Quant. 降为 24.2%。
     - **ViewSpatial-Bench（Table 4，含室外与混合场景）：** Base **36.9%** → **45.4%**（+8.5%），人视角朝向 `Obj. Ori. (Per)` 从 37.1% 跃升至 **60.6%**（+23.5%）。
     - **Spatial-MM / Spatial-Obj（Table A2，2,000 条真实生活场景）：** Base **69.71%** → **70.50%**。
     - **OSR-Bench（Table A1，360° 全景，另用 6K 稀疏四视图合成微调）：** Base **28.3%** → **36.5%**，高于 `Qwen2.5-VL-72B`（33.5%）。
3. **消融与直接 LLM 标注对比（§4.4, §4.5, Table 5–6）：**
   - **多智能体拓扑（100 个种子样本，Table 5）：** 1 Agent 全合并 QA **34.0%** / 相机 **60.0%** → 2 Agents（72.0% / 74.0%）→ 3 Agents（85.0% / 80.0%）→ 4 Agents（**92.0%** / **84.0%**）。将生成基座降为 `Qwen3.5-9B` 后跌至 48.0% / 56.0%。
   - **对比直接 LLM 标注（Table 6）：** 用 `Gemini3.5-Flash` 直接标注 10K VSI-Bench 样本微调 `Qwen2.5-VL-3B`，均分反而从 Base 35.3% 降至 **28.2%**（Appr. Order 跌至 6.3%）。
4. **效率与误差分析（§4.6, Appendix A.5–A.6）：**
   - 代码合成阶段为一次性开销：相机轨迹平均 **3 分钟/任务**，QA 脚本平均 **2 分钟/任务**；脱离 LLM 后，相机渲染约 **15 秒/场景**，100 场景 QA 批量实例化约 **25 秒**。
   - 在 4-Agent 配置剩余失败模式中，Track 1 的 FoV 遮挡/无效观测占比由单智能体的 15.0% 降至 **6.2%**，轨迹语义偏差占比 **62.5%**；Track 2 的代码执行失败由 30.3% 降至 **25.0%**，缺失预期 QA 占 **37.5%**。

### 4. 局限性（Appendix A.7）

1. 当前主要面向静态、以物体为中心的空间配置，尚不涵盖连续动态时空事件（如物体跟踪，实验中也导致 MMSI-Bench 的 `Obj.*` 指标下降）。
2. 依赖仿真器提供的显式 3D 场景元数据，尚不能直接从无标注真实世界照片反向合成 QA。
3. 多智能体代码生成成功率受限于基座代码大模型的能力边界与物理引擎偶发边界故障。

对 wiki 的映射：[Exemplar2VQA 论文实体](../../wiki/entities/paper-agentic-2609-37655.md)。
