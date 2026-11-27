# EmbodiedLGR: Integrating Lightweight Graph Representation and Retrieval for Semantic-Spatial Memory in Robotic Agents

- **作者：** Paolo Riva, Leonardo Gargani, Matteo Frosi, Matteo Matteucci
- **机构：** Politecnico di Milano (Department of Electronics, Information, and Bioengineering, DEIB)
- **版本 / 载体：** arXiv:2604.18271v2（`cs.RO`，2026-09-04 修订；v1 提交于 2026-04-20）；IEEE/RSJ IROS 2026 录用（8 页，3 图）
- **原文：** <https://arxiv.org/html/2604.18271>（摘要页：<https://arxiv.org/abs/2604.18271>）
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27，arXiv v2 全文、摘要页与注释均未提供独立项目页或官方 GitHub/Hugging Face 代码仓库链接，亦未宣称代码发布计划）
- **核对范围：** 通读 arXiv v2 HTML 全文，包括 Abstract、§I Introduction、§II Related Work（II-A/B/C）、§III Method（III-A Memory Building、III-B Memory Retrieval、III-C Inference，含公式 1–6）、§IV Experimental Setup、§V Results（Table I、Table II）、§VI Real-World Deployment、§VII Conclusions、Fig. 1–3 图注及 References [1]–[21]。
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[EmbodiedLGR 实体页](../../wiki/entities/paper-agentic-2604-18271.md)

### 1. 研究动机与核心定位

- **问题痛点：** 面向复杂非结构化环境的移动机器人需要维护长程语义-空间-时间记忆以支撑具身问答（EQA）和人机交互（HRI）。既有方案如密集 3D 场景图（Hydra、ConceptGraphs）、语义特征场（CLIP-fields、VLMaps）或单调追加帧级描述的 RAG 方案（ReMEmbR、Meta-Memory）存在两类瓶颈：依赖大参数量 VLM（如 VILA1.5-13B、Qwen2.5-VL-7B）难以本地边缘实时部署；对同一实体的重复观测未做去重合并，导致向量库冗余膨胀、检索与推理延迟升高。
- **核心解法（EmbodiedLGR-Agent）：** 将记忆结构拆分为双层混合架构：
  1. **语义空间记忆图（NetworkX）：** 存储低层对象级三元组 $\{E_j, P_i, T_i\}$（标签嵌入、观测位姿、时间戳），在探索时基于空间距离阈值 $\delta_P$ 与语义余弦相似度阈值 $\delta_E$ 实时合并与更新同一实体，避免重复节点堆积。
  2. **向量数据库（Milvus + ReMEmbR）：** 存储高层次整帧视觉描述三元组 $\{\mathbf{e}_i, P_i, T_i\}$，仅当图检索工具不足以回答复杂场景语义问题时由 LLM 触发回退调用。

### 2. 方法关键参数与公式（§III & §IV）

- **采样与感知：** 以每 2.0 秒 1 帧（0.5 Hz）采样 RGB 图像与机器人位姿/时间戳；本地运行轻量化 VLM `Florence-2`（`Florence-2-base` 0.22B 或 `Florence-2-large` 0.77B，最大生成长度 1024 tokens）：
  - `DENSE-REGION-CAPTION` 任务提取局部对象语义标签 $\{O_1, \dots, O_n\}_i$；
  - `MORE-DETAILED-CAPTION` 任务生成全局场景描述 $D_i$；
  - 文本编码器统一使用 `all-MiniLM-L6-v2` 计算嵌入 $\{E_j\}$ 与 $\mathbf{e}_i$。
- **实体节点更新规则（式 1–2）：**
  - 空间邻近判断：$\|P_k - P_h\| \le \delta_P$（实验设定 $\delta_P = 5\text{ m}$）；
  - 语义等价判断：$\frac{E_h \cdot E_k}{\|E_h\|_2 \|E_k\|_2} > \delta_E$（实验设定 $\delta_E = 0.75$，合并如 `"cup"` 与 `"mug"` 等近义标签）；
  - 满足条件时对 $\{x, y, z\}$ 采用滑动平均更新（保留初始 yaw）并刷新时间戳 $T_k$；单帧同义多实例（观测 $k$ 个、半径内已有 $h$ 个）按 $k < h$ 更新最近 $k$ 个、$k > h$ 新建 $k - h$ 个节点处理。
- **三类图检索工具与回退机制（式 3–6）：**
  - 语义检索 $t_S(query) \to \{E, P, T\}_{1:k}$（余弦相似度 Top-$k$）；
  - 空间位置检索 $t_P(x, y, z) \to \{E, P, T\}_{1:k}$（$L_2$ 欧氏距离 Top-$k$）；
  - 时间检索 $t_T(hh, mm, ss) \to \{E, P, T\}_{1:k}$（$L_1$ 时间差 Top-$k$）；
  - 推理循环使用 `GPT-4o`（经由 LangChain 调用，系全栈唯一云端组件），按有限状态循环调用工具直至作答或达迭代上限；回退率定义为 $Fallback = N_{vector} / N_{queries}$。

### 3. 实验结果核对（NaVQA 120 题，§V Table I & II）

- **评测口径：** 空间正确判定为预测坐标与真值 $L_2$ 误差 $\le 25\text{ m}$（面向 2.0 s 降采样与 <1B VLM 的区域级定位口径）；时间正确判定为 $L_1$ 误差 $\le 3\text{ min}$。
- **Table I 准确率（%）：**
  - `Florence-2-base`：Graph Memory (Pos `13.75`, Temp `20.48`, Avg `17.11`)；ReMEmbR (Pos `6.67`, Temp `38.63`, Avg `22.65`)；Graph Memory + ReMEmbR (Pos `19.59`, Temp `30.30`, Avg `24.94`)。
  - `Florence-2-large`：Graph Memory (Pos `13.75`, Temp `14.82`, Avg `14.29`)；ReMEmbR (Pos `19.59`, Temp `38.63`, Avg `29.11`)；Graph Memory + ReMEmbR (Pos `29.17`, Temp `38.63`, Avg `33.90`)。
- **Table II 延迟（s）与回退率（%）：**
  - `Florence-2-base`：Graph (`10.80 s`)、ReMEmbR (`19.14 s`)、Graph + ReMEmbR (`23.13 s`，回退率 `93.33%`)。
  - `Florence-2-large`：Graph (`9.97 s`)、ReMEmbR (`19.79 s`)、Graph + ReMEmbR (`23.73 s`，回退率 `80.83%`)。

### 4. 真机部署与原文披露局限（§VI & §VII）

- **真机系统：** 仿真于 Gazebo（AgileX Scout，Ryzen 7 3700X + RTX 3060 12 GB）；真机部署于 **AgileX Bunker Pro**（NVIDIA Jetson AGX Orin 64 GB + ZED X 相机 + Velodyne 3D LiDAR），基于 ROS 2、Nav2 与 ROSA 智能体框架实现。`Florence-2-large`（非量化 0.77B）、`all-MiniLM-L6-v2`、NetworkX 与 Milvus 全部本地运行于 Jetson，仅 `GPT-4o` 通过 API 访问。
- **校园 30 分钟实测（最高遥控速度 1.5 m/s）：** 能正确响应功能语义/原子物体导航查询（如维修机器人的实验室入口位置、最近消防栓位置并下发 Nav2 目标），但在细粒度外观属性查询（如“带红木色调的长椅”）上会遗漏细节并返回泛化长椅列表。
- **原文明确指出的局限：** 强依赖云端 LLM 进行工具调用与结果解析；<1B 轻量 VLM 牺牲了细粒度场景描述精度；当前空间坐标记录的是**观测时机器人的位姿**而非物体精确 3D 世界坐标；尚未构建层级化记忆图以支持更复杂的空间推理。
