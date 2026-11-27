# Code4Scene: Benchmarking Coding Agents for Constructing and Editing 3D Scenes

- **作者：** Xiaokang Ye, Siddhant Hitesh Mantri, Zimeng Chen, Edward Zhang, Zhaoxu Zheng, Yuanheng Li, Yizhao Chen, Tianyang Huang, Lianhui Qin
- **机构：** UC San Diego（UCSD）、UC Berkeley
- **版本 / 载体：** arXiv:2609.36777v2（2026-09-29 提交 v1，2026-11-17 修订 v2，46 页含附录，CC BY 4.0）
- **原文：** https://arxiv.org/abs/2609.36777 （HTML 全文：https://arxiv.org/html/2609.36777 ）
- **项目页 / 代码：** 项目页：https://simworld.org/code4scene （v2 摘要披露；缓存检索时项目页抓取状态为 0，未取回独立网页正文）；代码状态：宣称将开源或未列链接（Appendix A 声明将发布公开集任务代码、评测与分析脚本、智能体配置记录、共享 skill 规范与工具 schema 及裁判指令，但原文及缓存候选仓库列表中均未提供 GitHub 仓库链接）
- **核对范围：** 完整通读 arXiv HTML 全文（`/tmp/agentic_172_cache/paper-agentic-2609-36777/paper_text.txt` 第 1–3410 行），覆盖 Abstract、Section 1–3（任务设定、数据构建、评测流水线与指标、14 系统主实验、评测器验证、失败分析）、Table 1–19（含公共集/私有集成绩、工具接口 25 个工具、置信区间、分层难度、操作分布、Token 成本与下游 EUS 导航评测），以及 Appendix A–F（接口细节、案例构建、公共/私有集划分、VLM 与规则评测协议、相关工作对比、局限性与影响）。
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[Code4Scene 知识实体页](../../wiki/entities/paper-agentic-2609-36777.md)

### 1. 研究问题与任务形式化

- **核心定位**：将三维场景构建与编辑形式化为面向可执行引擎原生工件（Unreal Engine 5.8.0 `.umap` 关卡）的智能体代码生成（agentic code generation for 3D scenes），而非评估单一重建算法（如 3DGS）或纯渲染视图生成。
- **统一函数接口**：给定人工搭建的参考场景 $\mathcal{L}^\star$、资产目录 $\mathcal{C}$、初始场景 $\mathcal{L}^0$、智能体可见证据 $E$、25 工具共享接口 $\mathcal{A}$ 与资源预算 $B$，智能体通过执行代码与工具调用输出持久化场景 $\hat{\mathcal{L}}$：
  $$f: (\mathcal{L}^0, \mathcal{C}, E, \mathcal{A}, B) \rightarrow \hat{\mathcal{L}}$$
- **两大互补任务**：
  1. **Text-to-Scene (T2S) Construction**：$\mathcal{L}^0$ 为空场景，$E$ 为文本描述（平均 58 条、最多 92 条可核查需求），允许多种合理布局实现；预算 800 次工具调用、180 分钟。
  2. **Image-to-Scene (I2S) Editing**：$\mathcal{L}^0$ 为在人工场景上施加原子/组合破坏（移除、插入、复制、替换、移动、旋转、缩放、镜像等）后的副本，$E$ 为参考图（室内 2 张、室外 1 张），隐藏真值 $H$ 仅评测器可见；室内预算 120 次调用 / 45 分钟，室外 80 次调用 / 20 分钟。

### 2. 数据构建与集划分

- **样本规模**：从 522 个 Fab 内容包中筛选人工搭建的 Unreal Engine 场景，经过资产 AI 使用权限审计与退库剔除（移除 5 个 No-AI 包及未确认权限的 TrainStation 共 20 例），保留 **170 个任务实例**：
  - **26 个 T2S 构筑任务**（130 m 或 200 m 方形底板）。
  - **144 个 I2S 编辑任务**（72 个室内 Indoor + 72 个室外 Outdoor，源自 18 个基础场景，每场景 8 个编辑任务；室内约 95–3,168 个 Actor，室外约 260–6,700 个 Actor）。
- **公开集 / 私有集**：
  - **公开集（83 例）**：16 T2S + 22 Indoor + 45 Outdoor，评测了 14 个智能体系统配置。在 10 个共享配置上，公开集与全量集（170 例）总分相关系数达 $r=0.996$，且全量分差 $\ge 0.02$ 的系统对在公开集上无序位反转。
  - **私有集（87 例）**：10 T2S + 50 Indoor + 27 Outdoor，评测了 10 个配置。

### 3. 共享接口与评测流水线

- **25 个共享工具（Table 3）**：分为资产发现（4）、场景编辑（5）、状态检查（2）、程序控制（`execute_python_script` / `read_job_log`，2）、规划（6，含 `validate_scene_graph`、`solve_scene_graph`、`render_plan`、`tier_derivation`、`ground_pass`、`build_tier`）、校验（4，含 `take_screenshot`、`measure_scene`、`check_floating`、`check_collisions`）、环境与持久化（2，`setup_environment`、`save_scene_as`）。
- **准入门禁（Candidate Integrity）**：独立重开 `.umap` 检查场景图 schema 与最小 Actor 数、依赖包解析、资产库清单一致性；首次失败可重试一次，若均无效则计 0 分。
- **物理安全（Physical Safety, $P$）**：$S_{\mathrm{physics}} = 0.5(1-r_{\mathrm{unsup}}) + 0.5 S_{\mathrm{pen}}$。支撑容差 5 cm（AABB 接地或下层 Actor 垂直间隙）；穿透基于 Unreal `FHitResult` 最小平移深度，T2S 容差为最短 AABB 边长的 5% 限幅于 $[5,50]$ cm，I2S 为固定 5 cm。
- **T2S 语义校验**：
  - Detailed Alignment ($D$)：四类需求权重分别为 Identity/Environment 0.25、Content/Quantity 0.40、Attributes/Materials 0.15、Spatial Composition 0.20；三阶证据阶梯（引擎状态与几何 -> 定向 RGB 截图 -> Qwen 3.8 27B thinking-off 裁判）。
  - Overview Alignment ($O$)：4 张全局视图，Prompt-Aware（对齐 0.40、布局 0.25、风格 0.20、完整度 0.15）乘以 Prompt-Free 结构完整度修正项 $(0.75 + 0.25 S_{\mathrm{struct}})$，若两视图以上确信度 $\ge 0.8$ 检出严重几何缺陷则截断上限为 0.40。
  - 综合分：$S_{\mathrm{T2S}} = 0.2 D + 0.6 O + 0.2 P$（展示语义分 $= 0.25 D + 0.75 O$）。
- **I2S 真值比对校验**：
  - Actor 级一对一最优匹配（容差：位置 $\le 5\text{ cm}$、旋转 $\le 5^\circ$、缩放/包围盒尺寸 $\le 5\%$、资产与材质属性精确一致），不依赖 Actor 名称或 GUID。
  - $\mathrm{TP} = m+d$，$\mathrm{FP} = |C|-m+o$（含错误输出与非目标背景误改/误删），$\mathrm{FN} = (n_+-m)+(n_--d)$。
  - 综合分：$S_{\mathrm{I2S}} = 0.8 F_1 + 0.2 P$；总榜分：$S_{\mathrm{model}} = 0.5 \overline{S}_{\mathrm{T2S}} + 0.5 \overline{S}_{\mathrm{I2S}}$。
- **下游具身可用性（EUS，辅助指标）**：使用固定胶囊体代理（半径 50 cm、高 180 cm、最大坡度 $35^\circ$）测量地面可接近率 $A$ 与路径完成率 $W$，$\mathrm{EUS} = A \cdot W$。

### 4. 核心实验结论与数字

- **主榜单（83 例公开集，14 系统）**：
  - **GPT-6 Astra (max)**：总分 0.658（95% 源聚类 CI $[0.619, 0.700]$），T2S 0.785（Semantic 0.797，Physical 0.738），I2S 0.530（Repair F1 0.462，Physical 0.802）。
  - **Gemini 3.8 Flash (high)**：公开总分 0.644（$[0.596, 0.696]$），T2S 0.675，**I2S 领先（0.613，Repair F1 0.562）**；私有集（87 例）总分 0.664。
  - **Claude Fable 5.1 (max)**：公开总分 0.612，**T2S 领先（0.796，Physical 0.980）**，I2S 0.428（Repair F1 0.339）。
  - **Claude Opus 5 (max)** 0.606，**GPT-5.6 Sol** 0.570，**GLM-5.3 Flash (max)** 0.484，**Muse Spark 1.3** 0.443，**Grok 4.6** 0.434；开源权重模型（Qwen 3.8 27B、Gemma 4 31B、DeepSeek V4.1 Flash）总分在 0.285–0.388 之间，Repair F1 仅 0.001–0.132。
  - T2S 与 I2S Spearman 相关 $\rho = 0.67$；Astra 与 Gemini 的配对总分差为 $0.0134$（$[-0.0402, 0.0665]$），未能区分总分优劣。
- **结构性短板**：
  - **空间组合最弱**：所有 14 个系统在 T2S 四类需求中均在 Spatial Composition 得分最低（Astra 最高仅 0.587）；即便相关实体已确认存在，仍有 **74/387（19.1%，95% CI 12.1–26.6%）** 的可分析空间关系需求被判不匹配。
  - **编辑破坏周围场景**：14 系统在公开集全部 187 次完全恢复目标（Recall=1）的提交中，**64 次（34.2%，CI 25.5–42.3%）** 仍引入非目标假阳性（50 次修改非目标 Actor、27 次多出未匹配新增 Actor）。在 22 个室内任务上，Astra 完全恢复 20 例优于 Gemini 的 17 例，但加上「零误改」条件后反转为 Astra 15 例 vs Gemini 16 例。
  - **修复类别分化**：Astra 擅长 Transform（0.591 vs Gemini 0.529），Gemini 擅长 Layout（0.707 vs Astra 0.618）、Lifecycle 与 Restore（0.411），Fable 在 Symmetry 上以 0.555 略高于 Gemini（0.544）。
  - **执行报错与成本**：85.0% 含报错的提交出现 `AttributeError` 或 `TypeError`；按 2026 年 9 月定价快照，公开集加权单题估算 Token 成本中，Gemini 3.8 Flash 为 $1.780，Astra 为 $6.039，Fable 为 $7.280，Opus 为 $9.863。
  - **评测器验证**：T2S 人工偏好回顾性留出集上，采用权重与 Plackett-Luce 共识一致率达 86.90%（Kendall $\tau_b = 0.738$），个体偏好一致率 79.37%（接近标注员间 80.95%）；固定证据下 VLM 裁判 3 次重复的 $\tau_b$ 为 0.889–0.944。
