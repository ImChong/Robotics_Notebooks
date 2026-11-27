# MG-VQA: Manipulation Grounded Visual Question Answering with VLMs

- **作者：** Vineet Bhat, Mikaela Angelina Uy, Siyi Chen, Chan Hee Song, Alex Zook, Xuning Yang, Stan Birchfield, Valts Blukis, Jonathan Tremblay
- **机构：** NVIDIA
- **版本 / 载体：** arXiv:2608.17129v3 [cs.CV]（2026-08-17 提交 v1，2026-10-12 修订 v3；v1 原题为 *Probe: Manipulation-Grounded Visual Question Answering with VLM Agents*）
- **原文：** https://arxiv.org/abs/2608.17129 （HTML 全文：https://arxiv.org/html/2608.17129 ）
- **项目页 / 代码：** 已开源 — 官方代码与基准入口 https://github.com/vineet2104/MG-VQA （MIT License；已公开 MG-VQA-Sim 环境与 600 题 MG-VQA-Bench，扩展子集 MG-VQA-Hard 与训练数据生成引擎在仓库 README 标注为计划后续更新）
- **核对范围：** 通读 arXiv HTML 全文（Abstract、§1–§5、Limitations、AI use statement、References、Appendix A–D、Table 1–5、Figure 1–10 图注及系统/用户 Prompt 全文），并核对官方 GitHub 仓库主页、README 及目录文件清单
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [MG-VQA: Manipulation Grounded Visual Question Answering with VLMs](../../wiki/entities/paper-agentic-2608-17129.md)

### 核心问题与任务定义

1. **MG-VQA 设定**：在杂乱桌面场景中，回答问题所需的关键证据在初始视角下被遮挡（堆叠覆盖、压在物体下、朝向背离相机）。机械臂物理操作不是任务终点，而是中间态的信息收集手段（intermediate evidence-gathering operation）。
2. **MG-VQA-Bench（600 题，人工核验，4 类任务各 150 题）**：
   - **Count**（0–5 整数）：统计目标物体实例数量，初始视图目标像素可见度 ≤ 10%，主要失误为过少计数（undercounting）。
   - **Find**（Yes/No + 2D 点坐标 `[row, col]` ∈ `[0, 1000]`）：判断指定物体是否存在；正样本初始像素可见度 ≤ 10%，正负样本各 75 题。
   - **Beneath**（四选一，含 "None of the Above"）：询问被指代遮挡物下方隐藏的物体；通过反事实透明渲染筛选移除遮挡物后可显露的目标（初始可见度 ≤ 10%）。
   - **Compare**（A/B/C 三选一）：比较贴有 `circle-A`、`square-B`、`triangle-C` 贴纸的三个不同物理体积物体的最大或最小者；至少一个标签在初始顶视图被遮挡。
3. **混杂干扰项（Confounding distractors）与人工核验**：构造外观相近的干扰选项防止模型仅凭局部颜色/纹理猜测；每题由 2 名标注员使用能将非关键杂物透明化的核验界面独立检查无歧义性、初始证据缺失性与可恢复性。

### MG-VQA-Sim 与工具接口

1. **仿真与资产栈**：基于 PyBullet（重力、摩擦、刚体接触、抓取与推动）与 UR5 + Robotiq 2F-140 平行夹爪，Blender-EEVEE 渲染 `1024 × 1024` 顶视 RGB 图像；每回合在 `50 × 50 cm` 桌面区域放置最多 15 个物体（含 1 个用于暂存抓取杂物的 clearing basket），物理沉降 30 秒。共使用 319 个真实扫描日常物体（Meta DTC 136 个、MegaPose 来源 155 个、BOP 28 个）及 108 种 AmbientCG 桌面纹理。
2. **5 个标准工具（每轮调用 1 个 JSON 工具）**：
   - `perceive_scene()`：统一调用 Gemini 3.6 Flash 生成 YAML 场景图、桌面描述、可见物体 2D 点及空间关系谓词。
   - `find_object(point)`：调用 SAM3 根据场景图 2D 点输出连续分割掩码。
   - `grasp()`：基于 `find_object` 掩码、顶视深度图和相机内参生成 3D 表征，调用 GraspGen 预测 6-DoF 抓取位姿并将物体放入 clearing basket；可能因掩码不准、无可行抓取、不可达或周围杂物碰撞而失败。
   - `push(direction)`：基于掩码质心将闭合夹爪移至反侧，沿图像坐标 `left/right/up/down` 推 10 cm，易触发邻近物体联动或部分位移。
   - `answer_question(answer[, point])`：终止回合并提交答案。
3. **交互预算**：操作类工具（`grasp` + `push`）共享 10 次预算；感知工具免费，但连续 5 次纯感知调用或耗尽操作预算仍未答题即判错终止。

### 实验结论与关键数值核对

- **Table 1（8 个 VLM 在 600 题上的平均成功率 %）**：
  - 盲猜 Chance 基线：Count 22.0，Find 50.0，Beneath 26.0，Compare 32.7，Avg. 32.7。
  - Direct / Perception 全模型均值仅 36.2% / 37.6%，Manipulation 提升至 56.8%（区间 40.8%–83.0%）。
  - 开源模型 Manipulation Avg.：Gemma 4 31B IT 44.3%、Kimi K3 56.7%、GLM 5.3 40.8%、Qwen 3.8 Flash Next 54.0%（开源均值 49.0%）。
  - 闭源模型 Manipulation Avg.：Gemini 3.1 Pro 64.5%、Gemini 3.8 Flash 69.3%、Claude Opus 4.8 41.5%、GPT 6 Astra 83.0%（闭源均值 64.6%）。
- **Table 2（Prompt 与视角消融）**：Kimi K3 加入 `+ Search` / `+ Search-persistent` / `+ Multiview` 后平均成功率从 56.7% 分别升至 61.8% / 66.0% / 62.3%；Opus 4.8 从 41.5% 变为 44.0% / 41.5% / 43.7%；GPT 6 Astra 加 `+ Search` 从 83.0% 略降至 80.5%。
- **Table 3–4（失败恢复与重锚定）**：GPT 6 Astra 平均每题执行 3.9 次操作（其余模型 0.9–2.9 次）；在抓取失败后的回合中，GPT 6 Astra（`N=502`）82.7% 选择 `push`，且其中 98.8% 先更新场景图再分割（`Scene + seg.`），0.0% 沿用过时场景图；GLM 5.3（46.3%）和 Claude Opus 4.8（23.6%）则常在失败后直接答题（`None`）。
- **Table 5（MG-VQA-Hard，40 题，每任务 10 题）**：由 GPT 6 Astra 辅助设计的对抗性难题，利用平放分散物体不藏小件的先验偏见与顶视角高度估计盲区；5 个评测模型平均成功率降至 17.5%–27.5%。
