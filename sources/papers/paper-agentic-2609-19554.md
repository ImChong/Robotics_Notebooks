# VABench: Measuring Embodied Spatial Intelligence through Visual Demonstrations, Active Perception, and Metric Control

- **作者：** Zhongbo Zhang*, Jiayi Jin*, Yifan Wang, Zaibin Zhang, Haiwen Diao, Lijun Wang, Huchuan Lu（* 共同一作）
- **机构：** 大连理工大学（Dalian University of Technology）、南洋理工大学（Nanyang Technological University）
- **版本 / 载体：** arXiv:2609.19554v1 [cs.RO]（2026-09-17）
- **原文：** https://arxiv.org/abs/2609.19554 、https://arxiv.org/html/2609.19554
- **项目页 / 代码：** 已开源 — https://github.com/zhangzhongbo2213/VABench （MIT License；资产发布于 [GitHub Releases](https://github.com/zhangzhongbo2213/VABench/releases/tag/assets-v1) 与 [Hugging Face Datasets](https://huggingface.co/datasets/zhangzhongbo2213/VA-Bench)）
- **核对范围：** 通读 `paper_text.txt` 全文（Abstract、§1–§6、Table 1–6、Appendix A–D、Table 7–28）与 GitHub 仓库主页 `project_page.txt` 及 `context.json` 中仓库文件清单
- **通读：** 已完成

## 原文核对笔记

- 对 wiki 的映射：[VABench: Measuring Embodied Spatial Intelligence through Visual Demonstrations, Active Perception, and Metric Control](../../wiki/entities/paper-agentic-2609-19554.md)

### 1. 研究问题与协议边界
- **核心问题**：评测通用多模态大语言模型（MLLM）在部分可观条件下，能否独立闭合「从 RGB 示范提取任务程序上下文 → 主动选择相机视角获取缺失空间证据 → 输出世界系度量笛卡尔动作指令 → 根据执行反馈在线修正」的完整闭环（observe–reason–act–revise）。
- **特权信息隔离**：模型仅可见任务指令、基于示范视频自生成的文本摘要 $s_{m,\tau}$、当前纯净 RGB 图像 $I_t$（无关键点/掩码/坐标轴叠加）、滤波后的本体感知 $p_t$（末端位姿、夹爪开合、指尖中点 GC 与世界系下的夹爪局部轴），以及历史动作与规划器二值反馈 $f_{t-1}$。模型**无法**获取目标位姿、深度图、点云、分割掩码、接触状态、专家轨迹、相机标定矩阵或验证探针。
- **固定执行器**：采用基于 RoboTwin 的模型无关固定逆运动学（IK）与轨迹规划层执行模型输出的度量目标（世界系平移 $1\text{–}100\text{ mm}$、夹爪局部轴旋转 $1\text{–}90^\circ$、夹爪开合及双臂同步指令），不训练或挂载特定动作头（action head）或 VLA 策略。

### 2. 评测套件与种子验证
- **Base Track（14 个任务族）**：11 个单臂任务 + 3 个双臂任务（分属 Direct grasp grounding 5 个、Placement and stacking 4 个、Functional contact and tool use 2 个、Bimanual coordination 3 个）。每个任务保留 20 个经物理静置与完整 `task.play_once()` 专家执行验证的确定性种子（共 280 个实例，来自 318 个候选探测，条件通过率 88.1%）。主评测重复 3 次独立运行（每模型 840 局）。
- **Controlled Transfer Track**：7 个单臂任务族的留出（held-out）几何、容器或布局变体，直接复用基础任务的示范摘要，不提供新示范视频。
- **Long-Horizon Composition Track**：五物体放篮长程组合任务（20 局、共 100 次物体放置），仅提供 4 个原子技能视频而不提供完整任务示范。

### 3. 主动感知接口与行为诊断体系
- **离散有界相机动作 $\mathcal{A}_{\mathrm{cam}}$**：5 个以 GC 为中心的语义视角（`view_topdown`、`view_side`、`view_front_side_45`、`view_side_top_45`、`view_oblique_45`，双臂模式下支持 `view_left_*` / `view_right_*` / `view_both_*`）及增量调整（相机系平移 5 cm、缩放 8 cm、偏航/俯仰 $10^\circ$）。相机与机器人动作共享单局步数预算，且限制连续纯相机动作不超过 5 步；接口不模拟物理相机臂碰撞。
- **九维轨迹诊断与子任务进度**：由 3 名标注员在单次完整运行（12 个主模型共 3,360 局、29,232 个有效评估项）上采用两阶段多数投票标注（全员一致率 95.69%，Fleiss $\kappa=0.942$）：空间感知（TL/AE/SR）、机器人操作（MS/MP/FG）、错误恢复（ED/OC/PF，仅对可观测错误局计分，无错局记 NA）。另有检查点微平均子任务完成度（420 个单臂 + 140 个双臂检查点）。

### 4. 核心实证结果（§4–§5, Appendix A）
- **基线成功率（Table 3）**：头部 3 个模型处于同一区间且 3 次运行范围重叠——`Qwen3.8-max` $53.93 \pm 3.17\%$（单臂 $65.61\%$ / 双臂 $11.11\%$）、`Opus-5` $52.86 \pm 2.79\%$、`GPT-5.6-sol` $51.55 \pm 4.31\%$（双臂最高 $22.22\%$）；其后大幅跌落至 `GPT-5.6-terra` $23.10\%$ 及以下（其余均 $\le 15.00\%$）。所有模型在 `handover_horizontal_block` 上均为 $0.00\%$。
- **感知与执行的错配（Table 11）**：头部模型 TL 达 $100.0\%$、MS $\ge 99.6\%$，但空间关系 SR 仅 $70.4\%\text{–}78.9\%$、在线纠偏 OC 最高仅 $46.7\%$（`Qwen3.8-max`）。
- **主动视角 vs. 被动五视角（Table 4）**：匹配对照中，取消主动相机控制改为每步提供固定 5 视角，4 个测试模型成功率全面下降 9.64–29.64 个百分点（`Qwen3.8-max` 从 $57.50\%$ 降至 $27.86\%$，`GPT-5.6-sol` 从 $56.07\%$ 降至 $33.93\%$，`Opus-5` 从 $54.29\%$ 降至 $30.36\%$，`Doubao-2.1` 从 $15.00\%$ 降至 $5.36\%$）。
- **受控迁移与长程组合（Table 5）**：留出几何迁移使 `GPT-5.6-sol` 下降 32.14 pp（$80.00\% \rightarrow 47.86\%$），`Qwen3.8-max` 下降 10.00 pp（$77.86\% \rightarrow 67.86\%$）。五物体长程任务中，无模型完成哪怕一局严格成功（`Qwen3.8-max` 完成 61/100 个子放置，`GPT-5.6-sol` 完成 53/100）。
- **示范摘要迁移与本地小模型（Table 6, 19）**：强执行者的隐式摘要反而拖累其他读者（`Qwen3.8-max` 摘要使另 4 模型平均下降 31.25 pp），而约束显式的 `Opus-5` 摘要平均提升另 4 模型 +7.50 pp。本地小模型在 `Qwen3.8-max` 摘要引导下仅达到 $0.71\%\text{–}9.64\%$（仅限抓取和按铃）。
