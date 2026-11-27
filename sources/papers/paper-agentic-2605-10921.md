# RoboMemArena: A Comprehensive and Challenging Robotic Memory Benchmark

- **作者：** Huashuo Lei, Wenxuan Song, Huarui Zhang, Jieyuan Pei, Jiayi Chen, Haodong Yan, Han Zhao, Pengxiang Ding, Zhipeng Zhang, Lida Huang, Donglin Wang, Yan Wang, Haoang Li
- **机构：** 香港科技大学（广州）（HKUST-GZ）、浙江大学（ZJU）、西湖大学（Westlake University）、清华大学（Tsinghua）、浙江工业大学（ZJUT）、上海交通大学（SJTU）
- **版本 / 载体：** arXiv:2605.10921v1 [cs.RO]（2026-05-11 提交，HTML 全文版）及官方项目页
- **原文：** https://arxiv.org/html/2605.10921 、https://arxiv.org/abs/2605.10921 、https://robomemarena.github.io
- **项目页 / 代码：** 已开源 — 官方代码仓库 [OpenHelix-Team/RoboMemArena](https://github.com/OpenHelix-Team/RoboMemArena)（含 26 项任务 BDDL、LIBERO 兼容评测分支、`RoboMemArena_dataset_builder.py`、异步参考评测脚本与 PrediMem S2 `predictive_coding_head/` 训练插件），数据集托管于 Hugging Face（`RoboMemArenaBenchmark/RoboMemArena`）与 ModelScope（`haodong123`），模型权重元数据标注为 `huashuolei/PrediMem`；另论文 HTML 中发现的 `Diffusion-CoT/reflection2perfection` 仅为其项目页所套用的学术网页模板，非本工作代码仓库。
- **核对范围：** 通读 `paper_text.txt`（arXiv HTML 全文第 1–6 节与附录第 7–14 节，含 Table 1–4、Table S1–S4、Algorithm 1、公式 1–9 及全部参考文献列表）与项目页文本 `project_page.txt`（26 任务清单与结果摘要），并核对官方 GitHub 仓库 `README.md` 与目录文件清单 `files_sample`。
- **通读：** 已完成

## 原文核对笔记

- **基准定位与规模：** RoboMemArena 面向部分可观测长程操作中的具身记忆瓶颈，包含 26 项仿真任务（4 类失败模式：Multi-Object Transferring 4 项、Multi-Object Occlusion 11 项、Multi-Object Counting 7 项、Multi-Object Sequence 4 项），每任务采集 100 条成功演示（共 2,600 条长程视觉轨迹，生成 15,100 个关键帧对齐短片段），平均轨迹长度 1,076 步；总计 151 个子任务中 104 个为历史记忆依赖型（$R_{\mathrm{mem}} = 68.9\%$）。
- **三阶段自动数据管线：**
  1. **VLM 任务分解（Stage 1）：** 基于高层次指令与当前 RGB 图像，由 VLM 生成绑定至预定义规划器集合 `{Move, Place, Pour, Open, Close}` 的 JSON 子任务列表，并对不一致分解做人工精修。
  2. **AnyGrasp 闭环执行（Stage 2）：** 利用基于点云的 6-DoF 抓取位姿估计器 AnyGrasp 与预定义原子原语自主执行子任务，配合后置条件检查器对失败子任务自动重试。
  3. **多条件关键帧提取（Stage 3）：** 取物理交互锚点 $\mathcal{K}_{\mathrm{phys}} = \{t \mid g_t \neq g_{t-1}\}$（夹爪开合状态变化）与运动学拐点 $\mathcal{K}_{\mathrm{kin}}$（末端线速度低于 $\epsilon$ 或相邻速度余弦相似度低于 $\cos\theta$）的并集 $\mathcal{K} = \mathcal{K}_{\mathrm{phys}} \cup \mathcal{K}_{\mathrm{kin}}$。
- **评测指标与真机套件：** 定义阶段级谓词全满足的 **TSR**（Task Success Rate）与分阶段进度 **CSR**（Cumulative Success Rate，每任务 3–9 个验证阶段）；在 AgileX Cobot Mobile Aloha 双臂平台上配套 5 项真机记忆任务（`Pour Bottle × 2`、`Brush Plates with Swap`、`Transfer Objects`、`Shell Game`、`Imitate Human to Make Breakfast (IHMB)`，每任务 10 次 rollout）。
- **基线模型 PrediMem：** 异步双系统 VLA 架构。高层规划器 S2 基于 Qwen3-VL-8B-Instruct（冻结视觉编码器，其余模块在 4×H100 上以 $10^{-5}$ 学习率全参数微调 2 个 epoch），维护记忆库 $\mathcal{M}_t = \mathcal{M}_t^{\mathrm{key}} \cup \mathcal{M}_t^{\mathrm{rec}}$（最近 5 帧双视角滑窗 + 无上限关键帧缓存），输出严格 JSON（`current_primitive` 与窗口内 1-indexed `keyframe_positions`）；训练阶段引入仅用于训练的预测编码头 $f_{\mathrm{Pre}}$，以停梯度 ViT 下一帧视觉特征为监督（$\mathcal{L}_{\mathrm{Pre}} = \mathrm{MSE} + (1 - \cos)$，权重 $0.1$），推理阶段移除以保持双系统推理开销（S2 约 1.06 Hz，S1 约 3.40 Hz）。
- **关键实验结论：** 仿真中 PrediMem 平均达 38.5% TSR / 55.2% CSR，高于 $\pi_{0.5}$（21.5% / 38.7%）、HiF-VLA（16.9% / 39.8%）、MemoryVLA（15.0% / 35.3%）、MemER（27.3% / 49.1%）及闭源 GPT-5.4（8.7% / 30.5%），但仍低于 Oracle S2（46.1% / 64.8%）；真机 5 项任务平均成功率 PrediMem 为 52%，高于 MemER（40%）与 $\pi_{0.5}$（20%）。
- 对 wiki 的映射：[RoboMemArena 实体页](../../wiki/entities/paper-agentic-2605-10921.md)
