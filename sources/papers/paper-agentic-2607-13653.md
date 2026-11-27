# Exploratory, Communicative, and Deployable: Vision-Driven Embodied Agents for Open-World Mobile Manipulation

- **作者：** Boyu Mi, Mengchen Ma, Yifei Yao, Xing Gao, Junting Chen, Yangzi Li, Zihou Zhu, Guohao Li, Zhenfei Yin, Tai Wang, Yao Mu, Jiangmiao Pang, Hanqing Wang
- **机构：** 上海人工智能实验室（Shanghai Artificial Intelligence Laboratory，致谢显式署名；代码托管于 InternRobotics / Sage-PhysicalAI）
- **版本 / 载体：** ECCV 2026 录用论文 / arXiv:2607.13653v2 [cs.CV]（2026-07-15 提交 v1，2026-07-27 更新 v2，57 页）
- **原文：** [arXiv HTML v2](https://arxiv.org/html/2607.13653) / [arXiv Abstract](https://arxiv.org/abs/2607.13653)
- **项目页 / 代码：** 部分开源 — 官方代码仓库 [Sage-PhysicalAI/REAL](https://github.com/Sage-PhysicalAI/REAL)（论文与 README 别名 [InternRobotics/REAL](https://github.com/InternRobotics/REAL)，项目主页 [internrobotics.github.io/REAL](https://internrobotics.github.io/REAL/)，MIT 协议）；已公开 REAL-Bench 241 任务 YAML 定义、MCP 服务端、Qwen/VLM API Agent、程序化数据生成与标注脚本及 Qwen3-VL SFT/GSPO 训练代码（`gspo` 分支），不含场景资产、MesaTask-10K USD 物体文件、轨迹数据与模型权重。
- **核对范围：** 已通读 `paper_text.txt` 全部 1,527 行（Abstract、Sections 1–7 正文、Table 1–2、Appendix A–G 含全部公式 Eq.(1)–(14)、Algorithm B.1/B.2/E.1、Table A.1–A.4/D.1/G.1 及 References）与官方仓库页面 `project_page.txt`（README 与目录列表）。
- **通读：** 已完成

## 原文核对笔记

- **核心动机：** 现有具身智能体环境在两方面与真实部署脱节：（1）依赖仿真器特权感知 API（全局物体清单、真值 3D 位姿、oracle 类别检索），真机无法对齐；（2）默认指令完整且开环执行，忽略用户初始信息不全时所需的主动提问消歧。
- **核心贡献：**
  1. **REAL 无特权探索与交互环境：** 基于 GRUtopia / Isaac Sim 构建，初始仅提供物理可获取的 3D 占位图与容器（Receptacle）语义列表，暴露 11 个无特权 MCP 工具（`nav_to`、`walk_around`、`show_object_by_category`、`show_receptacles`、`gaze_at`、`pick`、`place`、`open`、`close`、`ask`、`finish`），并用 `gemini-3-flash` 驱动受信息边界约束的模拟用户（Progressive Disclosure + 辅助文本可见性摘要）。
  2. **POMDP 单步紧凑记忆与两阶段训练：** 为避免长程推理上下文爆炸，仅输入当前观测 $o_t^{\mathrm{env}}$、上一步动作 $a_{t-1}$ 与紧凑记忆 $h_{t-1}=(\sigma_{t-1}, q_{t-1})$（丢弃历史图像），输出推理链 $c_t$、更新记忆 $h_t$ 与工具调用 $a_t$。基于 `Qwen3-VL-8B-Instruct`（冻结 411M ViT 视觉编码器），先用 `gemini-3-pro` 离线标注的专家轨迹做行为克隆 SFT，再从 1-epoch SFT 检查点出发用 **GSPO**（序列级重要性比率与裁剪）结合 World Graph 差分密集奖励在 144 个并发 Isaac Sim MCP 端点上做在线 RL。
  3. **REAL-Bench（241 任务）与真机零样本迁移：** 涵盖 FDP（72）、FODP（56）、FDO（48）、SUL（65）四类任务；8B SFT+RL 模型在交互消歧子集 SUL 上达到 56.9% 成功率，超过 `gemini-3-pro-preview`（53.8%）和 `gpt-5`（52.3%）。在 ARX LIFT2 双臂移动机器人上通过同一套 MCP 接口与微调 $\pi_{0.5}$ 原语实现零样本迁移，60 个真机长程回合端到端成功率 78.3%。
- **关键局限：**
  - 冻结的 411M ViT 视觉编码器导致细粒度属性/颜色/几何辨识瓶颈（100 回合错误分析中 53 例失败里有 25 例为 Object Confusion；7,304 条 RL rollout 失败案例中 94.3% 属于抓错物体而非流程错误）。
  - 2-epoch SFT 出现行为克隆过拟合（FODP 从 30.4% 降至 28.6%），GSPO 可恢复探索与交互能力，但无法直接更新冻结的视觉表示。
  - 任务局限于跨容器重排，缺少复杂时序/空间关系约束；容器按整件建模（如 `cabinet_1`）而未细分到层板部件；模拟用户偏好固定在当前场景已存在物体内。
- **对 wiki 的映射：** [REAL（论文实体页）](../../wiki/entities/paper-agentic-2607-13653.md)
