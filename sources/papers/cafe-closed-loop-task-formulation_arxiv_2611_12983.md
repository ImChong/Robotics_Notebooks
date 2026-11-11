# CAFE：无实例指令的闭环自主任务构想与执行（arXiv:2611.12983）

> 来源归档（paper）

- **标题：** CAFE: Closed-Loop Autonomous Task Formulation and Execution without Instance-Specific Instructions
- **类型：** paper
- **作者 / 机构：** Mingjia Chen, Xuewen Zhao, Philip Dames, Qi Mao, Fei Xie, Jun Chen（通讯）；南京师范大学电气与自动化工程学院、美国天普大学机械工程系
- **arXiv：** <https://arxiv.org/abs/2611.12983>（v1，2026-11-08，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2611.12983v1>
- **代码：** https://github.com/submissionforpaper/CAFE-Closed-Loop-Autonomous-Task-Formulation-and-Execution-without-Instance-Specific-Instructions
- **项目页：** <https://submissionforpaper.github.io/CAFE-Closed-Loop-Autonomous-Task-Formulation-and-Execution-without-Instance-Specific-Instructions/>
- **入库日期：** 2026-11-11
- **一句话说明：** 在预设的清洁 / 安全 / 收纳整理三类任务下，用在线语义记忆 + 规则触发的 Trigger-LLM + Draft-Complete-Audit 三段式 LLM 规划，让机器人无需实例级指令自己提出并执行任务，执行结果回写记忆。
- **核对范围：** 论文 v1 PDF 正文全文（摘要、I 引言、II 相关工作、III 方法、IV 实验含 Table I-III 与 Fig. 1-5 的文字、V 讨论与结论、参考文献列表）；图中的曲线 / 雷达图仅读到文字标注，Table III 分栏数值由 PDF 抽文本还原（个别列顺序按行内文字交叉核对）。未读：论文无附录；项目页未打开；仓库仅静态读到根 README（见下），源码与提示词文件未读。
- **通读：** 已完成

## 开源状态

- 仓库 README 自述为「Anonymous materials for review」，实现位于 `CAFE/`，入口 `CAFE/aithor2/main_with_depth.py`，需自行配置 API 凭据；网站素材（图、字体、PDF、结果数据、项目视频）在 `assets/`。以上来自 README 静态阅读，未运行、未核验仓库其余内容。
- 项目页：未核实（未打开）。

## 核心摘录

1. **设定（摘要 / I）：** 在预定义的清洁、安全、收纳整理三类别下，不给实例级目标、目标物清单或动作序列。
2. **记忆（III-A，式 1）：** M_t = {O_t, Z_t, X_t}（物体及状态 / 功能区 / 时间戳、置信度与执行状态）。
3. **触发（III-B，式 2-3）：** 每个 heartbeat，Trigger-LLM 对比当前记忆与冻结参考 M̂_t；去重后估计的不同候选数 K̂_t ≥ 4，或相对参考新增物体 ≥ 20 个，则 Trigger 并更新参考；失败纠错绕过该门。
4. **三段式（III-C，式 4-6）：** Draft 提出 ⟨动作, 目标, 区域, 前置, 优先级, 几何⟩；Complete 单次前向补全目标 / 空间关联 / 几何并加原子步骤，不得新增目标；Audit 检查语义一致、物体可得、安全与前置连贯，修复或拒绝。
5. **执行反馈（III-D，式 7）：** 执行器按审计后顺序下发；目标缺失 / 前置违反 / 效果异常则中止该序列并请求纠错；结果写回 D_{t+1}、M_{t+1}。
6. **评测（IV）：** 12 个 AI2-THOR 场景，每任务重复 10 次；三档复杂度；Tri-LLM 与 Single-LLM 用 DeepSeek-V3 / Qwen3-Max / Gemini-2.5-Pro，对照实验用 Qwen3-Max；不训练不微调。指标 UC、sUC、CUR、SR，GCR = UC × SR。
7. **主结果（Table II，Qwen3-Max，Complex）：** Tri-LLM UC 0.922 / CUR 0.815 / sUC 0.892 / SR 0.856 / GCR 0.789；Single-LLM GCR 0.446（差 34.3 点）；ProgPrompt（适配）GCR 0.464（差 32.5 点）。ProgPrompt 在 Compound CUR 上高于 Tri-LLM（0.757 对 0.725）。
8. **消融（Table III，Qwen3-Max）：** 去 Complete，Complex sUC 0.892→0.194，UC 仍 0.905；去 Audit，Complex CUR 0.815→0.515；去功能区，Complex sUC 降到 0.375；反馈：Compound GCR 连续反馈 0.889 / 仅失败 0.802 / 无 0.684；选择性触发（5 s）相对穷举（5 s）Complex sUC 0.892 对 0.618，完整规划轮数 6.5→3.1；无示例（zero-shot）Complex Tri-LLM sUC 0.634 对 Single-LLM 0.392。
9. **真机（IV-E，Fig. 5）：** LimoCobot 上三段定性轨迹：存放掉落钥匙并丢垃圾；优先处理水渍并清洁后收布、丢两件垃圾、收胶带；抓取第二件垃圾失败后反馈识别未完成并重试成功。无真机量化指标。
10. **自述范围（V）：** 仅限预定义任务类别与已有的机器人技能；未来工作为泛化到更多场景与多机器人协作。

## 对 wiki 的映射

- [paper-cafe-closed-loop-task-formulation](../../wiki/entities/paper-cafe-closed-loop-task-formulation.md)
