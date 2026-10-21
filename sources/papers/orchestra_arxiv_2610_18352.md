# Let the Mind Conceive, Let the Hand Achieve（Orchestra）：Following Multimodal Instructions for Long-Horizon Robotic Manipulation（arXiv:2610.18352）

> 来源归档（ingest）

- **标题：** Let the Mind Conceive, Let the Hand Achieve: Following Multimodal Instructions for Long-Horizon Robotic Manipulation
- **arXiv：** <https://arxiv.org/abs/2610.18352>（PDF：<https://arxiv.org/pdf/2610.18352>；HTML：<https://arxiv.org/html/2610.18352v1>）
- **代码（官方仓库，已核查）：** <https://github.com/youpengwen/Orchestra>（归档见 [orchestra](../repos/orchestra.md)）
- **项目页：** <https://youpengwen.github.io/Orchestra>（归档见 [orchestra-project](../sites/orchestra-project.md)）
- **作者：** Youpeng Wen、Gang Wang、Tianyu Huang、Weiwei Zhang、Zhijian Huang、Wei Chen、Xieyuanli Chen、Yunhui Liu；单位：论文 HTML 与项目页均未在抓取内容中列出（未核实）
- **入库日期：** 2026-10-17
- **来源：** 具身智能小站 2026-10-17 公众号摘要 + arXiv HTML（前 100k 字符，附录 E 后未读）
- **一句话说明：** Orchestra 把指令分解、空间落地、进度评估与低层动作生成分离，共享子任务上下文；并提出 RMBench-Compose 测试短轨迹训练后的长序列组合。

## 开源状态（步骤 2.5）

- **项目页核查（2026-10-17）：** 页面链接 GitHub 仓库；Hugging Face 与 arXiv 标 "Coming soon"（权重 / 数据是否发布：未列）。
- **仓库核查（2026-10-17）：** 公开仓库（`web` 分支，6 次提交），含 `embodied_agent/`、`policy/modified_pi05/`、`annotation/`、`scripts/`、`tests/`、`pyproject.toml`、README 结果表与四段真机演示；README 抓取内容未显示安装 / 运行命令，许可证未显示。
- **结论：** **已发布仓库，复现入口未核实**（按「部分开源」处理）；权重 / 数据 HF 页标 coming soon。

## 摘录 1：方法

- 规划智能体：默认 DeepSeek-V4.1-Flash，可替换 Qwen3.7-Plus；把多模态指令（语言 + 参考图、纯语言、点 / 框指代）拆成有序子任务与实体绑定。
- 空间落地：MolmoPoint-8B 在头部相机图像中指向目标物体，SAM3 分割；子任务边界初始化、重规划时跟踪。
- 工作记忆：已完成子任务记录、有界多相机视觉历史、当前子任务索引；子任务转换时重置视觉历史与落地，任务记录保留。
- 动作策略：改造的 π0.5，输入子任务文本、空间提示（框 / 点 / 轮廓）、6 帧（stride 3）视觉窗口，增加时间注意力而不增加送入语言模型的视觉 token 数。
- 进度评估器：单独训练的 Gemma-300M，预测 11 类子任务内进度，终止类触发切换。

**对 wiki 的映射：** 核心原理与流程总览。

## 摘录 2：结果

- RMBench 九任务平均成功率 68.22%（Mem-0 42.00%，HarnessVLA 26.78%）。
- RMBench-Compose（由 Press Button、Pick Up Object、Cover and Uncover Block、Swap Block、Place Block on Mat 构成）：ASL 7.35，HarnessVLA 2.986（2.46 倍）；π0.5 单独 SSR 47.36% / ASL 0.932。
- 消融 ASL：去任务编排 1.502，去短时视觉记忆 2.500，去空间落地 5.814；评估器 7.350 对状态型 oracle 转换 7.374。
- 真机四任务各 20 条指令：ASL 2.20 → 2.75（+25.0%），子任务成功率 72.77% → 82.09%。

**对 wiki 的映射：** 实验与评测、结论。

## 摘录 3：局限

任务依赖明显（Pick Up Object 原子 SR 81% 但 ASL 仅 2.11）；仿真使用参考落地（真值掩码），需更长真机序列加预测落地评测；聚合分数无法区分操作失败与管理失败；仅测两个规划智能体；学习型评估器与 oracle 接近仅为聚合结果。

## 建议 wiki 动作

- 新建 [`wiki/entities/paper-orchestra.md`](../../wiki/entities/paper-orchestra.md)。
