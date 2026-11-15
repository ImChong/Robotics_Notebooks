# RoboBench：原文通读与评测实现复核（2026-11-15）

- **类型：** paper
- **原文：** <https://arxiv.org/html/2510.17801>
- **版本：** 2026-11-15 获取的 HTML 快照；18 模型、ECCV 2026 版结果。
- **项目页：** <https://robo-bench.github.io/>
- **代码：** <https://github.com/yulin-luo/RoboBench>，所读树 `08314edc65800086656a0ff93332b023180eaa35`。
- **核对日期：** 2026-11-15
- **核对范围：** 正文 §1–6、附录 A–G（含全部数据构造及 Q1/Q2/Q3 评测提示），全部表格与图注；打开项目页资源区，通读官方 README 和 PROMPTS_AND_PIPELINE.md；通读 cli.py、dataset.py、builder.py、pipeline/node.py、executor.py、aggregator.py，核对 planning.py 的 PlanningEvaluator 全类。未运行模型 API、机器人或论文实验。
- **通读：** 已完成

本次复核单独存档，[历史摘录](robo_bench_arxiv_2510_17801.md)保留既有记录。

## 方法、规模与指标

- 五维、14 能力、25 任务、6,092 QA，对应 4,336 unique items。当前发布物有 32 个 manifest：两类任务进一步按材料/属性分片，不代表增加了 7 类新能力。MCQ 为 1,895 题，Planning 为 4,197 题；affordance 也以候选点/轨迹/底座位置的选择题评测，不是直接测机器人轨迹执行。
- Q1 给 judge 首帧、参考动作、人工依赖 DAG 和被测计划，分别输出 NodeCorrectness 与 TaskCompletion 两个 0–10 整数；正文 LongHorizon 为二者之和除以 20，当前代码再乘 100。它是结构化 MLLM 评审，不是物理引擎或真实机器人成功率。
- Q2 比较 skill/object/parameter：skill 精确匹配，对象或技能为零时参数记零；Q3 是 yes/no accuracy。不同题型分数不可直接解释成同一种控制能力。
- 数据质量控制包含人工交叉检查与基于已测模型投票的难度筛选，不能把该分布上的成绩视为任意未来机器人任务的保证。

## 统计与外推边界

- 迁移实验只有 8 个 VLM 骨干，按 VLM4VLA 全参数微调，输入单视角 RGB、不含本体状态。CALVIN ABC-D 训练 30k 步、测 1,000 条五子任务序列，指标是平均完成子任务数；LIBERO-Long 训练 50k 步、10 任务各 50 trials，指标是成功率。
- CALVIN 与 object-centric / scene-centric 分数的相关为 `r=0.884, p=0.0035` / `r=0.833, p=0.0103`。LIBERO 与 static+dynamic affordance 的 `r=0.677, p=0.0652` 未达到常用 0.05 阈值；不能概括为两个基准均已证明显著相关，更不能视作因果或必要条件。
- Q1 的 200 题 judge 鲁棒性实验中，v3 节点 ICC 为 0.948，综合 ICC 为 0.854；v2 综合 ICC 为 0.900。节点一致性提高不代表所有综合指标均最优。标注噪声试验只有 30 题，删参考节点会机械性地抬高完成分，中间模型排序可以交换。
- 人类结果为抽样 Basic Reference，不是理论上界；低层失败诊断与下游 VLA 的相关性很弱。隐式指令 CoT 的 100 题消融只测三类前沿模型，不能推断弱模型也一定无收益。

## 当前运行入口与实现边界

- 项目页实际链接 GitHub、HF 数据与结果。README 说明 MIT 代码、API 或可选本地模型栈；完整复现仍需下载图片、配置被测模型与独立 judge。
- `robobench --config ... inspect-data --metadata-only` 检查记录与提示构造但不检查图片是否存在；去掉该参数检查图片。`pipeline --dry-run` 只列节点，不等价于数据验证。
- `cmd_pipeline` 依次安排 LoadDatasetNode、BuildPromptsNode、RunInferenceNode、EvaluateNode；executor 顺序执行 setup/validate/run/teardown 并传递输出。dataset 从发布的 questions.json 与 system_prompt.json 构造提示；PromptBuilder 的 messages 使用问题和图片，原始答案另保留供后续评价。
- RunInferenceNode 调 AsyncModelClient.run_batch，保存 raw.json；EvaluateNode 按类型调用评测器。PlanningEvaluator 的 Q1 读取首帧与 dag.json，Q2 抽取/比较下一步，Q3 规范化 yes/no，分别保留 judge 输出和解析失败数。
- 当前 Q1/Q2 均值只使用解析成功项，Q3 空响应按错计；必须同时记录解析覆盖率。CLI 多轮汇总以 `eval_file.parent.parent.name` 为键，RunAggregator 只接收数字或顶层 accuracy，而 planning 返回嵌套结果；不能把默认聚合当作论文所有任务的已验证总表。`report` 子命令仍为 TODO。

## 对 wiki 的映射

- [RoboBench](../../wiki/entities/robo-bench.md)
- [具身评测基准选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md)
