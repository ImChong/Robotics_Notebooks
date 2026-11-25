# Karpo Decision 27B Act：官方网页动作决策模型卡归档

- **标题：** Karpo Decision 27B Act
- **类型：** model / 专用 System 1 网页操作决策 checkpoint
- **发布方：** MachinePulse Pte. Ltd.
- **官方模型卡与权重：** https://huggingface.co/MachinePulseAI/karpo-decision-27B-act
- **对应浏览器：** https://github.com/machinepulse-ai/karpo-browser
- **对应推理代码：** https://github.com/machinepulse-ai/karpo-decision
- **核查版本：** 2026-11-21 可见官方模型卡
- **核对范围：** 通读模型卡 Results、How it works、System 1/2 分工、Action space、FILL/SELECT、训练、Use、Limitations、Licence；交叉核对 karpo-browser README 和官方推理仓既有归档。未下载 26B 级权重、未在 GPU 上推理、未复跑两套基准。
- **通读：** 已完成
- **开放状态：** 权重与推理框架均已公开；本卡未提供可独立验证的完整训练数据及端到端训练流水线。
- **许可：** 27B Act 模型卡标明 Apache-2.0；其推理代码仓亦为 Apache-2.0；应逐 checkpoint 核对，不能推断所有 Karpo 权重授权一致。
- **沉淀到 wiki：** [Karpo Decision 27B Act](../../wiki/entities/karpo-decision-27b-act.md)

## 机制

- 模型卡报告：Qwen3.8-27B 的多模态主干经 Karpo Decision 27B（未单独发布）后训练，再微调为网页动作模型；**26.1B BF16 参数 + 255×5,120 FP32 readout**。
- 输入：网页编号元素、截图、子任务目标、System 2 给定的可填写值、近期操作及其效果；也支持不发截图。输出：一轮中的选项概率与确定性选择，非自然语言生成。
- 每轮选择 `CLICK n` / `FILL n` / `PRESS_ENTER n` / `SELECT n` / 滚动、等待/后退/刷新等，以及七种终止状态。`FILL` 与 `SELECT` 可追问具体给定值；另有循环检测 `noul` 问题。
- 分工：System 2 负责拆目标、提供填写内容、跨任务记忆、最终阅读与判断；System 1 负责页内逐步动作和条件终止判断。

## 发布方报告的评测（不是本仓复现）

- **Browserbase Benchmark v2：** 与 DeepSeek V4.1 Flash 作为 System 2 组合后，38 个 core tasks 的成功率 **84%**（8 次均值）、每任务 **218 s / $0.096**；为发布方自行运行，不是该榜提交结果。
- **Karpo BrowserBench 1.0：** 模型卡列出 private split **35.5% 指令完成 / 78.5% 单步准确率**，all scored **35.3% / 78.3%**；同样是**自测、未进入榜单**。
- **延迟：** KBB 单请求中位 **931 ms** 为无推理加速、单请求逐次测量的口径，不等于机器人实时控制周期。

## 关键限制

- 只在给定选项中决策，不能自由打开 URL、产生文本、点坐标；训练偏向英语消费网站。
- KBB 为离线固定轨迹回放，不能证明偏离轨迹后自动恢复能力；数据中包含训练时见过的网站家族。
- 不可逆动作保护依赖模型选择 `NEEDS_APPROVAL`，浏览器**没有硬编码阻断**：正式部署需外部审批门控。
- **集成差异：** 模型卡注明 karpo-browser 当前客户端使用的 POST 路径与开源推理服务的 `/v1/systemone` 不同，**不能直接把浏览器指向该服务器**；需要明确的端点适配/检查更新。
- FP8、CUDA Graph 可选但需启动开销和硬件支持；模型卡报告双开启动约 7 分钟，不代表常规 BF16 启动。

