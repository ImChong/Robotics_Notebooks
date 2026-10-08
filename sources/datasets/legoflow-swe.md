# LegoFlow-SWE（Hugging Face 数据集）

> 来源归档（dataset / LegoX 官方发布）

- **标题：** LegoFlow-SWE
- **类型：** software-engineering tasks + coding-agent trajectories
- **Hugging Face：** <https://huggingface.co/datasets/Lego-X/LegoFlow-SWE>
- **来源项目：** [LegoFlow](https://github.com/LegoX/LegoFlow)
- **项目博客：** <https://www.legox.net/blog/legoflow/>
- **组织：** LegoX
- **核验日期：** 2026-10-08
- **一句话说明：** 从逾 12M 个 GitHub PR 候选筛出的 5,000 个 verified Harbor 软件工程任务，附两种提示版本及 GLM-5.2 在 OpenHands / OpenCode 上生成的 9,767 条轨迹。

## 数据概览

| 项目 | 公开内容 |
|------|----------|
| 任务 | 5,000 个 Harbor task；原始提示与 anti-hack 提示两目录，对应相同 task IDs |
| 来源筛选 | 超过 700K repositories、约 12M PR candidates → 5,000 verified tasks |
| 覆盖 | 8 种编程语言、20 类 task tags |
| OpenHands 轨迹 | GLM-5.2 × OpenHands SDK 1.33.0：4,753 条，1,350 条 reward=1 |
| OpenCode 轨迹 | GLM-5.2 × OpenCode 1.18.7：5,014 条，1,430 条 reward=1 |
| 体量 | Hugging Face 页面显示约 9.96 GB；轨迹数据行数 9,767 |

任务遵循 Harbor 目录结构，含 instruction.md、task.toml、环境 Dockerfile、bug patch、reference fix/solve script 和测试。任务以 source repository/issue/PR 标识组织；task.toml 保存难度、类别、tags 与资源限制。

轨迹 JSONL 中包含版本、reward、metadata、工具定义及 messages。官方说明 reward=1 表示 verifier 成功；reward=0 既可能是测试失败，也可能是缺失 verifier reward，因此不能简单解释为模型完全失败或 hack。轨迹发布未过滤，训练使用的约 1K 样本池另行抽样。

## 评测与防泄漏边界

官方 no-hack 评测组合三项措施：anti-hack prompt、在 agent 开始前移除原始 Git 历史/远程信息，以及限制网络访问（保留模型推理所需连接）。tasks-anti-hack/ 只在提示里加入约束；runner 仍需自行执行 Git 清理和网络隔离。公开 rollouts 的生成设置与此 no-hack 评测设置不同。

## 许可证状态

Hugging Face 页面公开可浏览和下载数据；本次可见的 dataset card 未明确展示独立数据许可证。仓库代码标记 Apache-2.0 不自动等于数据也采用该许可；再分发或商业使用前应核对数据卡和来源 PR/代码的适用条款。

## 训练结果（发布方报告）

在同一 OpenHands SDK no-hack 配方下，每个来源约 1K 轨迹训练 Qwen3.5-35B-A3B-Base；HF 卡报告：

| 基准 | LegoFlow-SWE | Qwen3.5 Instruct | 差值 |
|------|-------------:|-----------------:|-----:|
| SWE-bench Verified | 70.2% | 63.4% | +6.8 pp |
| SWE-bench Pro | 48.8% | 38.2% | +10.6 pp |
| SWE-bench Multilingual | 57.0% | 51.7% | +5.3 pp |

HF 卡列出 Verified/Pro/Multilingual 分别 500/731/300 道题，限制了 agent scaffold、turn budget、context 等配置。分数是 LegoX 报告结果，跨 scaffold 或不同任务快照的数字不可直接横向比较。

## 对 wiki 的映射

- 统一项目节点：[LegoFlow](../../wiki/entities/legoflow.md)（流程、代码与本数据集）
- 代码：[legox-legoflow.md](../repos/legox-legoflow.md)
- 项目页：[legoflow-project.md](../sites/legoflow-project.md)
- 博客：[legoflow_2026-09-18.md](../blogs/legoflow_2026-09-18.md)
