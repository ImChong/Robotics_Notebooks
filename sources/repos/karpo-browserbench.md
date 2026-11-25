# Karpo BrowserBench：官方浏览器决策评测与打分代码归档

- **标题：** Karpo BrowserBench (KBB): A Browser Decision Benchmark on Real Websites
- **类型：** repo / 浏览器候选动作离线评测
- **发布方：** Karpo Research / MachinePulse
- **代码：** https://github.com/machinepulse-ai/karpo-browserbench
- **评测说明：** https://karpo.ai/benchmarks/browserbench
- **数据集：** https://huggingface.co/datasets/MachinePulseAI/karpo-browserbench
- **核查版本：** KBB v1.0；官方网页 2026-11-17 更新；2026-11-21 核对
- **核对范围：** 通读官方 README、网站方法说明、HF 数据卡及许可声明；阅读 `run.py` 的 `main/body/post/parse` 与 `score.py` 的 `load/grade/instructions/score` 主要执行路径，查看 `tests/` 20-step toy set 说明。没有获取受限数据或执行大型评测。
- **通读：** 已完成
- **代码许可：** Apache-2.0；**公开数据集独立评测限定许可**，不得用于训练、蒸馏、少样本检索或再分发。
- **开放状态：** 评测代码已开源；仅 35% public split 需同意条款后获取，65% private split 由组织方持有。
- **沉淀到 wiki：** [Karpo BrowserBench](../../wiki/entities/karpo-browserbench.md)

## 官方协议与规模

- 官网报告来自 **584 个真实消费网站、3,026 条指令、23,075 个页面动作步骤**。
- 浏览器真实录制后固定轨迹离线回放：模型看到目标、历史、编号元素树及大部分步骤的截图，选一个动作/终止状态；**下一页永远用录制页**，因此不能测错误动作后的自主恢复。
- 同一页接收多个合理 `accepted` 选项，错误程度分 `worse`、`wrong`、`overstep` 等；`DONE`、`NEEDS_APPROVAL` 等结束选择也计入。
- public split 的网站家族与 private split 不重叠；官网与数据卡给出 public 35% 约 8,075 步，但实际发布文件为 **7,557 步 / 1,017 指令**（隐私清洗后）。不得将未发布的步骤视为可下载。
- 评价 completion（指令所有步骤都对）、step accuracy、每千步 stop/missedAsk/overstep、validity、延迟分位与成本；主分由 completion 乘以超过预算时的成本/延迟折扣，具体预算 v1.0：每指令 $0.0063、每步骤 P50 444ms。
- 私有榜排名只用 private split；public 结果须单独标注版本和分区。

## 可运行代码入口

```bash
python3 run.py kbb-public runs/model --model <model-id> --base-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY --limit 20
python3 score.py kbb-public runs/model --usd-per-step 0.00028
```

- `run.py` 通过兼容 Chat Completions 的 JSON-schema 请求写每步 `runs/model/<decision>.json`；网络错误可恢复，拒答/格式无效计错。
- `score.py` 加载 `manifest.jsonl`、`key.jsonl`、各 `pages/*.json` 和答题文件，校验 offered actions → 按任务路径聚合 → 以网站家族 bootstrap 计算置信区间 → JSON 输出。
- 数据许可证明确**不得公开数据样本、截图、答案行**，也不允许训练使用；仓库自身测试用 toy fixture 可用于理解打分逻辑。

## 相关来源

- [BrowserBench 官方方法与榜单](https://karpo.ai/benchmarks/browserbench)
- [公开集模型卡与授权](https://huggingface.co/datasets/MachinePulseAI/karpo-browserbench)
- [浏览器代理源码](./karpo-browser.md)

