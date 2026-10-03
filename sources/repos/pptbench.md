# PPTBench（Einsia）

> 来源归档（repo）

- **标题：** PPTBench
- **类型：** repo
- **链接：** <https://github.com/Einsia/PPTBench>
- **关联论文：** [pptbench_arxiv_2609_29718.md](../papers/pptbench_arxiv_2609_29718.md)
- **项目页：** <https://lab.einsia.ai/pptbench>
- **入库日期：** 2026-09-30
- **一句话说明：** 500 任务可编辑幻灯片重建 benchmark + Agentic Judge 与本地 materializer。

## 开源状态

- **已开源**：评测协议与工具链在仓内；参考图像素按 license 不随仓再分发。

## 对 wiki 的映射

- [paper-pptbench](../../wiki/entities/paper-pptbench.md)

## 运行入口核查（2026-10-03）

- `pptbench-materialize`：按固定 manifest 下载并验证任务素材。
- `pptbench-eval harness-run`：调用 coding agent，验证并渲染 `reconstruction.pptx`。
- `pptbench-vlm-judge`：语义门、渲染门与细节 findings。
- `pptbench-vlm-rank` 与 `pptbench-vlm-consensus-rank`：轮内计分与三轮共识。
- 本次为 README 入口核查，未执行完整评测；历史论文指标与当前 README 排行榜快照应分开阅读。
