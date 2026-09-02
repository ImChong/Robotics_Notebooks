# CounterDrive（Hugging Face 数据集）

> 来源归档（dataset）

- **标题：** CounterDrive
- **类型：** dataset / counterfactual-driving / world-action-models
- **Hugging Face：** <https://huggingface.co/datasets/COWARobot/CounterDrive>
- **组织：** COWARobot（酷哇科技）
- **论文：** <https://arxiv.org/abs/2608.20430>
- **项目页：** <https://cowarobot-ai.github.io/RISE/>
- **许可：** `cc-by-nc-nd-4.0`（非商业、禁止再分发衍生；以 HF 卡片为准）
- **入库日期：** 2026-09-02
- **一句话说明：** 在选定 NAVSIM / nuScenes 事实上生成并人工核验的 10 秒反事实驾驶视频，为 RISE 的未来预测与局部风险监督提供多样结局。

## 访问要点（截至 2026-09-02）

| 项 | 内容 |
|----|------|
| 体积 | **31.5 GB**（卡片 Total file size） |
| 文件 | `ConterDrive-NAVSIM.tar` **20.2 GB**、`ConterDrive-nuScenes.tar` **11.3 GB**（文件名拼写为 ConterDrive，缺 u） |
| Dataset card | README **几乎为空**（33 B）；viewer 因 split 解析失败不可用 |
| GitHub README | 仍写 Counterfactual 数据未发布 → **以 HF 实际 tar 为准** |
| 用途 | 论文：接受片段监督 Predictor；核实配对做 Risk Profile 排序，**不**提供可靠他车几何故不用标量 \(\mathcal{E}_{\mathrm{risk}}\) 当反事实目标 |

## 规模（论文 / 项目页口径）

| 子集 | train | test |
|------|-------|------|
| NAVSIM | 5,013 | 1,000 |
| nuScenes | 2,432 | 511 |

配对只覆盖所选源场景，**不是**对完整 NAVSIM 训练集的一一覆盖（论文 Limitations）。

## 关联资料

- 论文：[`sources/papers/rise_adaptive_imagination_arxiv_2608_20430.md`](../papers/rise_adaptive_imagination_arxiv_2608_20430.md)
- 项目页：[`sources/sites/cowarobot-ai-github-io-rise.md`](../sites/cowarobot-ai-github-io-rise.md)
- 代码：[`sources/repos/coowai-rise.md`](../repos/coowai-rise.md)
- Wiki：[`wiki/entities/paper-rise-adaptive-imagination-wam.md`](../../wiki/entities/paper-rise-adaptive-imagination-wam.md)
