# TOPReward 项目页（topreward.github.io/webpage）

> 来源归档（site / project-page）

- **标题：** TOPReward: Token Probabilities as Hidden Zero-Shot Rewards for Robotics
- **类型：** site / project-page
- **官方入口：** <https://topreward.github.io/webpage/>
- **论文：** <https://arxiv.org/abs/2602.19313>
- **代码：** <https://github.com/TOPReward/TOPReward>
- **机构：** 华盛顿大学（UW）；艾伦人工智能研究所（AI2）；亚马逊（Amazon）；北卡罗来纳大学教堂山分校（UNC–Chapel Hill）
- **入库日期：** 2026-07-27
- **一句话说明：** 官方项目页：零样本进度估计演示、OXE / ManiRewardBench 数字、成功检测与 TOP-AWR 真机部署视频；页头提供 arXiv 与 Code 链接。

## 页面公开信息（检索自 2026-07-27）

| 资源 | URL |
|------|-----|
| 项目页 | <https://topreward.github.io/webpage/> |
| arXiv PDF | <https://arxiv.org/pdf/2602.19313> |
| Code | <https://github.com/TOPReward/TOPReward> |
| ManiRewardBench（HF 子集示例） | <https://huggingface.co/datasets/ajyanggg/manirewardbench_lerobot> 等 |

## 开源核查（步骤 2.5）

- 项目页头部 **明确列出** arXiv 与 GitHub Code 按钮。
- 代码仓 MIT，含可运行 `predict_topreward` / `predict_gvl`（Hydra + `uv`）。
- ManiRewardBench 轨迹子集已在 Hugging Face 公开（`ajyanggg/manirewardbench_*`）。
- **结论：已开源**（推理评测代码 + 基准数据）；无单独训练的 TOPReward 权重（方法本身不训练 reward 模型）。

## Highlights（项目页）

- Zero-shot progress estimation：VLM token log-likelihood 作稠密时序奖励（无需标定训练）。
- OXE Mean VOC 0.857（Qwen3-VL）；ManiRewardBench Mean VOC 约 0.945。
- 下游：成功检测；advantage-weighted BC（TOP-AWR）。
- 兼容开源视频 VLM（Qwen3-VL-8B 等）。

## 对 wiki 的映射

- [`wiki/entities/paper-topreward.md`](../../wiki/entities/paper-topreward.md)
- [`sources/papers/topreward_arxiv_2602_19313.md`](../papers/topreward_arxiv_2602_19313.md)
- [`sources/repos/topreward.md`](../repos/topreward.md)
- [`wiki/concepts/progress-reward-modeling.md`](../../wiki/concepts/progress-reward-modeling.md)
