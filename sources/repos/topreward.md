# TOPReward/TOPReward（官方代码）

> 来源归档（repo）

- **标题：** TOPReward — Token Probabilities as Hidden Zero-Shot Rewards for Robotics
- **代码：** <https://github.com/TOPReward/TOPReward>
- **类型：** research-code / reward inference & evaluation
- **License：** MIT（NOTICE：大量适配自 OpenGVL；视频工具改编自 LeRobot Apache-2.0）
- **主页：** <https://topreward.github.io/webpage/>
- **论文：** <https://arxiv.org/abs/2602.19313>
- **首次入库：** 2026-07-27

## 一句话摘要

用 Hydra 配置跑 **TOPReward**（指令匹配 token log-likelihood）与对照 **GVL**（打乱帧上生成完成百分比）的视频奖励推理；支持多 VLM 客户端、HF/本地视频数据加载，以及 OXE 等数据集配置。

## 公开资源

| 资源 | URL |
|------|-----|
| 仓库 | <https://github.com/TOPReward/TOPReward> |
| 项目页 | <https://topreward.github.io/webpage/> |
| arXiv | <https://arxiv.org/abs/2602.19313> |
| 上游对照实现 | <https://github.com/budzianowski/opengvl>（OpenGVL / GVL） |
| ManiRewardBench HF | <https://huggingface.co/datasets/ajyanggg/manirewardbench_lerobot> 等 |

## 运行入口（README）

| 入口 | 说明 |
|------|------|
| `uv run python3 -m topreward.scripts.predict --config-dir configs/experiments --config-name predict_topreward model=qwen` | TOPReward 推理 |
| 同上 `predict_gvl` | GVL 对照 |
| `topreward/scripts/run_predict.sh` | 单脚本封装 + Hydra overrides |
| `configs/model/*.yaml` | Qwen / Gemini / Molmo2 / OpenAI 等客户端 |
| `configs/dataset/*.yaml` | OXE 子集与 `local_video` 等 |

**最短路径：** `uv sync` → 配置 `.env`（API / HF token）→ `predict_topreward` → 读 `outputs/` 下预测与指标。

## 对 wiki 的映射

- [`wiki/entities/paper-topreward.md`](../../wiki/entities/paper-topreward.md)
- [`sources/papers/topreward_arxiv_2602_19313.md`](../papers/topreward_arxiv_2602_19313.md)
- [`sources/sites/topreward-github-io.md`](../sites/topreward-github-io.md)
