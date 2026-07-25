# open-dreamer（reactor-team · 推理）

> 来源归档

- **标题：** Open Dreamer Inference
- **类型：** repo
- **来源：** reactor-team（GitHub 组织；赞助/runtime 方）
- **链接：** https://github.com/reactor-team/open-dreamer
- **权重：** https://huggingface.co/reactor-team/open-dreamer
- **训练仓：** https://github.com/next-state/open-dreamer
- **项目页：** https://next-state.github.io/open-dreamer/
- **星标（截至 2026-07-25）：** ~0（独立推理仓；流量主要在训练仓与 demo）
- **最近推送：** 2026-07-25
- **主要语言：** Python（JAX）
- **分类：** 世界模型推理 / rollout harness
- **入库日期：** 2026-07-25
- **一句话说明：** Open Dreamer 的本地 rollout 脚本：给定 MP4 + Minecraft/VPT 动作序列，编码上下文帧并在潜空间生成后续帧。
- **沉淀到 wiki：** 是 → [`wiki/entities/open-dreamer.md`](../../wiki/entities/open-dreamer.md)（与训练仓合并为同一实体）
- **训练仓归档：** [`sources/repos/open-dreamer.md`](open-dreamer.md)

---

## README 要点（编译自上游）

- 自包含 rollout harness；需 CUDA 可见的 JAX GPU。
- 安装：`uv sync`；样例数据：`download_vpt_sample.py`（OpenAI VPT contractor 索引中的配对 mp4/jsonl）。
- 入口：`inference.py --checkpoint_path ... --input_mp4 ... --actions_path ... --output_mp4 ...`（常用 `--use_ema`）。
- 输入约束：动作可为 JSON 数组或 JSONL（VPT `mouse`/`keyboard`）；视频 RGB `368×640`（或 `360×640` 零填充）。

## 开源状态

- **已开源**：推理脚本与依赖锁定；检查点托管在同组织 HF `reactor-team/open-dreamer`。
- 与训练仓分工：本仓不做 tokenizer/dynamics 训练。

## 对 wiki 的映射

- 实体页：[`wiki/entities/open-dreamer.md`](../../wiki/entities/open-dreamer.md)
- 训练仓：[`sources/repos/open-dreamer.md`](open-dreamer.md)
