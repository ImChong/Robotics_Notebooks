# NVlabs/Flora — 官方开源推理仓库核验

> 来源归档（repo / 官方代码）；2026-11-10 实际读取 README、AGENTS.md 和关键推理入口，不能把推理示例等同训练/生产服务。

- **仓库：** <https://github.com/NVlabs/Flora>（NVIDIA/Flora 会重定向至 NVlabs/Flora）
- **项目页：** <https://research.nvidia.com/labs/gear/flora/>
- **论文：** <https://arxiv.org/html/2610.31779v1>
- **模型：** <https://huggingface.co/collections/nvidia/flora>
- **代码开放：** **部分开源**。包括 TempSeg→Captioner 及三种 Rewarder 的示例推理与模型权重；**不含训练代码、生产推理编排和内部数据**。
- **许可证：** `LICENSE` 为 NVIDIA 非商业研究/评估许可；权重仓同类许可。不要据源码公开推断可商用。
- **核对范围：** 官方 `README.md`、`AGENTS.md`；`captioner/infer.py`、`captioner/batch_utils/{cli,api,pipeline,models,media}.py`；`rewarder/batch_utils/{terminal_reward,dense_reward,rubric_reward,engine,core}.py` 中实际 CLI、调用流程与输出约定；另确认 `tempseg/batch_utils/` 存在同一完整分段到描述流程。
- **通读：** 已完成

## 模块与入口

| 实际文件 | 关键调用或责任 |
|---|---|
| `captioner/infer.py` | `from batch_utils.cli import main`，CLI 程序唯一浅入口 |
| `captioner/batch_utils/cli.py` | `jobs()` 解析单视频/`--input-jsonl`；`main()` 创建 `DataEnginePipeline`，`--validate-only` 检查或调用 `annotate()`，写一视频一 JSON |
| `captioner/batch_utils/api.py` | `DataEnginePipeline._inputs()` 用 `media.probe/check_synchronized` 校验输入；`annotate()` 校验 CUDA、转入 `pipeline.annotate` 并检查各片段时间合法性 |
| `captioner/batch_utils/pipeline.py` | `annotate()` 先 `segment()` 后 `caption_single()` 或 `caption_three()`；`segment()` 中 `Session.load('tempseg')`、`FineSubtaskSegmenter.segment_async`；`Predictor.predict()` 按窗口取帧、构造输入、调用 `generate()`；按事件处理描述与溯源信息 |
| `captioner/batch_utils/models.py` | 模型权重/processor 会话加载与单模型 `generate()` |
| `rewarder/batch_utils/terminal_reward.py` | 载入任务/视频，抽取全集帧，用 vLLM 打 1–5 分、可选理由，输出 `score/probabilities/expected_score/progress` |
| `rewarder/batch_utils/dense_reward.py` | `core.dense_windows()` → `llm.generate()` → `core.parse_dense()` → `core.stitch_dense()` → 每帧/秒、片段与异常理由 JSONL；会报告 `invalid_clips` |
| `rewarder/batch_utils/rubric_reward.py` | rubric 输入验证、思维模式、多采样 `SamplingParams(n=--num-samples)`、`core.parse_rubric`、`core.medoid`，按步骤与总分输出 |
| `rewarder/batch_utils/engine.py` | `read_jobs`、`resolve_model` 下载 Hugging Face 权重、`load_processor`、`load_llm` 建立离线 vLLM、`write_record` JSONL |

## 实际运行链路

**标注：** `captioner/infer.py` → `cli.main` → `DataEnginePipeline.annotate` → 媒体校验 → `pipeline.annotate` → `segment`（模型加载/窗口预测/跨窗拼接）→ `caption_single` 或 `caption_three`（按事件补视觉上下文、Captioner 推理）→ 返回 `segments` 并写 JSON。前后模型不是一个联合网络，也不在其中调用奖励模型。

**奖励：** `python -m rewarder.batch_utils.{terminal_reward|dense_reward|rubric_reward}` → `engine.py` 解析输入/加载权重与 vLLM → 任务分支调用 `core.py` 视频拼图和 prompting → 解析/聚合评分 → 写 JSONL；三路入口独立、可选择性调用。Dense 采用后见窗口而不适合控制器在线奖励回路。

## 官方最小命令

```bash
# annotation 环境：CPU 输入验证；正式 GPU 推理另用 --output
python captioner/infer.py --video demo.mp4 --profile human --validate-only
python captioner/infer.py --video demo.mp4 --profile human --output demo.json

# robot 三路相机，要求与主相机时间戳同步
python captioner/infer.py --video main.mp4 --left-wrist left.mp4 --right-wrist right.mp4 --profile yam --output robot.json

# rewarder 的独立虚拟环境
python -m rewarder.batch_utils.terminal_reward --model nvidia/Flora-Rewarder --video episode.mp4 --task "Put the block in the silver bowl" --with-rationale --output outcome.jsonl
python -m rewarder.batch_utils.dense_reward --model nvidia/Flora-Rewarder --video left.mp4 top.mp4 right.mp4 --task "Put the cup on the tray." --output dense.jsonl
python -m rewarder.batch_utils.rubric_reward --model nvidia/Flora-Rewarder --jobs rubric_jobs.jsonl --output rubric.jsonl
```

## 输入输出和可运行限制

- 官方验证环境：Linux x86_64 / Python 3.12 / H100 80 GB，Rubric 默认两卡。Annotation 与 Rewarder 的 PyTorch 依赖需拆分环境；无法将 demo 的最低硬件当成可降配保证。
- 输入：annotation 单条 ≤120 秒、≤256 MiB、最大边 ≤4096 px、≥4 FPS；多机位需帧同步。`human` 一路相机，`yam`、`dex`、`tianji` 可用三视角。`--yam32` 仅符合三机位 YAM 条件时使用。
- Annotation 输出：`schema_version`、`duration_s`、`profile`、`segments:[{start_s,end_s,caption,status,segmentation_provenance}]`；段可重叠表示同时动作，`status != complete` 要当成缺标签而非有效描述。
- Outcome 输出 1–5 评分及概率，`progress` 为归一化期望分数、非成功概率。Dense 输出 `segments`、`per_second`、`invalid_clips` 等；Rubric 输出逐步骤 `scores` / `total` / `parsed_samples` 等。
- 首次需下载约 130 GB Annotation 权重、56 GB Rewarder 权重。官方 README 的样例 107 秒影片在一张 H100 首次执行耗时 23 分钟（含下载），**与论文生产吞吐无关**。
- 下载和运行均为用户自备环境操作：本次仅静态核验官方源码与可用 CLI，**未下载权重、未执行 H100 推理或测量吞吐**。

## 限制与安全

- 官方明确 **no training code** 和 **sample vanilla inference code rather than production inference system**，因此训练 GRPO/SFT 和 512 GB300 生产任务调度无法由仓库直接复现。
- Rewarder Dense 可使用事件之后的画面；若将其输出接入机器人执行控制，可能违反因果和安全假设。
- 研究用途许可，仅非商业研究和评估；官方禁止将输出作为机器人控制指令或唯一物理系统安全监测依据。
- 先在自有数据上验证 Rewarder 与人类标签一致程度，并审查 `invalid_clips`、未完成的 caption、解析样本不足情况。

## 对 wiki 的映射

- [FLORA 项目唯一实体](../../wiki/entities/flora-robotics-data-engine.md)
- [论文逐节核验](../papers/flora_nvidia_2026.md)
- [官方项目页核验](../sites/flora-nvidia.md)
