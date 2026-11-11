# LIFE / RoboBeats / SSS-Bench — 官方源码核验

> 来源归档（repo / 可运行原始源码）

- **标题：** LIFE — Learning to Live 官方实现
- **类型：** repo
- **仓库：** https://github.com/AgibotTech/LIFE
- **论文：** https://arxiv.org/abs/2611.00180v2
- **项目页：** https://agibotgeneral.github.io/life-site/
- **权重：** https://huggingface.co/superbigsaw/RoboBeats-0.8B 、 https://huggingface.co/superbigsaw/RoboBeats-2B 、 https://huggingface.co/superbigsaw/RoboBeats-4B
- **入库日期：** 2026-11-11
- **核对范围：** 官方根 README、robobeats/README.md、robobeats/gate.py（Gate._text / _chunk / stream / select）、sss_bench/README.md、sss_bench/reproduce/pipeline.py（stage_gate / stage_answer / main）、sss_bench/reproduce/agent.py（packs / write_memory / review_frames / read）；未覆盖所有源码 / 测试，也未实际执行 CUDA 推理或评测。
- **通读：** 已完成
- **一句话说明：** 官方公开的是 LoRA 门控模型、离线与实时推理入口及 SSS-Bench 基准 / 复现流水线，不应误写为论文中整个 Plastic Memory + EDM 自主演化栈的完整代码。

## 源码逐段对应

| 文件 / 函数 | 实际代码行为 |
|---|---|
| [robobeats/gate.py](https://github.com/AgibotTech/LIFE/blob/main/robobeats/gate.py) — \`Gate.__init__\` | 读取 Qwen3.5 基座 / PEFT 适配器；定位 false、true 的单 token id；离线实现设置 \`use_cache=False\` |
| \`Gate._text\` / \`Gate._chunk\` | 构造条件、时间戳、图像与标记；查找标记末 token 位置，一次前向计算多个标记的 2 类 softmax |
| \`Gate.stream\` | \`CHUNK=100\`、\`OVERLAP=10\`、\`STRIDE=90\`；后续 chunk 前 10 帧只做上下文，不重复打分 |
| \`select\` | \`p>0.5\` 开事件、\`p<0.3\` 关事件；保留开事件前 2 秒到关闭时刻；进行中的事件保留至截断 |
| [sss_bench/reproduce/pipeline.py](https://github.com/AgibotTech/LIFE/blob/main/sss_bench/reproduce/pipeline.py) — \`stage_gate\` | 按视频与观测起点分组，缓存 1 fps 帧与门控概率 |
| \`stage_answer\` | **逐题**截取 \`tick <= query_time_s\`；按照该题截断结果选帧并生成对应 memory cache，再调用 reader；写 JSONL |
| [sss_bench/reproduce/agent.py](https://github.com/AgibotTech/LIFE/blob/main/sss_bench/reproduce/agent.py) — \`packs\` / \`write_memory\` | 每包最多 60 帧、跨度小于 60 秒，用不含问题的 27B writer 写带原始时间戳的记忆；失败时重写或拆包 |
| \`read\` / \`review_frames\` | \`memoryonly\` 仅看记忆；\`hard10\` 最多向观察前缀发起 **一次**回看，且最多取 10 帧；JSON schema 与边界检查阻止读取提问后的帧 |

## 最小复现命令（以官方 README 为准）

\`\`\`bash
git clone https://github.com/AgibotTech/LIFE.git && cd LIFE
pip install -r requirements.txt
python -m robobeats my_video.mp4 --gate-model superbigsaw/RoboBeats-2B --out probs.json
# 若有 SSS-Bench 原始视频、27B 模型及多 GPU：
SSS_DATA_ROOT=/path/to/videos LLM_MODEL=/path/to/Qwen3.8-27B GATE=2B bash sss_bench/scripts/run_robobeats.sh
# 实时门控服务：
python -m robobeats.serve --gate-model superbigsaw/RoboBeats-2B
\`\`\`

**依赖注意：** 主 README 指定的 PyTorch CUDA 13 构建需较新的 NVIDIA 驱动；CUDA 12.x 需要按 README 替换 wheel，不能保证任意机器直接运行。

## 开放范围、许可与反例

- **已公开**：推理源代码、0.8B/2B/4B LoRA adapter、基准定义 / evaluator / reproduction scripts。README 2026-11-10 新增实时缓存版 \`LiveGate\`；**不要把离线 \`Gate._chunk\` 误称为使用 KV cache 的实时版本**。
- **未在本次核实范围内确认完整开源**：Plastic Memory 全部长期运行服务、EDM 训练 / 持续自改进的端到端可运行闭环。
- **许可**：官方 README 说明代码 Apache-2.0；SSS-Bench data / rubrics CC BY-NC 4.0；原始视频遵循来源各自许可证，不重新分发。
- 仅静态阅读代码；**未实测**速度、精度、显存或真实机器人闭环。
- **公开 benchmark 题目**并不等于视频也可以直接下载；需自行准备原始视频并按许可使用。

## 对 wiki 的映射

- [LIFE 项目实体](../../wiki/entities/paper-life-learning-to-live.md)
- [论文来源](../papers/life_arxiv_2611_00180_v2.md)
- [项目页来源](../sites/life-project.md)
