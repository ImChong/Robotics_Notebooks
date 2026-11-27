# Collab-Gen/MEMOBench（官方基准仓库归档）

- **仓库地址：** <https://github.com/Collab-Gen/MEMOBench>
- **对应论文：** `arXiv:2609.07047`（[MEMOBench 论文归档](../papers/paper-agentic-2609-07047.md)）
- **默认分支：** `main`
- **开源许可：** 仓库 `LICENSE` 与 `README.md` 标明 MEMOBench 本体采用 **CC BY-NC 4.0**（仅限非商业科研使用，GitHub API 识别为 `NOASSERTION`），`src/third_party/LIBERO/` 沿用 **MIT** 许可。（注：arXiv v1 论文 Ethics Statement 曾表述计划以 Apache-2.0 发布代码与 CC BY 4.0 发布数据集，实际以当前仓库 `LICENSE` 为准。）
- **配套数据集与资产：** [HuggingFace Datasets: SunSeaLucky/MEMOBench](https://huggingface.co/datasets/SunSeaLucky/MEMOBench)（`MEMOBench.zip` 示范轨迹与 `assets.zip` 物体/场景模型）。
- **核查边界：** 静态代码与文档核对（基于仓库目录采样与 README 说明，未在本地执行仿真或模型评测）。

## 已核对目录与可执行入口

根据仓库文件采样与 README 核对，核心结构如下：

- `src/eval.py`：主评测入口脚本。读取 `--task-dir` 下的任务配置，连接策略 WebSocket 服务端，执行仿真 rollout 并输出总体 SR、分操作过程指标（Storage / Update / Compression）、失败分类统计 JSON 与 rollout 视频。
  - 关键参数：`--task-dir`（如 `src/bddl/temporal`）、`--num-trials-per-task`（默认对比实验为 50）、`--memory-level`（0–2，选择不同粒度的记忆检查点注释）、`--host` / `--port`（默认 `0.0.0.0:8000`）、`--max-steps`、`--replan-steps`、`--seed`、`--video-out-path`、`--failure-out-path`。
- `src/bddl/{object,procedural,spatial,temporal}/*/task_config.bddl` 与 `task_config.meta.json`：30 个历史依赖任务的 BDDL 场景/目标规范及对应的多粒度语言、记忆操作标签（S/U/C）和仿真器可执行 checkpoint predicate 元数据。
- `src/get_object_pool.py`：辅助脚本，用于检视物体池并生成 `.scrapy/objects.json`。
- `src/third_party/LIBERO/`：内置（vendored）的 LIBERO / robosuite 仿真框架代码与基础资产目录（需安装 `pip install -e src/third_party/LIBERO`，并将 `assets.zip` 解压复制到 `src/third_party/LIBERO/libero/libero/assets/`）。
- `assets/teaser.png`、`LICENSE`、`README.md`：项目说明与许可文件。

## 策略服务接口与复现边界

- **通信协议：** `src/eval.py` 依赖 `openpi_client`（来自 `Physical-Intelligence/openpi`）通过 WebSocket 向外部策略服务（如 `openpi` 的 `scripts/serve_policy.py`）发送 `observation/image`、`observation/wrist_image`、`observation/state` 与 `prompt`，并接收 action chunk 执行。
- **参考视频上下文（Reference-video context）：** 固定提示视频（prompt videos）的关键帧采样与视觉编码（如 $\pi_{0.5}$ 的 SigLIP 编码器）在策略服务端离线预处理并缓存为视觉 token，初始化服务端时加载并拼接在 prompt token 之后；`src/eval.py` 本身不传入参考视频参数。
- **仓库边界说明：** 当前 `Collab-Gen/MEMOBench` 仓库主要提供基准任务定义、checkpoint 标注与在线仿真评测器；论文第 3.4 节所述三种记忆对齐训练损失（SMA / CMA / FMA）及各 VLA 基线的训练/微调脚本未包含在该仓库中。另 README 克隆命令中提及的 `https://github.com/SunSeaLucky/MEMOBench` 地址经核查返回不存在（404），应以 `https://github.com/Collab-Gen/MEMOBench` 为准。
