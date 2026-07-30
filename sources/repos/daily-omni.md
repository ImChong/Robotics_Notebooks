# Lliar-liar/Daily-Omni

> 来源归档

- **标题：** Daily-Omni（官方实现）
- **类型：** repo
- **组织 / 作者：** Lliar-liar（复旦大学 Zhou / Wang / Wu / Jiang 线）
- **代码：** <https://github.com/Lliar-liar/Daily-Omni>
- **许可证：** **GPL-3.0**
- **论文：** <https://arxiv.org/abs/2505.17862>
- **项目页：** <https://lliar-liar.github.io/Daily-Omni/>
- **数据集：** <https://huggingface.co/datasets/liarliar/Daily-Omni>
- **入库日期：** 2026-07-30
- **一句话说明：** Daily-Omni 基准官方仓：半自动 QA 生成管线、本地/API 模型评测脚本、训练无关 Daily-Omni Agent 基线；评测数据从 HF 下载 `Videos.tar` + `qa.json`。

## 开源核查（2026-07-30）

- GitHub API：`license.spdx_id = GPL-3.0`；默认分支 `main`；homepage 指向 arXiv。
- README 显式链到项目页、HF Dataset、Leaderboard；News 记录 2026-07-26 榜单更新（含 AGIBOT WITA-Omni Preview）。
- **已开源** 可运行：QA 生成（需 API Key）、API/本地评测入口、`baseline/` 诊断 Agent；大视频包在 HF，不在 git 内。

## 入口速查（对齐仓库树）

| 路径 / 命令 | 作用 |
|-------------|------|
| `run_pipeline.py` | QA 生成主入口（captioning → revision/alignment → qa_generation → optimize → filter） |
| `config.py` | API keys / `BASE_DIR` / `CSV_PATH` / worker 配置 |
| `captioning.py` / `revision.py` / `qa_generation.py` / `question_optimize.py` / `qa_filter.py` | 管线各阶段 |
| `baseline/` | Daily-Omni Agent：`base_model.py`、`v_caption.py`、`a_caption.py`、`v_event.py`、`segment_av.py` |
| `test_model_api/test_model.py` | 第三方 API 评测（Gemini / GPT-4o / Deepseek 等） |
| `test_model_api/test_config.py` | API 模型选项 |
| `test_model/*/testmodel.py` | 本地评测（Qwen Omni/VL、Ola、VideoLLaMA2、Unified-IO 2 等） |
| `--input_mode {all,visual,audio}` | 统一模态消融开关（默认 `all`） |
| `qa.json` / `example_videos/` / `example_metadata.csv` | 题库与管线模板 |
| HF `Videos.tar` + `qa.json` | 完整评测资产（解压到仓库根目录） |

## 最短复现路径

1. `pip install -r requirements.txt`。
2. 从 HF 下载 `Videos.tar` 与 `qa.json`，解压 `Videos/` 到仓库根目录。
3. **评测（API）：** `python test_model_api/test_model.py --model <name> --mode <Execution_mode> [--max_items N]`。
4. **评测（本地）：** 按各子目录 README 安装模型依赖后运行 `python test_model/<Model>/testmodel.py ... --input_mode all`。
5. **再生 QA（可选）：** 编辑 `config.py` 与 `run_pipeline.py` 的 `run_pipeline_flags`，再 `python run_pipeline.py`（可用 `example_videos` 冒烟）。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Daily-Omni](../../wiki/entities/paper-daily-omni.md) | 实体归纳：六任务族、管线、榜单、Agent 基线 |
| [RoboBench](../../wiki/entities/robo-bench.md) | 同属 MLLM 认知评测；Daily-Omni 偏 **日常 AV 时序对齐**，RoboBench 偏 **操纵 System 2** |
| [具身评测基准选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md) | ① 层补「跨模态同步」诊断轴 |
| [智元 / AgiBot](../../wiki/entities/agibot-lingxi-x1.md) | 榜首 WITA-Omni Preview 来自 AGIBOT X-Lab（闭源预览） |

## 对 wiki 的映射

- 论文摘录：[`sources/papers/daily_omni_arxiv_2505_17862.md`](../papers/daily_omni_arxiv_2505_17862.md)
- 项目页：[`sources/sites/daily-omni-github-io.md`](../sites/daily-omni-github-io.md)
- 沉淀 **[`wiki/entities/paper-daily-omni.md`](../../wiki/entities/paper-daily-omni.md)**
