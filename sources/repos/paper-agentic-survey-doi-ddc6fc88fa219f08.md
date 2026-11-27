# STARS 官方代码仓归档（Center-for-Integrated-Cognition/STARS）

- **官方仓库：** https://github.com/Center-for-Integrated-Cognition/STARS
- **关联论文：** [STARS: extending interactive task learning with large language models（Frontiers in Computer Science 2026 / DOI: 10.3389/fcomp.2026.1848830）](../../wiki/entities/paper-agentic-survey-doi-ddc6fc88fa219f08.md)
- **开源协议：** 仓库根目录及元数据未声明单独 `LICENSE` 文件（关联期刊论文采用 CC BY 4.0，仓库附带 AAAI 2024 论文 *"Improving Knowledge Extraction from LLMs for Task Learning through Agent Analysis"* 配套源码并在 2026 年 Frontiers 论文 Footnote 1 中作为官方代码入口引用）
- **默认分支与体量：** `main`（截至采集记录 4 stars，0 forks）
- **开源状态判定：** **已开源（Soar/Python ITL 智能体、MobileSim 仿真器与论文实验数据分析脚本完整发布；用户需自行配置 OpenAI API Key 与底层 C++/Java/Soar 依赖）**
  - 已发布内容：面向 `tidy-kitchen`、`store-groceries` 与 `organize-office` 三项任务的完整 ITL-Rosie 智能体（含 STARS Search Tree、Analysis & Repair、Selection 模块）、基于 April Robotics Toolkit 的 `mobile-sim` 机器人仿真器及环境配置文件，以及全部实验原始日志与 LaTeX 表格/图表绘制脚本（`DataAnalysisScripts/`）。
  - 边界说明：第 5.2 节展示的 AI2-THOR 厨房扩展案例（烹饪胡萝卜与类别概念接地）以论文补充 YouTube 演示视频提供，主仓库聚焦 MobileSim / APRIL MAGIC 仿真实验复现。
- **核对边界：** 静态代码核对，未本地运行（依据仓库目录结构、`files_sample`、`README.md`、`ITL-agent/README.md`、`mobile-sim/README.md`、`ITL-agent/examples/tidy-kitchen/run-STARS.sh`、`Makefile` 及 `ITL-agent/python/rosie/lm/{LMConnector.py,LLMHandler.py}` 与 `ITL-agent/agent/satisfy-purpose/dialog-event/language-model/` 文件清单核对入口与调用链）。

## 仓库结构与核心模块（基于实测文件清单）

- **根入口与说明（`README.md`）：**
  - 将仓库划分为 `ITL-agent`（智能体与 STARS 核心实现）、`mobile-sim`（机器人仿真器）与 `DataAnalysisScripts`（实验日志处理与图表复现）。
- **ITL 智能体与 STARS 核心逻辑（`ITL-agent/`）：**
  - **Python LLM 交互与 Search Tree 模块（`ITL-agent/python/rosie/lm/`）：**
    - `LMConnector.py`：继承 `pysoarlib.AgentConnector`，监听 Soar `output-link` 上的 `lm-request`、`lm-request-step`、`lm-request-retry` 与 `lm-selection-request` 指令，提取评估反馈（`uninterpreted`、`unknown-word`、`ungrounded-obj1`、可供性不匹配）并通过 `input-link` 的 `language-model` WME 回写候选节点。
    - `LLMHandler.py`：封装模板拼接（`construct_prompt`、`get_task_template`）、GPT-3 completions 束搜索分支生成（`get_LM_Goal_tree_temp0`）、定向修复重提示（`retry_lm_query`、`retry_node`、`get_retry_example`）与 GPT-4 候选索引选择（`parse_selection_request_GPT4`）；需在同目录放置 `APIKey.py` 配置 `OPENAI_API_KEY`。
    - `LMReponse.py`、`example_1.txt`、`example_2.txt`、`examples/`：响应数据结构（`LMResponse`、`LMEvaluation`、`LMSelection`）与提示词上下文示例。
  - **Soar 认知分析与修复控制模块（`ITL-agent/agent/satisfy-purpose/dialog-event/language-model/`）：**
    - `evaluate-parse-of-lm.soar`、`evaluate-affordance.soar`、`analyze-lm-response.soar`：在 Soar 工作记忆中执行语言解析验证、情景物体接地检查与物体可供性（affordance）校核。
    - `process-lm-actions.soar`、`preferences.soar`、`language-model_source.soar`：驱动候选选择、用户监督偏好问询与动作学习流程。
  - **任务复现入口（`ITL-agent/examples/{tidy-kitchen,store-groceries,organize-office}/`）：**
    - `run-STARS.sh`（及 `STARS+O`、`STAR`、`ST`、`TBP` 等对照脚本）：复制对应 `.script`、`rosie.config` 与 `settings.soar`，调用 `Makefile` 中的 `make run_example` 并行启动 `java soargroup.mobilesim.MobileGUI -w cleaner.world -f` 与 `python3 -m mobilesim.rosie.MobileRosieGUI agent/rosie-client.config --hide-script`。
- **移动机器人仿真器（`mobile-sim/`）：**
  - 基于 LCM（Lightweight Communications and Marshalling）、April Robotics Toolkit（Java/C）与 Soar SML（需 Python 3.10 + Java 17 + `pysoarlib`）构建；配置 `mobile-sim/config/robot.config.local` 后通过 `build_mobile_sim` 与 `run_mobile_sim` 启动物理/语义感知仿真循环。
- **日志分析与论文图表脚本（`DataAnalysisScripts/`）：**
  - `ITLAgent-LogProcessingScripts/DialogProcessing-VerificationAnalysis.ipynb`、`DialogProcessing-VerificationAnalysis-NewTasks.ipynb` 及 `GroceryTask/` 下的 `raw_data/{ST,STAR,STARS,STARS+O}/` 与 `output/GroceryTask-tables.tex`，用于复现响应分类统计、token 开销与完成率图表。

## 对 wiki 的映射

- [STARS：基于大语言模型扩展交互式任务学习（论文实体页）](../../wiki/entities/paper-agentic-survey-doi-ddc6fc88fa219f08.md)
