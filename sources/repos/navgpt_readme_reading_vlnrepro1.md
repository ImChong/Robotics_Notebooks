# NavGPT 官方仓库 README 通读（VLN 复现范式页支撑）

> 来源归档（原文通读 · 支撑 `wiki/overview/vln-open-source-repro-paradigms.md`）

- **标题：** NavGPT: Explicit Reasoning in Vision-and-Language Navigation with Large Language Models（官方 GitHub README）
- **作者：** Gengze Zhou, Yicong Hong, Qi Wu
- **类型：** repo / navigation / llm-agent
- **原文：** <https://github.com/GengzeZhou/NavGPT>（master 分支 README，2026-11 经代理获取全文）；论文 <https://arxiv.org/abs/2305.16986>
- **一句话说明：** 纯 LLM 的指令跟随导航智能体：每步输入视觉观测的文本描述、导航历史与可探索方向，零样本预测下一步动作。
- **核对范围：** 通读 README 全文（Abstract、Method 图占位、TODOs、安装、数据、OpenAI API、R2R 复现命令、自定义 LLM、Citation）。未读论文正文、未读源码；wiki 中 `NavAgent.test()` 等类名未在 README 中出现，未核对。
- **通读：** 已完成

## 核心摘录（均出自 README）

- **方法（Abstract）：** 每步以视觉观测文本描述、导航历史、未来可探索方向为输入，推理当前状态并决策；展示子目标分解、常识利用、地标识别、进度跟踪、异常时调整计划；也能由路径生成导航指令、画俯视度量轨迹。
- **局限（作者自述）：** 零样本 R2R 性能仍不及训练过的模型；建议为 LLM 引入多模态输入，并把 LLM 的显式推理用于增强学习式模型。
- **环境：** conda Python 3.9，`pip install -r requirements.txt`；R2R 数据从 Dropbox 下载放入 `datasets`；预处理脚本在 `nav_src/scripts`。
- **API：** 需 `OPENAI_API_KEY` 环境变量。
- **复现：** 论文结果用 `--llm_model_name gpt-4` 跑 `R2R_val_unseen_instr`；默认模型为 `gpt-3.5-turbo`；经济试跑用 GPT-3.5 加 `--iters 10`（前 10 个样本）。
- **自定义 LLM：** 在 `nav_src/LLMs/` 下加入自己的模型代码并按 `Add_Custom_Models.md` 配置。

## 对 wiki 的映射

- [`wiki/overview/vln-open-source-repro-paradigms.md`](../../wiki/overview/vln-open-source-repro-paradigms.md)
