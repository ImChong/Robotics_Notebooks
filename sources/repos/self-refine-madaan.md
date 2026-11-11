# madaan/self-refine — Self-Refine 官方源码

> 来源归档（官方 GitHub）

- **仓库：** <https://github.com/madaan/self-refine>
- **类型：** repo / research implementation / prompting
- **对应论文：** [arXiv:2303.17651v2](../papers/self_refine_arxiv_2303_17651.md)
- **项目页：** [Self-Refine](../sites/self-refine.md)
- **开源状态：** 已开源（公开代码、prompt 模板与数据，运行需外部模型 API）
- **收录日期：** 2026-11-11
- **核对范围：** README 的 Setup / 任务命令 / General setup；src/acronym/run.py、task_init.py、feedback.py、task_iterate.py；src/responsegen/run.py。未在本环境运行收费 API 实验或复现论文全套分数。

## 真实运行入口

| 任务 | README 入口 | 观察 |
|---|---|---|
| 缩写生成 | python -u src/acronym/run.py "Using language models of code for few-shot commonsense" | 本归档时序图主要依据；默认五次尝试 |
| 对话生成 | python -u src/responsegen/run.py --output OUTPUT --size N | iterative_response 分别创建 Init / Feedback / Iterate |
| 数学 GSM8K | python -u src/gsm/run.py | 输出再经 gsm_selfref_eval.py 评测 |
| 代码优化 PIE | python -u src/pie/run.py --slow_programs_file ... --max_attempts 4 ... | 代码优化任务级入口 |
| 代码可读性 | python -u src/readability/readability.py --output OUTPUT | 需先解压 CodeNet 数据包 |

## 缩写生成入口的实现事实

1. src/acronym/run.py 在 iterative_acronym(title, max_attempts) 中实例化 AcronymGenTaskInit、AcronymGenFeedback、AcronymGenTaskIterate。
2. **第 0 轮** task_init(title) 读取 data/prompt/acronym/init.jsonl，使用 prompt_lib.backends.openai_api 生成缩写。
3. 每轮 task_feedback(title, acronym) 读取 data/prompt/acronym/feedback.jsonl，通过同一个 OpenaiAPIWrapper 获取多个维度的评分文本；run.py 用正则抽取 Total score。
4. **后续轮** task_iterate(acronyms_to_scores=...) 构造包含旧候选和分数的提示，调用 API 生成新的 title / acronym。
5. 达到 max_attempts 后 run.py 返回 all_acronyms_to_scores；**并未**在这一入口中实现按反馈质量自动提前结束的停止器。run.py 中 total_score >= 0 的判断并非真正的「比历史最好高」，不应宣传为单调择优。
6. 样例依赖 reasoning-machines/prompt-lib，并使用历史 OpenAI 模型标识；是否能在当下 API 直接运行未经本次复测。

## 对 wiki 的映射

- [Self-Refine：基于自反馈的迭代改进](../../wiki/entities/paper-self-refine-2303-17651.md)

## 参考来源

- [官方 README](https://github.com/madaan/self-refine/blob/main/README.md)
- [缩写任务入口](https://github.com/madaan/self-refine/blob/main/src/acronym/run.py)
- [缩写初稿](https://github.com/madaan/self-refine/blob/main/src/acronym/task_init.py)
- [缩写反馈](https://github.com/madaan/self-refine/blob/main/src/acronym/feedback.py)
- [缩写改稿](https://github.com/madaan/self-refine/blob/main/src/acronym/task_iterate.py)
- [对话任务入口](https://github.com/madaan/self-refine/blob/main/src/responsegen/run.py)
