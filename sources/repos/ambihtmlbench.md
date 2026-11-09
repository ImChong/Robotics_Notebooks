# AmbiHTMLBench｜官方源码

> 来源归档（公开代码仓，静态核对）

- **类型：** repo
- **链接：** <https://github.com/theseus-labs-rsi/AmbiHTMLBench>
- **项目页：** <https://ambihtmlbench.tech/>
- **论文：** <https://arxiv.org/abs/2611.06449>
- **入库日期：** 2026-11-09
- **核对范围：** README、入口脚本和文中列出的主要运行模块；没有在本次环境运行完整依赖或评测。
- **通读：** 未完成

## 已核查入口

- **运行入口：** `main.py` 调用 `benchmark.cli.main`；后者的 `eval-objective-v3` 子命令调用 `benchmark.objective_v3.evaluate_all_models_v3`。
- **核心模块（静态核对）：** `objective_v3.py` 的 `load_gt_v3` 读含 ID/contentList 的 Ground Truth，联合 `benchmark.html_extractor.extract_html_with_page`、结构匹配与布局评分，输出 CSV。
- **主观路径：** `benchmark/frozen_proxy_cli.py`、`benchmark/frozen_proxy.py` 及 `configs/subjective_ols5_frozen.json` 用于冻结判断重放和校准。
- **公开边界：** 预测 HTML 不随主仓提供；主观原始投票和人类数据需外部获取。主仓 MIT。

## 工程复现建议

1. Python 3.10+ 安装 `pip install -e .`，如跑客观路径则另 `python -m playwright install chromium`。
2. 从 HF 数据集取图像，将模型输出放到 `all_model_infer/<model>/html/0001_*.html` 等路径。
3. 执行 `python main.py eval-objective-v3 --gt-json AmbiHTMLBench_GT_v5_1115.json --infer-dir all_model_infer --out-dir outputs/objective_v3`。
4. 主观复现须另取 `single_results.jsonl` 录制判决，执行 `python -m benchmark.frozen_proxy_cli replay ... --config configs/subjective_ols5_frozen.json`；该 replay **不调用**裁判 API，也不能替代重新评测。

## 局限

README 提供评分实现和标注 JSON，但主观评测的原始 judge 结果、图像及人工注释需额外供应；主观 replay 本身不重新调用模型或生成新判断。

## 对 wiki 的映射

- [AmbiHTMLBench 主实体](../../wiki/entities/paper-ambihtmlbench.md)
- [论文归档](../papers/ambihtmlbench_arxiv_2611_06449.md)
- [官方项目页归档](../sites/ambihtmlbench.md)
