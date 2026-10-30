# MMAU 官方仓库

- **链接：** <https://github.com/Sakshi113/MMAU>
- **核查日期：** 2026-10-30
- **论文：** <https://arxiv.org/abs/2410.19168>
- **项目页及榜单：** <https://sakshi113.github.io/mmau_homepage/>
- **开放状态：** 提供评测脚本和数据入口；test 真值不公开，需提交官方评测。

## 已核对的评测入口

README 给出 `python evaluation.py --input INPUT_JSON_PATH`。仓库的 `evaluation.py` 接收已经生成的输出，基于 token 集合匹配正确答案并按领域、难度等计算准确率。README 写入字段为 `model_prediction`，脚本读取 `model_output`；缺该字段的记录直接跳过，应核对输出数。

## 对 wiki 的映射

- [MMAU](../../wiki/entities/paper-mmau.md)
