# Sparks of Weight-Space Programming / 权重空间编程的初步探索

- **类型：** 博客；2026-11-02，入库时网页版本（2026-11-03）
- **作者：** Rikiya Takehi、Jayden Teoh、Alex Zhang、Omar Khattab；麻省理工学院 CSAIL
- **原文：** <https://rikiyat.github.io/blog/weight-space-programming/>
- **实验日志与代码：** <https://github.com/RikiyaT/lm_understand-logs>
- **编辑后的模型：** <https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming>
- **阅读范围：** 博客全文（实验、附加实验、局限和日志）；仓库 README、实验 8 README、权重重建脚本。
- **开放状态：** 已发布实验日志、部分脚本、网络和编辑差分及 Qwen 权重；实验 8 的 agent harness、部分中间文件等未包含。仓库仅收录文章报告的运行，早期提示词试验未收录。

## 摘要与证据

1. 作者将通用 LLM 根据自然语言目标直接读取、修改或编写权重，且不使用梯度训练，称为权重空间编程。不同实验允许的工具和反馈不同，不能一概视为零反馈。
2. 小网络实验中，模型从权重辨认游戏、图形和字符模型训练语言；Kinetix 小车实验给出场景与一次轨迹，在单次编辑后从 0/256 到 1024/1024 成功。MinAtar 中，生成权重在三款游戏的设定下超过训练 1 亿步的 PPO。
3. 9×9 围棋实验允许 Python 和 0–50 次有限前向反馈；50 次时作者报告超过同规模 PPO。谄媚模型辨别仅靠权重四次失败；允许注意力模式后辨别三组模型。
4. Qwen3.5-0.8B 实验允许练习题及最多 50 次前向反馈。作者报告在指定采样设置下，MATH-500 7.9%→48.1%，GSM8K 13.8%→67.1%，IFEval 21.0%→62.3%；触及长度上限比例 93.0%→33.1%。编辑触及两层 131,072 个权重。
5. 对少有循环的 Gemma 3 270M，数学成绩提升不显著；作者明确区分消除阻碍与创造新能力。实验日志披露部分复现组件未开放，不能据此认定方法已能普遍替代训练。

## 对 wiki 的映射

- [Weight-Space Programming 实体](../../wiki/entities/weight-space-programming.md)
