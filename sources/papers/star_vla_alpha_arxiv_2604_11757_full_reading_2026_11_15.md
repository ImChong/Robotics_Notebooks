# StarVLA-α：原文通读与实现复核（2026-11-15）

- **类型：** paper
- **原文：** <https://arxiv.org/html/2604.11757>
- **版本：** 2026-11-15 获取的 HTML；具体正文、附录和代码范围如下。
- **代码：** <https://github.com/starVLA/starVLA>
- **核对日期：** 2026-11-15
- **核对范围：** arXiv:2604.11757 HTML（2026-11-15 获取）正文 §1–7、附录 A–I（基准、训练、初始化/宽度/batch消融、RoboChallenge、真机OOD、逐任务结果、可视化说明、LIBERO-Plus）；核对全部结果表与图注。另读官方 starVLA_dev 的 QwenOFT.__init__/forward/predict_action 及 LIBERO 启动脚本；沿用已完成的框架论文/官方指南归档，未进行GPU训练或真机运行。
- **通读：** 已完成

本次复核单独存档；[历史摘录](star_vla.md)保留原有日期和发布状态，不能代替下述版本及协议核对。

## 原文与实现核对

- [α主论文](https://arxiv.org/html/2604.11757)与[框架论文](starvla_framework_arxiv_2604_05014_full_reading.md)是两篇独立论文，分别回答受控基线与可组合基础设施问题。
- α默认Qwen3-VL-4B+MLP回归动作chunk、无额外机器人预训练；动作归一化只用训练集。Generalist将不同动作补零到32维，但环境适配器仍需还原各本体接口。
- 表1/5的specialist LIBERO98.8%、GoogleVM76.0%与generalist97.8%/74.3%是不同训练协议；不能合并成一个模型的成绩。RoboChallenge ARX5十一任务SR33.6%对π0.5的12.7%，差20.9个百分点；进度分54.5对27.6，不能写成同一种指标的“提升20%”。
- 附录F的Franka三任务ID/OOD均值85.3%/76.3%，与ARX5主表属于不同平台和协议。LIBERO-Plus同样单独报告specialist77.8%、generalist79.7%。
- 附录C backbone/head学习率分别1e-5/1e-4、最多100k步，单基准与多基准算力不同；表5与表1的若干基线值并不一致，精确比较须点名表格/设置，不能混列。
- [当前QwenOFT源码](https://github.com/starVLA/starVLA/blob/starVLA_dev/starVLA/model/framework/VLM4A/QwenOFT.py)读取图像/指令，构造action token，取末层对应hidden state，调用action_model.predict_action；训练使用masked_l1_loss，推理返回normalized_actions。其docstring有残留diffusion字样，但该路径没有扩散采样。
- [当前LIBERO脚本](https://github.com/starVLA/starVLA/blob/starVLA_dev/examples/simBenchmarks/LIBERO/train_files/run_libero_train.sh)默认为QwenPI和Qwen3.5-0.8B，不能直接当作α论文Qwen3-VL-4B+MLP的复现实验；需要显式匹配framework、backbone、数据和归一化。所读仓树13d8ad43。

## 对 wiki 的映射

- [StarVLA-α 详情](../../wiki/entities/paper-starvla-alpha.md)
- [StarVLA 方法入口](../../wiki/methods/star-vla.md)
