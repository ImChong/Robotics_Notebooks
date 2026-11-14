# DSWAM: A Dual-System World Action Foundation Model for Fine-Grained Robot Manipulation

- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2607.04927>
- **全文版本：** <https://arxiv.org/pdf/2607.04927v1>
- **核对日期：** 2026-11-13
- **核对范围：** arXiv v1 PDF，全文 §1–§5（11 页正文），表 1–5 与图 1；参考文献浏览。图表以文字层核对，未观看演示视频，未运行训练或真机实验。
- **通读：** 已完成
- **项目页：** <https://ds-wam.github.io/>

## 通读核对

- §3.2：System 2 用 Rynnbrain4B 风格 VLM，5 帧 / 1 FPS 历史；末窗口监督下一子任务或 done，部署每 2 秒更新。不是一次性固定拆解全部任务。
- §3.3：动作与视频 flow matching 共训；动作不能读取未来视觉 token；推理无未来视频生成。RTC 只在推理期，未使用训练期 action conditioning 或 suffix-only loss。
- §4.1–4.3 / 表 2：4 类衣物，各 2 实例 × 10 次，共 80 次；完整流程含取衣、铺展和折叠。96.3% 为四类均值；失败计超时时间；System 2 关闭。裤子 75%→90%，衬衫 95% 持平。
- §4.4 / 表 3：子任务监督排序实验，原始指令均值 75.7% / 3.53 错误，子任务 100% / 0.65；须与折叠执行器对照分开。
- §4.5 / 表 4–5：RTX 5090、batch 1 热启动，PyTorch 198.2 ms→BF16 TensorRT 73.8 ms；异步 RTC 的时间表仅容易衣物，不能与表 2 混比。

## 对 wiki 的映射

- [DSWAM: A Dual-System World Action Foundation Model for Fine-Grained Robot Manipulation](../../wiki/entities/paper-dswam-dual-system-wam.md)
