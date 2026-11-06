# Democratizing Robot-Use Agents（arXiv:2611.01560）

- **类型：** paper；版本 v1（2026-11-01）
- **作者：** Jiarui Yuan、Yuxin Zuo、Xiaoyu Huang 等；清华大学、加州大学伯克利分校。
- **原文：** <https://arxiv.org/html/2611.01560v1> · [PDF](https://arxiv.org/pdf/2611.01560v1)
- **项目：** <https://uni-robot-agent.com> · **代码：** <https://github.com/Open-RUA/uni-robot-agent>
- **博客：** <https://openrua.ai/uni-robot-agent> · **文档：** <https://uni-robot-agent.com/docs>
- **入库日期：** 2026-11-03

## 核对范围与待办

- **通读：** 未完成

已核对论文摘要、方法、主要实验、消融、真机评分、运动监控及案例等部分；全文尚未逐节通读，需继续核对其余部分。项目页、博客及在线文档访问失败；代码仓库可获取。因原文未全部通读，本收录保持草稿。

## 核心摘录

- 框架用共享工具和适配器统一不同环境，结合资源调度使推理与执行重叠；依赖同一资源的调用仍按顺序执行。
- RoboDojo 42 项任务、252 episodes：Qwen3.8-Flash-Next 为 25.00% 成功率 / 31.39 分；对照是已发表的 Astra–RoboProbe 22.48% / 28.97 分。论文还报告 Astra 在本框架中获得 51.90 分。
- 真机 19 项任务的 49.50 是部分得分制的 100 分总分，不是成功率；agent 使用深度，而对照模型使用 RGB。

## 对 wiki 的映射

- [Uni-Robot-Agent](../../wiki/entities/paper-uni-robot-agent.md)
