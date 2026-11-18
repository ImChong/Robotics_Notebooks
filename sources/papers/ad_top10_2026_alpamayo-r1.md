# Alpamayo-R1: Bridging Reasoning and Action Prediction for Generalizable Autonomous Driving in the Long Tail

- **类型：** 原始论文摘录（不转存论文）
- **论文：** https://arxiv.org/html/2511.00088
- **arXiv ID：** 2511.00088
- **机构：** NVIDIA
- **代码：** https://github.com/NVlabs/alpamayo
- **核对日期：** 2026-11-18
- **核对范围：** 论文 §3–§6（模型、CoC数据、监督及RL后训练、实验），附录及官方README、test_inference.py；核对公开10B非RL版本
- **通读：** 已完成

## 方法

多相机和自车历史经 Cosmos-Reason VLM 处理，Chain of Causation（CoC）训练场景解释，扩散轨迹专家预测未来 6.4 秒（64 waypoint）。

## 原文实验及边界

论文展示困难场景规划精度和闭环近距离风险的改进；仓库公共10B权重没有完成论文所述RL后训练。

官方仓已转为有限维护；HF权重和数据需单独申请；推理要求GPU显存至少24GB；公共10B不接收显式导航路线输入。

## 对 wiki 的映射

- [唯一详情节点](../../wiki/entities/paper-alpamayo-r1-2511-00088.md)
- [2026 自动驾驶十项目](../../wiki/overview/autonomous-driving-open-source-top10-2026.md)
