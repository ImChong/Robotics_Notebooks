# MAWAM NVIDIA 项目页与发布状态

- **项目页：** <https://research.nvidia.com/labs/lpr/mawam/>
- **论文：** <https://arxiv.org/abs/2611.25105v2>
- **机构：** NVIDIA、香港大学、MIT
- **核查日期：** 2026-11-29
- **核对范围：** 正式 NVIDIA 项目页的作者栏、资源入口、双阶段训练、真机与 RoboFactory 演示、消融和预测画面对照；与论文 v2 及其附录相互核对。
- **通读：** 已完成

## 公开内容

正式网站可访问，替代此前不可读的 staging 链接。页面包含三项真机任务、八项仿真任务的演示与结果，以及未来画面和实录的对照。按样本逐帧看的实录在第 8 步后可能来自下一次重规划，不能把它当作整个 24 步动作块的开环预测对照；论文附录 C.2 另有完整执行 24 步的定量实验。

项目页顶部代码入口显示 `Code soon`，未核实可下载的 MAWAM 源码、权重或训练示范。页面的 GR00T 2 链接是初始化底座；渲染项目页的前端代码也不是训练或推理实现。虽然网页顶部仍显示 `Paper soon`，arXiv 已公开论文全文。

## 对 wiki 的映射

- [MAWAM 唯一详情节点](../../wiki/entities/paper-mawam-coordinated-manipulation.md)
- [论文全文归档](../papers/mawam_arxiv_2611_25105.md)
