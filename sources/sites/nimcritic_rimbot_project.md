# NimCritic 官方项目页与在线演示核查

> 此归档刻意区分已核验的信息与网络读取阻塞；“网页加载失败”不等于“项目不存在”或“未开源”。

- **标题：** NimCritic — Predictive Long-Horizon Progress Modeling with Action-Aware Quantification / 长时程动作感知进度预测
- **类型：** site / project / interactive-demo
- **项目：** https://rimbot.com/research/nimcritic/
- **在线体验：** https://critic.rimbot.com/
- **机构研究列表：** https://rimbot.com/research/
- **论文：** https://arxiv.org/abs/2610.32776（[原文归档](../papers/nimcritic_arxiv_2610_32776.md)）
- **代码：** 未核验到官方 GitHub / 可运行仓库 URL（非“确定未开源”）
- **权重与 LOOM 标注：** 发布状态待官方页面/下载入口核验
- **核对日期：** 2026-11-15
- **核对范围：** 打开研究列表，确认其收录 2026-10-23 NimCritic 研究条目；尝试打开项目详情页发生超时，在线 Demo 页可响应但仅返回无正文的客户端页面；论文 HTML 全文另已阅读。
- **通读：** 未完成
- **沉淀到 wiki：** [NimCritic 论文实体](../../wiki/entities/paper-nimcritic.md)

## 已确认的官网信息

- 日冕研究列表公开了 2026-10-23 研究条目《基于动作感知量化的机器人长程操作进度预测建模》，摘要描述物体级 VLM 分段、加权动作贡献以及 NimCritic 进度预测。
- 官网索引摘要使用了 **NimProgress** 名称；论文 v1 把带有 300 段轨迹和 19,618 帧的测试集称为 **LOOM**。两者是否为同一数据集的更名，本次**不能确认**，wiki 保持论文 v1 的 LOOM 命名。
- `critic.rimbot.com` 是用户提供的在线体验地址；当前抓取只确认 URL 可响应，无法从零行可读正文的客户端入口确认所用模型、访问条件或可上传视频的实际流程。

## 访问阻塞与开源核验结果

- 项目详情 `https://rimbot.com/research/nimcritic/` 多次读取超时，无法检查 Footer、Code、Resources、GitHub、Hugging Face 下载按钮；不要把超时当作已经完成的资源开放核查。
- 未在本次可核实的一手资料中取得 NimCritic 训练/推理代码、模型权重或 LOOM 标注数据的官方可下载 URL；因此**代码/模型/数据开放程度待核实**。不为未知的官方仓库创建虚假的 `sources/repos/` 记录，不添加推断出的 GitHub 链接。
- 若项目页恢复访问，先核查 Code / Data / Model 权限与实际 README/入口，再补 `sources/repos/` 与可核验的源码运行时序图；未有源码前本项目的源码运行时序图不适用。

## 一手链接

- [日冕研究索引](https://rimbot.com/research/)
- [NimCritic 项目](https://rimbot.com/research/nimcritic/)
- [在线体验](https://critic.rimbot.com/)
- [论文 v1 全文](https://arxiv.org/html/2610.32776v1)
