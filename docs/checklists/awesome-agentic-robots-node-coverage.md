# Awesome-Agentic-Robots 独立节点分批补全

来源快照：[`56f77326c08b9cc5918d50d379f2c35955bcc2a5`](https://github.com/chonghanqin/Awesome-Agentic-Robots/tree/56f77326c08b9cc5918d50d379f2c35955bcc2a5) 的 `data/papers.yaml`，307 条记录。

机器可读台账：[node-coverage.json](awesome-agentic-robots-node-coverage.json)。`candidate_nodes` 为元数据中的 arXiv 编号匹配候选，不能当作身份和内容均已验收；空候选仍需按 DOI、标题、项目页、源码进行全库查重。多分类复用同一个独立论文节点。后续批次更新状态与候选，勿重新建同篇论文页。

## 当前阶段：先覆盖 stub 与地址

按用户最新要求，健康度与深读暂不作为建页条件。范围补入同一快照的 additional_papers.yaml 23 条，合计 **330 条**。

- 独立节点覆盖：**330/330**；本次新建 **229** 个 stub，复用 **101** 个既有节点；**缺节点 0**。
- arXiv 地址：**324** 条；其余 **6** 条保留出版方 / 报告 / 项目入口，未猜测 arXiv。
- 项目页或官方代码/发布入口：**244** 条；另 **86** 条未从已查一手来源确认地址，逐条标记待核实。
- 本任务已深读：**4**；未深读验收：**326**。节点覆盖不等于深读完成。

全量导航：[330 条独立节点](../../wiki/overview/awesome-agentic-robots-node-index.md)。JSON 台账保存每条节点路径、地址、地址来源和待核状态。

## 原深读批次记录

此前深读按主清单 307 条计数；当前完整范围修正为 330 条，深读仍完成 4 条。

## 第一批

- 已完成：[PAFF](../../wiki/entities/paper-paff-2212-07398.md)，原文 v4 正文通读，官方代码入口和默认重标注分支静态核查，未运行训练。
- 元数据候选：79 条，尚未人工验收。

## 第二批

- [TidyBot](../../wiki/entities/paper-tidybot-2305-05658.md)：固定 v2、官方代码入口与真机/文本基准口径；不与 TidyBot2 合并。
- [OLAF](../../wiki/entities/paper-olaf-verbal-correction.md)：固定 v1、介入前动作重标注；Code 空链接、预算差异与键盘反馈边界已说明。
- [LMPC](../../wiki/entities/paper-lmpc-2402-11450.md)：固定 v2、官方教学 Colab；区分示例实现与论文完整训练/部署系统。
- 本批完成 3 条；累计 **4/307**，剩余 **303**。
- 下一批：Yell At Your Robot（2403.12910）、π*₀.₆（2511.14759，已有候选优先升级）、PhysMem（2602.20323）。

## 验收规则

当前先保证一项目一节点及来源地址；新页保持 stub，已有页状态保留。后续深读再补流程图、评测、源码时序与健康度。PR 保留草稿，地址尚未确认的记录不冒充完整链接覆盖。
