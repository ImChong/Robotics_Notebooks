# Awesome-Agentic-Robots 独立节点分批补全

来源快照：[`56f77326c08b9cc5918d50d379f2c35955bcc2a5`](https://github.com/chonghanqin/Awesome-Agentic-Robots/tree/56f77326c08b9cc5918d50d379f2c35955bcc2a5) 的 `data/papers.yaml`，307 条记录。

机器可读台账：[node-coverage.json](awesome-agentic-robots-node-coverage.json)。`candidate_nodes` 为元数据中的 arXiv 编号匹配候选，不能当作身份和内容均已验收；空候选仍需按 DOI、标题、项目页、源码进行全库查重。多分类复用同一个独立论文节点。后续批次更新状态与候选，勿重新建同篇论文页。

## 总进度

**总计 307 条，累计完成 4 条，剩余 303 条待验收。** 已有候选但未核查的条目仍计入剩余。

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

每篇独立论文一个唯一详情节点；已有节点优先升级。须有来源、稳定身份、原文流程图，有可运行官方实现时增加对齐源码的时序图。健康度与检查以每批提交实际运行结果为准；不得把清单候选数计为完成数。全部条目验收前保留 PR 草稿。
