# Awesome-Agentic-Robots 独立节点分批补全

来源快照：[`56f77326c08b9cc5918d50d379f2c35955bcc2a5`](https://github.com/chonghanqin/Awesome-Agentic-Robots/tree/56f77326c08b9cc5918d50d379f2c35955bcc2a5) 的 `data/papers.yaml`，307 条记录。

机器可读台账：[node-coverage.json](awesome-agentic-robots-node-coverage.json)。`candidate_nodes` 为元数据中的 arXiv 编号匹配候选，不能当作身份和内容均已验收；空候选仍需按 DOI、标题、项目页、源码进行全库查重。多分类复用同一个独立论文节点。后续批次更新状态与候选，勿重新建同篇论文页。

## 第一批

- 已完成：[PAFF](../../wiki/entities/paper-paff-2212-07398.md)，原文 v4 正文通读，官方代码入口和默认重标注分支静态核查，未运行训练。
- 元数据候选：79 条，尚未人工验收。
- 下一批按来源顺序推进：TidyBot（2305.05658）、Interactive Robot Learning from Verbal Correction（2310.17555）、Learning to Learn Faster（2402.11450）。先做标题、DOI、官方地址查重，再核读和建页。

## 验收规则

每篇独立论文一个唯一详情节点；已有节点优先升级。须有来源、稳定身份、原文流程图，有可运行官方实现时增加对齐源码的时序图。健康度与检查以每批提交实际运行结果为准；不得把清单候选数计为完成数。全部条目验收前保留 PR 草稿。
