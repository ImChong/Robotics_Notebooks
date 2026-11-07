# Dreaming with a dynamic workflow（Claude Cookbook）

> 来源归档（site / 官方 cookbook 页面与 notebook）

- **标题：** Dreaming with a dynamic workflow
- **类型：** site（Claude Cookbook · Claude Managed Agents · Agent Patterns）
- **作者：** Brad Abrams
- **原始链接：** https://platform.claude.com/cookbook/managed-agents-cma-dreaming-with-a-dynamic-workflow
- **源码：** <https://github.com/anthropics/claude-cookbooks>（MIT；浅克隆提交 `ce8b262`）的 `managed_agents/CMA_dreaming_with_a_dynamic_workflow.ipynb`、`dreaming_nightly.py`、`memory_distill_utils.py`、`example_data/memory_distill/`
- **发表日期：** 2026-11-02
- **入库日期：** 2026-11-07
- **抓取方式：** curl 取网页 HTML 去标签（约 7.5 万字符）+ 浅克隆仓库，用 `python3 -I` 读 notebook 的 35 个单元（总计约 4 万字符）
- **核对范围：** 网页叙述、四份提示词（工作请求模板、lead / analyzer / organizer 的系统提示）、示例运行输出、全部 markdown 说明单元与关键代码单元（创建智能体、启动会话、挂载只读记忆库、定时脚本说明）读完；**未读** `memory_distill_utils.py`（约 12.6 万字符的辅助代码）与 `dreaming_nightly.py` 的函数体，只确认其存在；**没有运行任何代码**（需要 Managed Agents beta 凭据）。
- **通读：** 已完成
- **一句话说明：** 一个可改造的「小型 dreaming」：Managed Agents 动态工作流读取已结束的会话转写，由 analyzer 智能体蒸馏成 wiki 式记忆（小文件 + `_index.md`），organizer 整理，供下次会话只读挂载，可用 cron 每晚运行。
- **说明：** 与 Anthropic 博客 [The components of agent memory](../blogs/anthropic_components_of_agent_memory_2026-11-06.md) 配套（博文文末指向本 cookbook）。Dreams 本身是 Managed Agents 的 research preview，本 cookbook 是用动态工作流自己实现同一思路。

## 页面要点（作者表述）

- **三步：** ① 把智能体已结束的会话导出成转写文件（每会话一个文本文件，事件标签 USER / AGENT / TOOL CALL / TOOL RESULT）；② 一条消息描述工作，lead 智能体（`memory-distiller`）写工作流并启动 workflow run：每份转写交给一个 `transcript-analyzer`（读全文、核对已有记忆、补充、整理并维护 `_index.md`），随后一个 `memory-organizer` 合并重复、修链接、补全索引；③ 下次会话把记忆库**只读**挂载。
- **模型与成本：** lead 用 Claude Sonnet 5.5，analyzer 与 organizer 用 Claude Haiku 5.5（均可用 `COOKBOOK_` 前缀环境变量覆盖）；11 份示例转写（283 KB）约 $0.36、约 2.5 分钟；50 多份长转写（约 12 MB 文本）默认设置约 $8、约 10 分钟，不用降本设置约 $10、9 分钟（作者表述，60 与 30 次 dream 的统计）；预估成本可能是真实成本的数倍，需按预估设预算（默认 `BUDGET_USD` = 40）。
- **工作方式：** 转写按约 20 个一组分批；一个 workflow run 最多同时 64 个线程，一个会话最多挂 500 个文件；分析器 prompt 的四步为读全文 → 核对（verify）→ 补充（enrich）→ 整理（organize）；要求并发写同一目录时「写前重读、写后读回验证、合并而非覆盖」；记忆文件带 frontmatter（`description:`）与 `[[slug]]` 链接，每个文件 < 100 KB。
- **安全设计：** 转写当不可信数据；analyzer 只有 read / write / edit / glob / grep 五个文件工具，无 shell、无网络（环境允许主机列表为空）；每个工具走平台 `auto` 权限策略；不得把密钥写进记忆；记忆库挂载给使用会话时用只读；dreaming 会话是唯一读写挂载处；每次写入是一个保留 30 天的记忆版本，可审计、可回写、可脱敏。
- **加载到每个会话：** 可读取记忆库里的小文件（如 `_index.md`）作为挂载的 `instructions` 注入系统提示，上限 4,096 字符；注意每个会话都耗 token、错误行会触及每个会话。
- **每晚运行：** `dreaming_nightly.py` 用 nbclient 跑该 notebook，cron 示例 `0 1 * * *`；导出发生在你机器上（会话不能以其他会话为输入）；失败的夜晚下次重试；只导出已结束（idle / terminated）的会话；默认无人工审核。
- **数据处理：** Managed Agents 会话不符合零数据保留或 HIPAA；会话保存其读取的内容，notebook 导出后删除会话；记录写入临时目录下的私有文件夹。

## 对 wiki 的映射

- [Claude Cookbook：用动态工作流做 dreaming](../../wiki/entities/claude-cookbook-dreaming-dynamic-workflow.md)
- [Agent 记忆与 dreaming](../../wiki/concepts/agent-memory-and-dreaming.md)
