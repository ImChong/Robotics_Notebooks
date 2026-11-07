# The components of agent memory（Anthropic Engineering）

> 来源归档（blog / 官方工程博客）

- **标题：** The components of agent memory
- **类型：** blog（Anthropic Engineering 博客，主题：Agents、Context engineering、Agent memory）
- **作者：** Lance Martin（编辑指导 Caitlin Love 等；致谢含 David Hershey 的 Claude Plays Pokémon）
- **原始链接：** https://www.anthropic.com/research/components-of-agent-memory
- **发表日期：** 2026-11-06
- **入库日期：** 2026-11-07
- **抓取方式：** curl 取 HTML 后去标签，正文约 2.4 万字符
- **核对范围：** 全文逐段读完（含各节与图注文字）；**图片本身未读**（Figure 1–15 只读到图注与正文对图的描述）；文中引用的 CL-Bench 1.0、Harvey LAB、Claude Code 文档、Stainless 语料均为作者自述，未回各自原始资料核对；唯一对外部文献的核对是 sleep-time compute（arXiv:2504.13171，见下）。
- **通读：** 已完成
- **一句话说明：** 把 agent 记忆拆成「记忆笔记 + dreaming（后台整理）」两部分，用 CL-Bench、Pokémon Red、企业文档 wiki 与 Harvey LAB 的评测说明：让强模型自己判断显著性比写死规则更好；dreaming 可纠错、建 wiki 表示、把笔记提升到上层、按目标塑形、补全缺口。

## 文章要点（作者表述）

- **定义：** agent memory = Claude 从经验中获得并在会话间保留、以后可读的上下文；**不含**权重更新、上下文内学习、压缩（compaction）。类比人脑：海马体快速编码，睡眠 / 静息时重放并巩固；sleep-time compute 由 Letta 与 UC Berkeley 研究者提出，dreaming 即在空闲时回看日志整理记忆。
- **记忆是对未来的预测：** 写下一条规则 = 预测该模式会再次相关。CL-Bench 1.0 数据库问答任务：每题新会话，只有持久的记忆目录；Sonnet 4.6 只到「提出未检验的猜测」，Opus 4.7 合并成单一参考文档并标注已验证 / 未确认，Sonnet 5 在评分后标注确认并写更正，**Opus 5 首次闭环**（规则 + 验证依据 + 先前答案 + 陷阱警告），Opus 5.5 与 Sonnet 5.5 同样闭环（有记忆时得分与 Opus 5 相近，无记忆基线略高）。
- **显著性判断（Pokémon Red）：** 约 10,000 步时，Sonnet 3.7 写了 413 个记忆文件（225 个文件夹）、拿到 1 枚徽章，记录很快过时；Fable 5 写 9 个文件（2 个文件夹）加 `global.md`（当前目标、道馆计划、记忆使用规则），拿到 4 枚徽章；Opus 5.5 同样用小文件夹并通关。结论：记忆应存「游戏没明说的东西」（策略、计划、验证过的规则），屏幕上每回合可读的信息（金钱、位置）不必存。
- **引导：** 告诉 Claude「记忆是什么用的」，不告诉它「怎么用」。CL-Bench 上只告知有记忆文件夹就足以让 Opus 4.7 / Opus 5 明显受益；规则「只记录失败」让两者丢掉大部分记忆收益；引导对旧模型帮助最大（提示先读记忆再答、并显式检验猜测使 Sonnet 4.6 得分近乎翻倍；Sonnet 5 自己 9/10 次先读记忆，Sonnet 4.6 不到一半）。
- **dreaming 的五种用法：**
  1. **纠错：** 对照原始转写校验笔记。Pokémon Mt. Moon 例：笔记误记「宠物中心在东侧」，未纠正的运行三次掉回原路线且最终没到下一城；dreaming 纠正后直接回洞并抵达 Cerulean City。
  2. **建新的记忆表示（wiki）：** Stainless 内部文档约 35 万个文件、约 4.2 亿 token，25 道题；用 Opus 5 代理跑 128 个会话建 wiki，会话从 2 增到 128 时得分由 13% 升到 48%；wiki + 检索工具从 54% 升到 61%，仅检索 53%，仅 wiki 最好 48%；wiki + 检索每题省约 2 万 token。Harvey LAB 并购问答 24 题：需跨文档综合的题，wiki 87% 对检索 73%；单文档即可满足的任务 wiki 无明显增益；另一任务里 wiki 比原文还大（Borges 地图比喻）。
  3. **把笔记提升到记忆层级顶端：** 由 harness 在每个会话开头预加载；顶层必须保持很小（错误笔记会带进每个会话，指令文件越长遵循度越低）；Claude Code 的 auto memory 只加载 `MEMORY.md` 的前 200 行或 25 KB，作为索引。
  4. **按目标塑形：** LAB 合同任务，初始 wiki 多代理共写一张表，57 个合同事实只保住 41 个（72%）；让 Claude 反思失败并重写建 wiki 的指令后，每份合同一页、逐字引用条款，57/57（100%）。
  5. **补缺口：** 第二轮 dreaming 把指向缺失页面的 9 个链接改指现有页，并新增 3 页（含索引）。
- **开放问题：** 何时写记忆（当场写会分散主任务注意力；后台 dreaming 产物不一定立刻可用）；如何保持更新（dreaming 要算力，可聚焦易变信息或按访问频率调节节奏）；以转写检索为基础是否足够；表示（wiki）需要维护算力并防止累积错误或幻觉。
- 文末指向两个 cookbook（dreaming 与 knowledge wikis）。

## 外部引用核对（我做的）

- 「Letta 与 UC Berkeley 研究者提出 sleep-time compute」：arXiv:2504.13171 *Sleep-time Compute: Beyond Inference Scaling at Test-time*（2025-04-17；作者 Kevin Lin、Charlie Snell、Yu Wang、Charles Packer、Sarah Wooders 等）✓ 与文章口径一致；论文本身本次未读。
- 其余数字与模型名（Opus 5 / Sonnet 5 / Opus 5.5 / Sonnet 5.5 / Fable 5、Claude Duet、Stainless 收购）均按文章原样转述，**未独立核实**。

## 对 wiki 的映射

- [Agent 记忆与 dreaming](../../wiki/concepts/agent-memory-and-dreaming.md)
- [Claude Cookbook：用动态工作流做 dreaming](../../wiki/entities/claude-cookbook-dreaming-dynamic-workflow.md)
