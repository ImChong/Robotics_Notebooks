# Zero to SGLang / 从零手搓 SGLang：Datawhale × RadixArk 课程一手资料

- **类型：** course（中英双语 LLM 推理系统教程）
- **课程仓库：** https://github.com/datawhalechina/zero-to-sglang
- **中文在线阅读：** https://zero-to-sglang.datawhale.cc/ch/
- **配套视频：** https://www.bilibili.com/video/BV1Dg4w6tExD/
- **PDF / Releases：** https://github.com/datawhalechina/zero-to-sglang/releases
- **入库日期：** 2026-11-18
- **版本：** 2026-11-18 查阅 GitHub 默认分支；官方 README_zh.md 记载截至 2026-11-10 中英文章全部完成，未锁定具体 commit SHA。
- **核对范围：** 通读仓库 README_zh.md 的课程介绍、完整大纲、学习要求、仓库结构和资源声明；通读 Part II 第 2 章《一个请求的旅程》及其配套代码走读；对照 mini-sglang 官方 docs/structures.md 核对请求链路。**本次未逐篇通读整套课程全部章节，也未跑 GPU 实验。**
- **通读：** 已完成

## 官方课程结构与核心知识

| 部分 | 原文重点 | 推荐输出 |
|---|---|---|
| Part 0 开课之前 | 开源贡献伦理、启动首个 SGLang 服务（Qwen3-0.6B） | 理解环境与服务入口 |
| Part I 基础 | Transformer、自回归、prefill/decode、GPU/Roofline、KV Cache、TTFT/TPOT/ITL/Goodput | 能解释推理瓶颈与测量口径 |
| Part II 自建 Mini SGL | 前向生成、KV Cache、HTTP 并发、Continuous Batching、Paged KV Cache、RadixAttention、多进程与张量并行、投机解码 | 能定位 tokenizer/scheduler/engine 的输入输出 |
| Part III 进阶 | Attention backends、量化、分层缓存、DP Attention / EP / PP、Prefill-Decode 分离 | 能用负载特征解释优化取舍 |
| Part IV 贡献 | Cookbook 部署、Profiling/Trace、PR 流程 | 可复现且可审查的性能改动 |

## 原文技术核对：一个请求如何流转

Part II 第 2 章将 **API Server → tokenizer → scheduler/engine → detokenizer → API Server** 串成一条请求链。请求 UID 随消息流转；scheduler 决定队列、batch 和 KV Cache 资源，engine 执行模型前向与采样。decode 可持续产生 token，detokenizer 维护跨 token 的字符解码状态，API 使用 SSE 流式返回；取消时需向调度侧传递 abort 并回收资源。该章节配套代码走读引用的是 **独立上游仓库** `sgl-project/mini-sglang` 的 `server/launch.py`、`server/api_server.py`、`tokenizer/server.py`、`scheduler/scheduler.py`、`engine/engine.py` 和 `tokenizer/detokenize.py`，而不是声称本课程仓库拥有同名可执行工程。

## 发布与访问边界

- 官方 README 将全套讲解视频标为 **预计 2026 年 11 月底完成**；其 B 站链接是可追踪视频入口，不能推定所有视频已上线，也未核对单条视频的完整内容。
- README 标有中文和英文 PDF 的 `releases/latest/download/...` 地址；**2026-11-18 GitHub Releases 页面显示尚无 Release**，因此不能宣称该 PDF 当前可直接下载。
- README 指向 `/ch/` 在线站；本次在线站抓取未成功，课程目录和技术事实改以仓内的原始 Markdown 核实。
- README 授权标识为 **CC BY-NC-SA 4.0**；不得未经许可将课程正文整篇镜像进本知识库。
- 本条只对上文列出的主要原文核对范围给出“通读已完成”，不代表审核了所有章节、视频或复现实验。

## 与机器人研究的关系

本课是 LLM/VLM 推理基础设施材料，不是机器人控制方法；适合作为 VLA/WAM 语义层服务端的缓存、吞吐和响应时延知识前置。不能由服务器端 TTFT/吞吐成绩直接推断真机动作环的端到端实时性。

## 一手资料

- [官方中文 README（完整课程目录）](https://github.com/datawhalechina/zero-to-sglang/blob/main/README_zh.md)
- [Part II 第 2 章：一个请求的旅程](https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/第2章_一个请求的旅程.md)
- [Part II 第 2 章：源码走读](https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/第2章_一个请求的旅程_代码.md)
- [Mini-SGLang 官方系统结构](https://github.com/sgl-project/mini-sglang/blob/main/docs/structures.md)
- [发布页面](https://github.com/datawhalechina/zero-to-sglang/releases)
- [Wiki 独立节点](../../wiki/entities/zero-to-sglang.md)
