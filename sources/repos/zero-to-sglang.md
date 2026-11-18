# zero-to-sglang：课程仓库结构与参考实现的区别

- **类型：** repo（课程文档源码仓库，非独立的推理引擎实现）
- **仓库：** https://github.com/datawhalechina/zero-to-sglang
- **入库日期：** 2026-11-18
- **版本：** main 分支 2026-11-18 查阅；未锁定 commit SHA。
- **核对范围：** 通读 README_zh.md 的项目结构、学习路线、贡献与许可证说明；查阅 `course-material/ch/part2/` 的文件布局，并通读《一个请求的旅程》概念篇及代码走读；对照上游 `sgl-project/mini-sglang/docs/structures.md` 的组件说明。未逐一验证所有章节内代码片段或跑完整课程实验。
- **通读：** 已完成

## 仓库定位与开源边界

本仓库公开 `course-material/ch/` 与 `course-material/eng/` 课程 Markdown、`docs/.vitepress/` 站点配置、`community/` 社区内容，以及文档构建脚本。官方 README 的项目结构不列出可以独立启动的课程版推理引擎 Python 包；Part II 的“手搓”是教学步骤与代码走读，**不能等同于本仓提供可直接启动的 mini-sglang 实现**。

课程注明使用的参考实现另属以下上游仓库：

- [Mini-SGLang](https://github.com/sgl-project/mini-sglang)：`python/minisgl/server/launch.py`、`server/api_server.py`、`scheduler/scheduler.py` 等真实执行模块；可以根据其官方 README 单独部署。
- [SGLang](https://github.com/sgl-project/sglang)：生产服务框架，代码组织和调度策略比 mini-sglang 更复杂。

## 代码走读证据

根据课程 Part II 第 2 章代码篇：

1. `server/launch.py` 启动 Scheduler、Tokenizer/Detokenizer 子进程与 API Server。
2. API Server 以请求 uid 封装消息，Tokenizer 转为 token ids。
3. Scheduler 使用 `_schedule_next_batch` 选出 prefill/decode batch，通过 `_forward` 调用 Engine。
4. Engine 输出 logits，采样后返回 next token；Detokenizer 将 token 转成增量文本，API Server 经 SSE 返回。
5. 断连通过 AbortMsg / AbortBackendMsg 触发取消和资源清理。

这些模块路径属于上游 `sgl-project/mini-sglang`；`zero-to-sglang` 本仓提供的是讲解与对应的外链代码走读，而非复制了一套可执行源码。因而该课程实体的“源码运行时序图”只示意**有来源的上游参考实现**。

## 复现建议与局限

- 先按课程 Part 0 跑 SGLang 示例，再按 Part II 读/写教学版，最后对照上游实现。
- 记录模型与环境、提示词和生成长度分布，区分 TTFT、TPOT、ITL、吞吐、P95/P99。
- 不要从文档中的教学示例推导 SGLang 本体的实际性能，也不要把推荐的 GPU 环境当作已完成复现实测。

## 对 Wiki 的映射

- [Zero to SGLang / 从零手搓 SGLang](../../wiki/entities/zero-to-sglang.md)

## 一手链接

- [官方课程 README_zh](https://github.com/datawhalechina/zero-to-sglang/blob/main/README_zh.md)
- [Part II 第 2 章源码走读](https://github.com/datawhalechina/zero-to-sglang/blob/main/course-material/ch/part2/第2章_一个请求的旅程_代码.md)
- [Mini-SGLang 官方架构](https://github.com/sgl-project/mini-sglang/blob/main/docs/structures.md)
