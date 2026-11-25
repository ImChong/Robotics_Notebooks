# Loop Behind 官方 vLLM 插件｜Official vLLM Plugin

- **类型：** repo
- **代码：** <https://github.com/XijieGo/LoopBehind>
- **版本：** `a10c12a14a64b9b10c667796f6b7aeb10ffbbc70`
- **许可证：** Apache-2.0
- **论文：** <https://xijiegong.com/loopbehind.pdf>
- **论文归档：** [作者预印本](../papers/loop-behind.md)
- **入库日期：** 2026-11-25
- **核对范围：** 固定版本的 README、`pyproject.toml`、`activate` 和插件注册、overlay 构建与补丁、待完成状态与自定义算子、速度配置及结果、基准说明和参考实现入口；静态源码走读，未执行 GPU 基准。

## 运行入口

- [插件 README](https://github.com/XijieGo/LoopBehind/blob/a10c12a14a64b9b10c667796f6b7aeb10ffbbc70/vllm-plugin/README.md)：入口包括 `loop-behind doctor`、Python `activate(...)` 和 `vllm serve`。论文速度协议采用 vLLM 0.26.0；其他版本需验证补丁兼容性。
- [激活与覆盖层](https://github.com/XijieGo/LoopBehind/tree/a10c12a14a64b9b10c667796f6b7aeb10ffbbc70/vllm-plugin/src/loop_behind)：先激活再导入 vLLM。覆盖层生成补丁副本，改变调度输出、调度器、CUDA 图分发、GPU model runner 和 Dory 前向；不改已安装 vLLM 的原文件。
- [补丁实现](https://github.com/XijieGo/LoopBehind/blob/a10c12a14a64b9b10c667796f6b7aeb10ffbbc70/vllm-plugin/src/loop_behind/overlay/patches.py)：调度器给出请求内各循环阶段的行数，runner 组织槽位及新行掩码；稀疏路径的 `dory_lb_begin_sparse` 和 `dory_lb_finish_sparse` 负责读写双缓冲状态，新行才走输入与输出计算。
- [速度基准协议](https://github.com/XijieGo/LoopBehind/blob/a10c12a14a64b9b10c667796f6b7aeb10ffbbc70/vllm-plugin/docs/BENCHMARK_PROTOCOL.md)：仓库已包含 Dory 的 LB-2/3/4、Self-Spec 基线、单 H100 配置/结果与参考检查。两类解耦干预、Dory 12 项任务完整评测及 Huginn/Ouro 分析流水线尚未在该提交公开，不能用插件速度检查代替完整论文复现。

## 对 wiki 的映射

- [Loop Behind 详情](../../wiki/entities/paper-loop-behind.md)
