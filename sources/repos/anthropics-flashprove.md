# anthropics/FlashProve

> 官方源码来源归档

- **仓库：** https://github.com/anthropics/FlashProve
- **版本：** README 标记 0.16.0；核查日期 2026-11-04
- **许可：** Apache-2.0；研究原型，仓库注明不维护
- **博客：** [Anthropic 可证明推理](../sites/anthropic-provable-inference.md)
- **实体：** [FlashProve](../../wiki/entities/flashprove.md)

## 已核对入口

- `examples/quickstart-gpu.sh` 检查环境，构建 verifier 与 prover，调用 `prover/examples/run_target_example.sh target_small_f2_roomy`。
- `run_target_example.sh` 准备配置，调用 `cp_commit`（合成权重）、准备请求、`cp_prove`，最后以 commitment ID 调用 `cp_verify`；验证失败不公开输出。
- README 的存储证明验证入口是 `verifier/build.sh` 和 `bin/cp_verify`。真实使用时承诺 ID 必须从可信方获取，而非从被验证的服务方获取。
- 小模型 GPU 路径要求 H100、指定 CUDA 13.2 等环境；完整 Maverick 例子另需获取权重。未在本环境运行。

已阅读上述脚本的入口及调用，尚未系统审计 prover/verifier 内部和 Lean 定理。
