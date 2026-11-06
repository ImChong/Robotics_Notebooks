# FlashProve: an open-source library for provable inference（可证明推理开源库）

- **机构：** Anthropic、EPFL 等
- **版本：** 2026 年 11 月技术报告，共 23 页
- **原文：** https://www.anthropic.com/document/flashprove-technical-report
- **项目：** https://www.anthropic.com/research/provable-inference
- **代码：** https://github.com/anthropics/FlashProve
- **项目节点：** [FlashProve](../../wiki/entities/flashprove.md)
- **核对范围：** 2026 年 11 月版本：摘要、正文第 1–7 节及关于事后验证与数值差异的附录；核对官方 GPU 示例入口与目标运行脚本。未独立审计 Lean 证明及 prover/verifier 内部实现。
- **通读：** 已完成

## 已核实要点

- 系统先对模型权重作隐藏式承诺，再证明给定输入对应的输出符合承诺模型；Lean 对协议性质的分析基于随机预言机模型，C++ 实现与 Lean 规格的等价性尚未形式化证明。
- 多请求共享权重编码检查；分段处理限制 GPU 驻留轨迹大小。link proofs 将新随机编码绑定到同一权重，延长承诺的可用寿命。
- 实验使用 4-bit Llama 4 Maverick 转换的整数模型；1024 token 请求的每个证明检查下一个 token 的贪心选择。
- 事后证明只能事后检测错误，日志若不可信也可能掩盖用户实际收到的输出；浮点服务与整数模型的输出可能不同。


- 形式化分析基于随机预言机模型；报告明确批量请求的零知识性质尚未形式化证明。
- 示例按 `quickstart-gpu.sh`、`run_target_example.sh`、`cp_commit`、`cp_prove`、`cp_verify` 顺序运行；实际验证以本机实验为准。
