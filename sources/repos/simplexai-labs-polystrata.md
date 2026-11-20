# simplexai-labs/PolyStrata：多引擎本地 MoE 推理

- **类型：** repo
- **原始链接：** https://github.com/simplexai-labs/PolyStrata
- **入库日期：** 2026-11-20
- **核对范围：** README 全文、polystrata.py 启动 CLI 源码、serve/ 目录入口
- **通读：** 已完成

## 原始资料要点（归纳，非全文转载）

- 多模型本地推理框架；将 MoE 常用专家分层放在 GPU 显存、CPU RAM 或 SSD，目标是较小显存运行较大模型。
- 官方明确 Qwen3.8-Flash-Next 使用 Niko1221/Strata 引擎，其他支持模型由 PolyStrata 自有引擎运行，二者不能等同。
- polystrata.py 发现安装模型、按配置启动 serve/server.py --engine strata、轮询 /health，亦可复用现有本地服务并切换模型。
- 提供面向编码助手/API 的本地服务；它本身不是长期记忆或主动代理，性能数字是上游硬件特定测试。

## 对 wiki 的映射

- [simplexai-labs/PolyStrata：多引擎本地 MoE 推理](../../wiki/entities/polystrata-local-inference.md)
