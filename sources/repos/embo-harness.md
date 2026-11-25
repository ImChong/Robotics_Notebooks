# Embo-Harness: Towards Understanding Frontier-Model-as-Policy Harness｜源码归档

- **仓库：** <https://github.com/DravenALG/Embo-Harness>
- **论文：** [原文归档](../papers/embo-harness.md)
- **项目页：** [官方项目](../sites/embo-harness.md)
- **许可：** NTU S-Lab License 1.0（README；完整条款待复核）。
- **核对日期：** 2026-11-25
- **核对范围：** README 与下述入口源码；未执行实验，完整调用链待补。
- **通读：** 未完成

## 运行入口

`python scripts/serve_harness.py --config config/harness/robolab.yaml`：读取配置，构造 CodexClient/ClaudeClient，可选 VLAService，构造 HarnessPolicy 后启动 WebsocketPolicyServer。README 另给 RoboLab client 与批量启动脚本；未运行。
