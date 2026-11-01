# Auriga 官方代码仓库

> 来源归档（repo）；本条为待完成的 ingest。

- **标题：** Auriga: An Agent Harness for Embodied Manipulation
- **链接：** https://github.com/lose4578/Auriga
- **项目页：** https://auriga.stelledge.com/
- **技术报告：** https://auriga.stelledge.com/assets/docs/Auriga-Technical-Report.pdf
- **演示：** https://auriga.stelledge.com/demos.html
- **入库日期：** 2026-11-01
- **核查版本：** README blob `b17083541135dc14ca847ff7a22073d3d1234cd9`。

## 已核实信息

官方 README 将 Auriga 描述为具身操作智能体框架：模型在持久 Codex 会话中观察机器人状态、规划并调用工具，动作由统一执行网关管理；默认环境为 RoboDojo 和双臂 ARX X5。README 提供代码、安装与运行入口，仓库声明 MIT 许可；Isaac Sim 等依赖和模型权重不随仓库分发。README 自报 RoboDojo 42 项任务平均 Score 64.0、成功率 57.7%；完整评测条件尚待技术报告核实。

入口示例为 `python -m auriga.apps.gateway --benchmark --task general_pickup ...`，命令实现位于 `auriga/apps/gateway/cli.py`。目前仅读取 README 和 CLI 入口，尚未核实完整运行链路。

## 待完成

项目主页、演示页和技术报告本次无法访问，尚未通读报告或检查完整相关源码；不能据此断定数据开放情况。后续需获取全文、核对实验和局限，阅读运行链路，补充项目实体及两张基于原文和源码的 Mermaid 图，运行仓库质量检查后再将 PR 转为可审阅状态。
