# SpiderBot 官方实现

> 来源归档（repo）

- **链接：** <https://github.com/ERC-BPGC/SpiderBot>
- **项目页：** <https://erc-bpgc.github.io/SpiderBot/>
- **关联论文：** <https://arxiv.org/abs/2609.26989>
- **核查日期：** 2026-10-03
- **开放状态：** 已开源：机械设计、mjlab 训练、Sim2Sim 与硬件部署；入口已按 README 核查，未实际运行机器人。

## 运行入口

- `uv run train Mjlab-Velocity-Rough-Spiderbot`：mjlab 仿真训练；另有 Flat 任务。
- `sim2sim/test.py`：读取匹配的 ONNX 与 MuJoCo XML，默认策略路径是占位值。
- `sim2real/hardware_deploy.py`：键盘控制；`hardware_deploy_fc.py`：固定速度命令。
- `sim2real/scservo_sdk/`：硬件舵机 SDK；需核对舵机 ID、校准、限位、策略输入和动作参数。
- README 列出已有 ONNX 文件；训练 checkpoint 到 ONNX 的衔接不能用默认占位路径代替。

## 对 wiki 的映射

- [Spiderbot](../../wiki/entities/paper-spiderbot-hexapod-open-source.md)
