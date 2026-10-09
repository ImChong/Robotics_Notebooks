# RFPO 官方仓库归档

- **仓库：** https://github.com/AIGeeksGroup/RFPO
- **项目页：** https://aigeeksgroup.github.io/RFPO/
- **对应论文：** https://arxiv.org/abs/2610.10453
- **仓库 README 项目名：** rfpo_new_method / FPO++ experiment code
- **核对日期：** 2026-10-09

## 仓库内容与入口

README 将仓库定位为基于 Isaac Lab 的 locomotion 与 manipulation 实验代码，说明其中的 Go2/G1 方法更改，并要求初始化递归 submodule、分别安装各实验目录环境。文档列出 Go2/G1 训练入口，以及用 64、32、16、8、4、1 等积分步数评估 checkpoint 的流程；manipulation 相关脚本包括预训练、在线微调与 checkpoint evaluation。

## 复现边界

- README 明确写明 deployment packages 未包含；仓库不能直接视为完整的机器人端运行栈。
- 论文报告 Spot 和 H1 等评测不代表仓库已公开这些平台的实验配置。
- 具体命令与依赖应以仓库当前 README、各子目录说明和 submodule 状态为准；这里保留概要，不复制易变化的安装命令。
