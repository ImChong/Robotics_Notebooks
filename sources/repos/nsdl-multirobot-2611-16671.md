# NSDL: Neuro-Symbolic Division of Labor for Multi-Robot Task Planning 官方源码核对

- **类型：** 官方源码仓
- **代码：** https://github.com/Xincheng-Tan/NSDL
- **论文：** https://arxiv.org/abs/2611.16671
- **公开运行入口：** `python scripts/eval_lnsdl_agentic_viki_l2.py --level LOCUS --split ood_m --model-size 7b`
- **相关模块：** `nsdl.tool_runtime 与 nsdl.planning_tool`
- **运行产物：** 联合计划与 benchmark 得分
- **核对范围：** 官方 README、该仓库公开入口脚本与关键模块的静态核查；外部数据和模型依赖未执行。
- **通读：** 已完成
- **wiki：** [论文独立详情](../../wiki/entities/paper-nsdl-multirobot-2611-16671.md)
- **原文：** [论文来源](../papers/nsdl-multirobot-2611-16671.md)

依赖任务描述语言、工具输出语义和符号评价器；实际机器人感知/控制/通信约束未在该符号回放中完整验证。
