# Robot Weight Debugging 官方源码与运行入口

- **类型：** repo
- **代码：** https://github.com/jushengzhang/robot-weight-debugging
- **论文：** https://arxiv.org/abs/2611.09376
- **核对日期：** 2026-11-28
- **发布：** [v0.1.0](https://github.com/jushengzhang/robot-weight-debugging/releases/tag/v0.1.0)；开发分支 HEAD 校验时为 [d9cf19d](https://github.com/jushengzhang/robot-weight-debugging/tree/d9cf19d973c63203b68e25c94a2831d680779fb6)。
- **核对范围：** 官方 README、v0.1.0 发布页、稀疏行补丁与公开审计入口及许可说明；仅静态阅读，未执行训练、模型推理或真机实验。
- **通读：** 已完成
- **wiki：** [独立论文节点](../../wiki/entities/paper-robot-weight-debugging-2611-09376.md)

## 开源与复现边界

`apply_sparse_rows.py` 通过原始 checkpoint memmap、recipe 与 npz row IDs 一致性、原始行值检查后生成覆盖 `.npy`，不会修改原 checkpoint，更不会运行模型；`evidence/audit_new_results.py` 读取已发布 JSON/JSONL episode 配对结果生成统计。仓库 Apache-2.0。

- 官方入口：python edits/pi05-language-c/apply_sparse_rows.py <checkpoint> --dry-run；python evidence/audit_new_results.py（审计已有结果）

## 发布范围与审计约定

- 首发包括会话记录（16 次）、执行工具记录、已发布逐回合记录、65 行稀疏配方及补丁。基础模型与仿真资产由上游提供。原版代码 Apache-2.0，模型衍生载荷、外部权重和数据保留各自条款。
- `audit_new_results.py` 对回合键作交集，拒绝重复键；不同控制组单列有效对与异常数，某些对照额外剔除有错误的任一侧。不把输出目录中存在的候选文件等同于编辑器的保留意图。
- `apply_sparse_rows.py` 校验张量形状、行索引及原值；`--dry-run` 只核对。写出流程用临时文件再以硬链接安装新文件，拒绝覆盖已存在的目标；不原位改动基础检查点，也不会运行策略。
- 截至核对日作者说明 v2 和真机实验仍在进行；本次未执行仿真、模型推理或真机。
