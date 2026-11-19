# Robot Weight Debugging 官方源码与运行入口

- **类型：** repo
- **代码：** https://github.com/jushengzhang/robot-weight-debugging
- **论文：** https://arxiv.org/abs/2611.09376
- **核对日期：** 2026-11-19
- **核对范围：** 官方 README、已公开的代码入口/脚本及许可说明；仅静态阅读，未执行训练、模型推理或真机实验。
- **通读：** 已完成
- **wiki：** [独立论文节点](../../wiki/entities/paper-robot-weight-debugging-2611-09376.md)

## 开源与复现边界

`apply_sparse_rows.py` 通过原始 checkpoint memmap、recipe 与 npz row IDs 一致性、原始行值检查后生成覆盖 `.npy`，不会修改原 checkpoint，更不会运行模型；`evidence/audit_new_results.py` 读取已发布 JSON/JSONL episode 配对结果生成统计。仓库 Apache-2.0。

- 官方入口：python edits/pi05-language-c/apply_sparse_rows.py <checkpoint> --dry-run；python evidence/audit_new_results.py（审计已有结果）
