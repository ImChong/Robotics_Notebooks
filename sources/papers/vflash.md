# VFlash 技术报告

- **标题：** VFlash: Fast, Accurate Visual Planning via Flash Primitive Decoding
- **论文：** <https://research.nvidia.com/labs/lpr/vflash/assets/paper/vflash.pdf?v=20261105-2>
- **项目页：** <https://research.nvidia.com/labs/lpr/vflash/>
- **机构：** 英伟达；约翰斯·霍普金斯大学、香港理工大学、普渡大学、香港大学（额外机构）
- **版本：** 20261105-2 PDF；2026-11-06 核对全文、实验与附录。

## 核心摘录

1. §2：冻结目标 VLM，使用标注图元训练轻量 drafter；并行预测整个图元，结构合法时不要求目标 token 一致。文本保留目标验证。
2. §3 表 4：InternA1、2D+3D 变体的 ADE 从 0.153 m 降至 0.062 m。
3. §3 表 6：MolmoSpaces 四类任务平均成功率由原始 WAM 基线的 62.6% 升至 70.2%，但条件编码不同。
4. 附录 E–F：Thor 模型流水线延迟 3552.7 → 703.3 ms；约 25 倍点运动生成速度来自 H100 单次 rollout。

## 对 wiki 的映射

- [VFlash](../../wiki/entities/paper-vflash.md)
