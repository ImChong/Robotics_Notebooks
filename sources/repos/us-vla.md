# US-VLA 官方实现

> 来源归档（repo）

- **链接：** <https://github.com/VMVLab/US-VLA>
- **关联论文：** <https://arxiv.org/abs/2608.16074>
- **核查日期：** 2026-10-03
- **开放边界：** 训练与 WebSocket 推理代码公开；数据部分公开，真机部署循环需自行实现。

## 运行入口

- `scripts/compute_norm_stats.py`：`pi05_ur_usfm_fusion` 数据统计。
- `scripts/train.py`：USFM 编码与超声融合的 LoRA 微调。
- `scripts/serve_policy.py`：checkpoint 服务；`openpi-client` 查询动作 chunk。
- 输入：6 维 TCP 状态、侧视 RGB、腕部 RGB、超声图、临床目标 prompt；输出：6 维绝对 TCP 位姿目标。
- README 明示 USFM 权重另行下载，真机控制与图像采集不包含在仓库内。

## 对 wiki 的映射

- [US-VLA](../../wiki/entities/paper-us-vla-ultrasound.md)
