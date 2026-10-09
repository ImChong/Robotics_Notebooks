# X-Tokenizer

> 来源归档（国内具身开源全景；2026-10-09 按仓库与 HF 复核）

- **标题：** X-Tokenizer
- **类型：** repo
- **机构：** 自变量机器人（X Square Robot）
- **链接：** https://github.com/X-Square-Robot/X-Tokenizer
- **许可：** Apache-2.0（仓根 `LICENSE`；`pyproject.toml` 包名 `x-tokenizer` v1.0.0）
- **权重：** https://huggingface.co/x-square-robot/X-Tokenizer（`xtokenizer.pth`）
- **论文：** [arXiv:2606.14752](https://arxiv.org/abs/2606.14752)
- **项目页：** https://x2robot.com/pages/x-tokenizer ；镜像 https://x-square-robot.github.io/X-Tokenizer_projectPage/
- **分类：** VLA/操作模型（动作分词器）
- **入库日期：** 2026-09-06（复核 2026-10-09）
- **一句话说明：** 26 维双臂末端 + 底盘 + 升降 + 头部动作的残差 VQ 分词器推理包：4 倍时间压缩，每个潜步 4 级 × 2048 码，`chunk_size ∈ [8, 64]` 免重载，可输出 `time_major` / `quantizer_major` 两种 token 排布。来源一：[国内具身开源全景](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)。
- **沉淀到 wiki：** [`wiki/entities/cn-os-x-tokenizer.md`](../../wiki/entities/cn-os-x-tokenizer.md)

## 仓库结构（2026-06-18 commit `9733990`）

| 路径 | 作用 |
|------|------|
| `xtokenizer/api.py` | `XTokenizer.from_pretrained`；`encode` / `decode` / `reconstruct` / `encode_from_absolute` / `decode_to_absolute` 及 `_numpy` 变体 |
| `xtokenizer/model/` | `encoder.py`、`rvq.py`、`decoder.py`、`tokenizer.py`、`position_encoding.py` |
| `xtokenizer/data/` | 动作布局（delta / absolute 互转，6D 旋转 SO(3) 合成）、归一化、统计量读写 |
| `xtokenizer/tools/` | `StatisticsAccumulator` 与 `xtokenizer-compute-statistics` CLI |
| `xtokenizer/configs/robot_types.yaml` | 18 个规范本体槽位（0 Unknown，1 X2Arm 为 `robot_type=None` 默认值，其后为 Franka、UR5、Piper、Viperx、ARX5、WidowX、GoogleRobot、AgiBot、UMI、Realman、Ark、R1Lite、AlphaBot、MMK2、Leju、A2D） |
| `examples/` | `01_encode_decode.py`（Case A 六步管线），`02_compute_statistics.py`（Case B 统计量），两条合成 `.npz` |

## 开源状态

- **部分开源**：推理侧 encode/decode 库和预训练权重已公开（Apache-2.0）。
- **未随仓发布**：预训练代码（三个语义辅助头与训练循环）、下游 Wall-OSS 共训代码、2.4M 轨迹预训练语料、训练用归一化统计量。
- 关键提示：README 要求自算统计量时 `--chunk-size 32`（与发布 checkpoint 的训练口径一致），否则 `delta_001` 量级漂移，归一化会严重截断；推理时 `T ∈ [8, 64]` 均可。

## 对 wiki 的映射

- [wiki/entities/cn-os-x-tokenizer.md](../../wiki/entities/cn-os-x-tokenizer.md)
- 项目页归档：[sources/sites/x2robot-x-tokenizer.md](../sites/x2robot-x-tokenizer.md)
