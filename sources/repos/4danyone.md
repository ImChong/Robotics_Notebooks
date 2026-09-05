# 4DAnyone（ant-research/4DAnyone）

- **URL：** <https://github.com/ant-research/4DAnyone>
- **组织：** ant-research（蚂蚁 / Robbyant）
- **许可证：** Apache-2.0（代码）；权重条款以 HF `LICENSE` / `licenses/` 为准，资产分项授权
- **权重：** <https://huggingface.co/AntResearch/4DAnyone>
- **项目页：** [4danyone-github-io](../sites/4danyone-github-io.md)
- **关联论文：** [4danyone_arxiv_2608_20335](../papers/4danyone_arxiv_2608_20335.md)
- **子模块：** `third_party/GVHMR` → [GVHMR](./gvhmr.md)

## 一句话说明

单目随意视频 → 多视角一致视频 → 下游 3DGS/4DGS。入口是 `inference.py`；默认 **4DAnyone-Turbo**。论文里的 4DGS 用 FreeTimeGS；仓内开源 4DGS 重建截至 2026-09-05 仍在 Roadmap。

## 运行入口

| 路径 | 作用 |
|------|------|
| `inference.py` | 生成目标视角视频；`views_per_layer` 须被 4 或 6 整除 |
| `scripts/download_model.py` | 拉 HF 权重（也可首次推理自动下） |
| `scripts/download_smplx.py` | SMPL-X 资产 |
| `third_party/GVHMR` | 单目 3D 骨架 |
| `docs/nerfstudio.md` | 已支持的 3DGS 重建 |

## 工程数字（README，2026-09-05）

- 峰值显存 **< 24 GB**（可上消费级 4090）。
- 单条 121 帧视频约 **27 s**（RTX 4090）。
- Turbo 相对 Base 去噪 **5.58×**。
- 输入约定：单人、全身或半身、相机不要大动、≥1080p、9:16、至少 121 帧。

## 交叉链接

- [4DAnyone 论文实体](../../wiki/entities/paper-4danyone.md)
- [GVHMR](../../wiki/entities/gvhmr.md)
