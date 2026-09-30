# GAE 项目页（jiah-cloud.github.io/GAE.github.io）

> 来源归档

- **标题：** GAE · Geometry-Native World Generation
- **类型：** site / project-page
- **URL：** <https://jiah-cloud.github.io/GAE.github.io/>
- **论文：** <https://arxiv.org/abs/2609.24981>（PDF：<https://arxiv.org/pdf/2609.24981>）
- **Hugging Face 论文卡：** <https://huggingface.co/papers/2609.24981>
- **代码：** <https://github.com/TencentARC/GAE-GeometricAutoEncoder> — 归档见 [`sources/repos/gae-geometric-autoencoder.md`](../repos/gae-geometric-autoencoder.md)
- **权重（推理）：** <https://huggingface.co/TencentARC/GAE-D64-1B>
- **入库日期：** 2026-09-30
- **一句话说明：** Tencent ARC × HKUST 等提出的 **Geometry-Native Autoencoder（GAE）**：把 DA3 四级几何特征压成 **64/128 通道** 潜空间，同一生成态联合解码 **RGB、深度、相机与点云**；项目页含 81 视角 demo、latent 对照与 text-to-image 3D 示例。

## 开源状态（步骤 2.5，2026-09-30）

| 项 | 状态 |
|----|------|
| 项目页 | **已发布** — 交互 demo、81 场景视频、latent 六方对照、Camera Studio 说明 |
| GitHub | **已开源** — 训练 / 推理 / demo / eval CLI；[`TencentARC/GAE-GeometricAutoEncoder`](https://github.com/TencentARC/GAE-GeometricAutoEncoder) |
| Hugging Face 权重 | **已发布** — `TencentARC/GAE-D64-1B`（demo 默认）；首次运行自动缓存至 `ckpts/` |
| 许可 | **Tencent 自定义学术许可**（`LICENSE.txt`）：**仅限学术用途**，禁止非学术 / 商业 / 生产用途 |
| 规模 | 论文叙事约 **1B** 参数 flow 骨干 + codec；DA3-GIANT 首次使用时从 Hub 拉取 |

**核查结论：** 项目页 **Code on GitHub** 与 README 一致；复现入口为 `bash scripts/demo/run_demo.sh` 或 `scripts/demo/generate.py`。**非 Apache/MIT** — 商用前需单独授权。

## 页面要点（策展）

- **主张：** 把 3D 归纳偏置放进 **生成态本身**，而非在 appearance-only latent 外再挂几何约束。
- **数字：** 3072→128 通道（约 24×）；条件数 κ 约 10⁸→10²；RealEstate10K / DL3DV 上 matched DiT 的 FVD 分别 **−12.7% / −23.1%**；RealEstate10K 相机轨迹误差约 **减半**。
- **两 operating point：** **GAE-64** 偏紧凑生成与轨迹一致；**GAE-128** 偏重建与跨视角对应。
- **任务：** text-to-image、相机控制视频、参考条件新视角；同一 flow 骨干 + 条件 dropout（文本 / metric Plücker rays / clean reference latents）。

## 对 wiki 的映射

- 论文摘录：[`sources/papers/gae_arxiv_2609_24981.md`](../papers/gae_arxiv_2609_24981.md)
- 仓库：[`sources/repos/gae-geometric-autoencoder.md`](../repos/gae-geometric-autoencoder.md)
- 实体页：[`wiki/entities/paper-gae-geometry-native-autoencoder.md`](../../wiki/entities/paper-gae-geometry-native-autoencoder.md)
