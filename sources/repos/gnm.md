# GNM（Generative aNthropometric Model，Google 官方仓库）

> 来源归档

- **标题：** GNM — Generative aNthropometric Model and Ecosystem
- **类型：** repo
- **机构：** Google
- **链接：** <https://github.com/google/GNM>
- **子包入口：** <https://github.com/google/GNM/tree/main/gnm/shape>（**GNM Head**）
- **许可证：** Apache License 2.0
- **入库日期：** 2026-07-20
- **一句话说明：** Google 开源的 **参数化人体统计模型生态**；首期发布 **GNM Head**——高保真 3D 头脸统计模型，解耦身份、表情、头姿与平移，含眼球/牙齿/舌头等内部解剖，并提供 NumPy / JAX / PyTorch / TensorFlow 多后端与语义参数采样。
- **沉淀到 wiki：** [GNM Head](../../wiki/entities/gnm-head.md)

---

## 生态路线图（README ingest 快照，2026-07-20）

| 包 | 状态 | 说明 |
|----|------|------|
| **GNM Head** (`gnm/shape`) | **已开源** | 参数化 3D 头脸几何；v3.0 模型资产（`.npz`）、纹理、语义采样器（`.h5`）随仓发布 |
| **全身 / 感知栈** | 路线图 | README 称将陆续发布更完整统计模型与感知分析技术；截至入库日仓库仅含 Head |

## GNM Head 技术要点

| 项 | 说明 |
|----|------|
| **模型族** | 3D Morphable Model（3DMM）/statistical shape model |
| **解耦参数** | Identity、Expression（blendshape）、Head Pose（含眼球）、Translation |
| **几何细节** | 密集网格：皮肤、眼球、牙齿、舌头 |
| **语义采样** | `ExpressionSampler`（如 happy / surprise）、`IdentitySampler`（性别、族裔等属性） |
| **多框架** | `gnm_numpy.py`（主实现）、`gnm_jax.py`、`gnm_pytorch.py`、`gnm_tensorflow.py` |
| **Python** | 测试环境 **Python 3.13**；`pip install -e .` 可选 `[jax]` / `[pytorch]` / `[all,dev]` |
| **演示** | `gnm/shape/demos/` 交互 notebook；`gnm_colab_viewer.py` 供 Colab 3D 可视化 |

## 开源状态

- **已开源**：GNM Head 代码、v3.0 模型数据、语义采样权重与 CI（Linux / macOS / Windows）均在主分支公开；**BibTeX 引用条目 README 标注 coming soon**。
- **未发布**：全身 GNM、完整感知栈与配套论文预印本链接（截至 2026-07-20 仓内未列 arXiv）。

## 对 wiki 的映射

- 主实体页：**`wiki/entities/gnm-head.md`**
- 人体网格对照：**`wiki/entities/sam-3d-body.md`**（全身 HMR）、**`wiki/entities/paper-face-anything-4d-face-reconstruction.md`**（单目 4D 脸重建管线）
