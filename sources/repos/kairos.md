# kairos-agi/kairos

> 来源归档

- **标题：** Kairos（官方世界–动作模型实现）
- **类型：** repo
- **组织：** kairos-agi（权重镜像常落在 **ACERobotics** / Ace Robotics）
- **代码：** <https://github.com/kairos-agi/kairos>（Apache-2.0；旧仓 `kairos-agi/kairos-sensenova` **301 →** 本仓）
- **技术报告：** <https://arxiv.org/abs/2606.16533>（v3：*Regret-Aware Native World-Action Model Stack*）
- **平台页：** <https://kairos.acerobotics.com>
- **权重：** <https://huggingface.co/kairos-agi>；集合 <https://huggingface.co/collections/kairos-agi/kairos30>；ModelScope <https://modelscope.cn/collections/kairos-team/kairos30>
- **入库日期：** 2026-07-24（承接 2026-06-18 的 `kairos_sensenova` 索引）
- **一句话说明：** Kairos **原生世界–动作模型栈** 官方实现：`kairos/`（DiT / MoT / pipeline）、`examples/inference.sh`（T2V/I2V/TI2V）、`benchmarks/{libero_plus,robotwin}`（可执行动作预测评测）；配套 **Kairos3.1** 具身生成与 LIBERO-Plus / RoboTwin 权重。

## 开源状态核查（2026-07-24）

| 项 | 状态 |
|----|------|
| 训练/推理代码 | **已开源**：推理入口 `examples/inference.sh` + `kairos/pipelines/`；WAM 评测 `benchmarks/libero_plus`、`benchmarks/robotwin` |
| 权重 | **已开源**：HF/ModelScope；README 列 Kairos3.1-4B-robot-480P、RoboTwin2.0、LIBERO-plus、720P 等 |
| Docker | GHCR 镜像仍可能带旧名标签 `ghcr.io/kairos-agi/kairos-sensenova:v0.0.1`（A800/A100、`-rtx5090`、`-metax`） |
| 真机闭环 regret 验证代码 | 报告定位为 **未来工作**；当前公开以生成推理 + 仿真 WAM 评测为主 |

## 关键目录 / 入口

| 路径 | 作用 |
|------|------|
| `examples/inference.sh` / `examples/inference.py` | 单卡推理入口；配合 `examples/example_{t2v,i2v,ti2v}*.json` |
| `examples/multi_gpu_inference.sh` | 多卡推理 |
| `kairos/configs/kairos_4b_*.py` | 4B / DMD 蒸馏 / video-only / WAM 配置 |
| `kairos/pipelines/kairos_embodied_pipeline*.py` | 具身生成管线（含 DMD） |
| `kairos/pipelines/kairos_embodied_wam_pipeline.py` | WAM 管线 |
| `kairos/modules/dits/` | LinearDiT / MoT / mask utils |
| `benchmarks/libero_plus/` | LIBERO-Plus WAM 评测与 `run.sh` |
| `benchmarks/robotwin/` | RoboTwin 2.0 WAM 评测 |
| `docs/QUICKSTART.md` | 依赖下载与配置细节 |
| `docker/` | 容器与 requirements |

## Model Zoo（README，2026-07-02）

| 权重（HF 卡 ID） | 用途 |
|------------------|------|
| `kairos-agi/Kairos3.1-4B-robot-480P` | 具身世界生成基础模型（480P） |
| `kairos-agi/kairos-4B-robot-RoboTwin2.0` | RoboTwin 2.0 可执行动作预测 |
| `kairos-agi/kairos-4B-robot-LIBERO-plus` | LIBERO-Plus 可执行动作预测 |
| `kairos-agi/kairos-sensenova-4B-720P` | 720P 高清生成 |

> 注：上述 `kairos-agi/...` URL 在 Hugging Face 上可 **重定向到 `ACERobotics/...`**；下载时以当前解析目标为准。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Kairos（原生世界–动作模型栈）](../../wiki/entities/paper-kairos-native-world-model-stack.md) | 实体归纳页：regret / CEDC / SWA·DSWA·GLA / WAM / 开源时序 |
| [Generative World Models](../../wiki/methods/generative-world-models.md) | **4B 级** 具身视频 WM + **线性可扩展** DiT 推理对照 |
| [World Action Models](../../wiki/concepts/world-action-models.md) | **Joint WAM**：Video/Action 双 DiT + action-only / joint |
| [Cosmos 3](../../wiki/entities/cosmos-3.md) | 同为 Physical AI 世界模型平台；Cosmos 偏 16B/64B，Kairos 偏 **4B 边缘部署 + 原生 CEDC** |
| [HomeWorld](../../wiki/entities/paper-homeworld-whole-home-scene-generation.md) | **品牌名易混**：静态全屋 3D ≠ 本仓视频/WAM |

## 对 wiki 的映射

- 技术报告：[`sources/papers/kairos_arxiv_2606_16533.md`](../papers/kairos_arxiv_2606_16533.md)
- 平台页：[`sources/sites/kairos-acerobotics.md`](../sites/kairos-acerobotics.md)
- 旧仓索引（重定向说明）：[`sources/repos/kairos_sensenova.md`](./kairos_sensenova.md)
- 沉淀 **[`wiki/entities/paper-kairos-native-world-model-stack.md`](../../wiki/entities/paper-kairos-native-world-model-stack.md)**
