# dexmal/opendm（OpenDM · DM0.5）

> 来源归档（ingest）

- **标题：** OpenDM — DM0.5 开放世界 VLA 官方开源栈
- **类型：** repo
- **组织：** Dexmal（大晓智能）
- **代码：** <https://github.com/dexmal/opendm>（Apache-2.0）
- **技术博客：** <https://www.dexmal.com/blog/dm0.5>（中文）、<https://www.dexmal.com/blog/dm0.5/index_en.html>（英文）
- **权重集合：** <https://huggingface.co/collections/Dexmal/dm05> · ModelScope <https://www.modelscope.cn/collections/Dexmal/DM05>
- **MaaS：** <https://maas.dexmal.com/>
- **入库日期：** 2026-08-08
- **一句话说明：** **OpenDM** 是 Dexmal 为 **DM0.5** 发布的 **训练 / 推理 / 数据注册 / 评测** 开源仓库：提供 **DM05** 基础权重与 LIBERO / RoboTwin2.0 / VLA-Arena / SO101 / Table30v2 等下游 checkpoint，统一经 `script/dm05_launcher.sh` 启动 HTTP 推理（default / TensorRT **fast** backend）与 JSONL SFT。

## 开源状态（项目页 / README 核查）

| 项 | 状态（截至 2026-08-08） |
|----|-------------------------|
| **训练 / 推理代码** | **已开源**（`opendm/` 包 + `playground/` + `script/dm05_launcher.sh`） |
| **基础权重** | **已开源**：[Dexmal/DM05](https://huggingface.co/Dexmal/DM05)（≈5.8B 参数量级；HF `pipeline_tag=robotics`） |
| **下游评测权重** | **已开源**：`DM05-libero`、`DM05-robotwin2`、`DM05-SO101-Pick-Cube`、`DM05-Vla-Arena`、Table30v2 collection |
| **数据集** | LIBERO / RoboTwin2.0 等经 HF 数据集卡与 runner 脚本获取（见 docs） |
| **许可** | Apache-2.0 |

## 公开权重一览（README）

| 模型 | 用途 | Checkpoint |
|------|------|------------|
| **DM05** | 通用预训练底座（微调入口） | [HF](https://huggingface.co/Dexmal/DM05) / [ModelScope](https://modelscope.cn/models/Dexmal/DM05) |
| **DM05-libero** | LIBERO 评测 SFT | [HF](https://huggingface.co/Dexmal/DM05-libero) |
| **DM05-robotwin2** | RoboTwin2.0 generalist | [HF](https://huggingface.co/Dexmal/DM05-robotwin2) |
| **DM05-SO101-Pick-Cube** | SO101 pick-cube LoRA/SFT | [HF](https://huggingface.co/Dexmal/DM05-SO101-Pick-Cube) |
| **DM05-VLA-Arena** | VLA-Arena 评测 | [HF](https://huggingface.co/Dexmal/DM05-Vla-Arena) |
| **DM05-Table30v2** | RoboChallenge Table30 v2 集合 | [HF collection](https://huggingface.co/collections/Dexmal/dm05-table30v2) |

## README 报告基准（与 Pi0 / Pi0.5 / GROOT-N1.7 对照）

| 基准 | 指标 | DM0.5 | 备注 |
|------|------|-------|------|
| LIBERO | SR | **99.0%** | vs Pi0.5 96.9% / GROOT-N1.7 97.0% |
| RoboTwin2.0 Clean / Rand | SR | **93.6% / 93.3%** | vs Pi0.5 82.7% / 76.8% |
| VLA-Arena L0 / L1 / L2 | SR | **89.0% / 53.6% / 44.1%** | 难度递增 |
| RoboChallenge Table30v2 | Score / SR | **54.42 / 43.0%** | vs Pi0.5 31.48 / 14.3% |

## 架构与运行时入口

| 路径 | 作用 |
|------|------|
| `opendm/model/dm05/` | DM05 模型定义（含 LoRA） |
| `opendm/exp/dm05_exp.py` | Base 预训练模型实验入口 |
| `opendm/infer/dm05_infer*.py` | default / TRT fast 推理实现 |
| `opendm/dataset/{demo,libero,robotwin2,so101,vla_arena}.py` | 数据集注册 |
| `opendm/trainer/trainer.py` | SFT 训练器 |
| `script/dm05_launcher.sh` | 统一 train / inference 启动器 |
| `playground/dm05_*.py` | LIBERO / RobotWin2 / SO101 / Arena / demo SFT 入口 |
| `docs/en|zh/*.md` | 数据、推理、各 benchmark 与真机平台说明 |
| `third_party/robochallenge_inference/` | Table30 / RoboChallenge 推理侧策略封装 |

## 推理接口（docs/en/dm05_inference.md）

- **统一启动：** `script/dm05_launcher.sh --task inference --exp <entry> --model-config.model-name-or-path <ckpt> …`
- **HTTP：** 新集成优先 **`/v1/infer`**；旧 **`/process_frame`** multipart 兼容路径将逐步淘汰。
- **Default backend：** 标准 PyTorch；单进程即可（无需 `--nproc_per_node`）。
- **Fast backend：** `--inference-config.backend fast` 需 `pip install -e ".[fast-infer]"`（**TensorRT + Triton + FlexAttention** 缺一不可）；首次启动会导出 ONNX / 构建 vision TRT engine。
- **Norm stats：** checkpoint 内 `norm_stats.json`（可含 `norm_stats_by_robot`）；缺失时回退 `./norm_stats/`。按 `observation.robot_type` 选 profile，**未知机型不会静默回退默认**。
- **典型配置对齐：** Base → 3 图 / chunk 50 / action_dim 14；LIBERO → 2 图 / chunk 10 / action_dim 7；RobotWin2 → 3 图 / chunk 50 / action_dim 14。

## 数据与 SFT

- **格式：** 每 episode 一个 JSONL；每行一帧，含 `images_*`、`state`、`prompt`；`action` 可选（缺省则由未来 `state` 构造）。
- **注册：** `opendm/dataset/register.py` + 自定义 `register_dataset({...})`；训练时 `--data-config.dataset-name` 须匹配。
- **Demo 烟测：** `assets/demo/` + `playground/dm05_sft_demo.py`；完整流程见 `docs/en/dm05_finetuning.md`。
- **环境：** 推荐 Docker `dexmal/opendm:latest`；本地 Python 3.10 + CUDA torch + flash-attn；训练建议 8 GPU，推理 1 GPU。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Dexmal DM0.5](../../wiki/entities/dexmal-dm05.md) | 实体归纳页：架构主张 + OpenDM 复现栈 |
| [DM0.5 技术博客](../blogs/dexmal_dm05.md) | 方法叙事（历史上下文 / CoT / 轨迹对齐）一手来源 |
| [Dexmal DW05 / OpenDW](./dexmal_opendw.md) | 同机构 **世界模型** 开源线；OpenDM 为 **VLA** 线 |
| [VLA](../../wiki/methods/vla.md) | flow/chunk VLA 族谱定位 |
| [RoboTwin 2.0](../../wiki/entities/robotwin.md) | `DM05-robotwin2` 与 `robotwin2_generalist` 数据注册对齐 |

## 对 wiki 的映射

- 升格 / 大幅更新 **[`wiki/entities/dexmal-dm05.md`](../../wiki/entities/dexmal-dm05.md)**（补开源状态、权重表、工程实践与运行时序）。
- 交叉更新 [VLA](../../wiki/methods/vla.md)、[Dexmal DW05](../../wiki/entities/dexmal-dw05.md)。
