# dexmal/dexbotic（Dexbotic · VLA 工具箱）

> 来源归档（ingest）

- **标题：** Dexbotic — Open-Source Vision-Language-Action Toolbox
- **类型：** repo
- **组织：** Dexmal（大晓智能 / 原力灵机）
- **代码：** <https://github.com/dexmal/dexbotic>（MIT）
- **技术报告：** [arXiv:2510.23511](https://arxiv.org/pdf/2510.23511)
- **在线文档：** <https://dexbotic.com/docs/>
- **HF 组织：** <https://huggingface.co/Dexmal>
- **DM0.5 叙事：** <https://www.dexmal.com/blog/dm0.5>（中文）、<https://www.dexmal.com/blog/dm0.5/index_en.html>（英文）
- **DM05 基础权重：** <https://huggingface.co/Dexmal/DM05>（与 OpenDM 共用 checkpoint）
- **入库日期：** 2026-09-29
- **二次核查：** 2026-09-29（GitHub README、`docs/DM05.md`、HF 权重链与 OpenDM 对照）
- **一句话说明：** **Dexbotic** 是 Dexmal 的 **一站式 VLA 研发工具箱**（π0、CogACT、OFT、MemVLA、DM0、**DM05** 等）：分层配置 + 工厂注册 + playground 入口；**2026-09** 起在主线集成 **DM0.5**（LIBERO 全量/LoRA SFT、历史帧推理、高性能推理 backend），与 **[OpenDM](./dexmal_opendm.md)** 共享 **DM05** 权重但工程入口与多模型共栈不同。

## 开源状态（README / docs 核查）

| 项 | 状态（截至 2026-09-29） |
|----|-------------------------|
| **工具箱代码** | **已开源** — `dexbotic/` 包 + `playground/benchmarks/*` + `docs/`（MIT） |
| **DM05 训练 / 推理** | **已开源** — [`docs/DM05.md`](https://github.com/dexmal/dexbotic/blob/main/docs/DM05.md)；入口 `playground/benchmarks/libero/libero_dm05.py` / `libero_dm05_lora.py` |
| **DM05 权重** | **与 OpenDM 相同** — [Dexmal/DM05](https://huggingface.co/Dexmal/DM05)、[Dexmal/DM05-libero](https://huggingface.co/Dexmal/DM05-libero) 等 |
| **Docker** | 通用 `dexmal/dexbotic`；**DM05 专用** `dexmal/dexbotic:dm05` |
| **评测客户端** | 可选 [dexbotic-benchmark](https://github.com/Dexmal/dexbotic-benchmark) + `dexmal/dexbotic_benchmark` 镜像 |

## 与 OpenDM 的分工（选型读法）

| 维度 | **Dexbotic** | **OpenDM**（[dexmal_opendm.md](./dexmal_opendm.md)） |
|------|--------------|------------------------------------------------------|
| **定位** | 多算法 VLA **统一工具箱**（DM0/DM05/π0/CogACT/…） | **DM0.5 专用**训练·推理·数据注册·HTTP 服务 |
| **许可** | MIT | Apache-2.0 |
| **DM05 入口** | `libero_dm05.py`、FSDP2/DDP、统一 [Inference API](https://github.com/dexmal/dexbotic/blob/main/docs/InferenceAPI.md) `:7891/process_frame` | `script/dm05_launcher.sh`、JSONL 注册、default/TRT **fast** backend |
| **文档侧重** | LIBERO 全 SFT/LoRA、历史帧推理、与 π0 等 **同栈对比** | RobotWin2、VLA-Arena、Table30v2、SO101、LeRobot、RoboDojo-MEM、真机 `robot_platforms.md` |
| **权重** | 同一 HF **DM05** 系列 | 同一 HF **DM05** 系列 |

二者 **不是** 互斥 fork：官方在 Dexbotic News 将 DM05 作为工具箱一等公民；OpenDM 仍维护 **更广 benchmark 与 MaaS** 叙事。复现 **LIBERO 官方 99.0% 行** 可优先 Dexbotic `docs/DM05.md` + `DM05-libero`；复现 **RobotWin / Table30 / HTTP fast infer** 仍优先 OpenDM docs。

## 官方动态（README News · DM05 相关）

| 日期 | 要点 |
|------|------|
| 2026-09-11 | [DM05 历史帧推理](https://github.com/dexmal/dexbotic/blob/main/docs/DM05.md#history-frame-inference)（`history_images` 多帧上下文） |
| 2026-09-10 | DM05 **高性能推理 backend**，文称约 **5×** 核心推理加速（相对 default） |
| 2026-09-01 | **Dexbotic DM05 发布**；LIBERO 见 `libero_dm05.py` / `libero_dm05_lora.py` |
| 2026-02-10 | [DM0](https://github.com/dexmal/dexbotic/blob/main/docs/DM0.md) 与 Dexbotic 技术报告同步发布 |
| 2025-10-20 | Dexbotic 首次开源（arXiv 2510.23511） |

## DM05 工程要点（docs/DM05.md）

| 模块 | 路径 / 说明 |
|------|-------------|
| **LIBERO 全 SFT** | `playground/benchmarks/libero/libero_dm05.py` · FSDP2 · 8× H20/A100/H100 |
| **LIBERO LoRA** | `playground/benchmarks/libero/libero_dm05_lora.py` · DDP · 8× 4090 可行 |
| **数据** | HF [Dexmal/libero](https://huggingface.co/datasets/Dexmal/libero) → `data/libero/libero_pi0_all` |
| **推理服务** | `--task inference` · 端口 **7891** · `camera_order=["agentview","wrist"]` |
| **历史推理** | POST `history_images=@...`（与当前双视角并列） |
| **官方 LIBERO 分数** | 四套件平均 **99.0%**（Serving `Dexmal/DM05-libero`） |

### LIBERO 官方对照（Dexbotic docs 摘录）

| Method | Average |
|--------|---------|
| π0 | 94.2 |
| π0.5 | 96.9 |
| DM0.5 | **99.0** |

### Table30 v2（Dexbotic docs 摘录）

| Metric | DM0.5 | π0.5 |
|--------|-------|------|
| Score | **54.42** | 31.48 |
| SR | **43.0%** | 14.3% |

## 目录速查

| 路径 | 作用 |
|------|------|
| `dexbotic/exp/` | 各模型实验配置（含 DM0 hybrid co-train 等） |
| `dexbotic/model/` | 模型定义（含 **DW05** 等于 [OpenDW](./dexmal_opendw.md) 叙事） |
| `playground/benchmarks/libero/` | LIBERO 上 DM0/DM05/π0/CogACT 等 recipe |
| `docs/DM05.md` | DM0.5 安装、训练、推理、评测 |
| `docs/InferenceAPI.md` | v1 统一推理 JSON / `DexClient` |
| `hardware/docs/` | SO-101、DOS-W1、XLeRobot 等真机对接示例 |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Dexmal DM0.5](../../wiki/entities/dexmal-dm05.md) | 实体页：方法 + **OpenDM / Dexbotic 双栈复现** |
| [OpenDM](./dexmal_opendm.md) | DM0.5 **专用**栈与更广 benchmark docs |
| [DM0.5 博客](../blogs/dexmal_dm05.md) | 架构与 benchmark 叙事一手来源 |
| [Dexmal DW05](./dexmal_opendw.md) | DW05 模型加载示例在 Dexbotic `dexbotic.model.dw05` |
| [VLA](../../wiki/methods/vla.md) | 工具箱覆盖的主流 VLA 算法族 |

## 对 wiki 的映射

- 更新 **[`wiki/entities/dexmal-dm05.md`](../../wiki/entities/dexmal-dm05.md)**：补 Dexbotic 开源状态、LIBERO 复现路径、与 OpenDM 分工、Dexbotic 运行时序图。
- 交叉更新 [VLA](../../wiki/methods/vla.md)、[Dexmal DW05](../../wiki/entities/dexmal-dw05.md)。
