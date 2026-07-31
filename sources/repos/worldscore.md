# WorldScore（官方评测实现）

> 来源归档

- **标题：** WorldScore: A Unified Evaluation Benchmark for World Generation
- **类型：** repo + benchmark toolkit + HF dataset/leaderboard
- **组织：** Stanford（论文作者单位）；仓库维护 haoyi-duan
- **代码：** <https://github.com/haoyi-duan/WorldScore>
- **License：** MIT
- **论文：** <https://arxiv.org/abs/2504.00983>
- **项目页：** <https://haoyi-duan.github.io/WorldScore/>
- **Hugging Face：** 数据集 [Howieeeee/WorldScore](https://huggingface.co/datasets/Howieeeee/WorldScore)；榜单 [Howieeeee/WorldScore_Leaderboard](https://huggingface.co/spaces/Howieeeee/WorldScore_Leaderboard)
- **入库日期：** 2026-07-27
- **一句话说明：** 官方 WorldScore 工具链：配置 `.env` → 下载 3000 例数据集 → 在 `world_generators/` 适配 3D/4D/视频模型并生成 → `worldscore/run_evaluate.py` 跑十项指标 → 可选提交 JSON 上 HF 榜。
- **沉淀到 wiki：** [WorldScore（论文实体）](../../wiki/entities/paper-worldscore.md)

---

## README 归纳（环境、生成、评测）

1. **路径配置：** 根目录 `.env` 设 `WORLDSCORE_PATH` / `MODEL_PATH` / `DATA_PATH`，每会话 `export $(grep -v '^#' .env | xargs)`；可选 `.secrets` 放 API key（如 Gen-3 类）。
2. **数据集：** `python download.py` → `$DATA_PATH/WorldScore-Dataset`（对应 HF Howieeeee/WorldScore）。
3. **模型适配：**
   - `config/model_configs/<model>.yaml` 注册分辨率 / `generate_type`（i2v|t2v）/ frames / fps；
   - `worldscore/benchmark/utils/modeltype.py` 的 `type2model` 归入 `threedgen` / `fourdgen` / `videogen`；
   - 在 `world_generators/` 实现 `generate_video`（返回 `List[Image]` 或 `[N,3,H,W]`∈[0,1]），或按 `world_generators/README.md` 适配 WonderJourney / WonderWorld。
4. **生成：** `python world_generators/generate_videos.py --model-name <name>`（支持 submitit/Slurm 多卡）。
5. **评测环境（较重）：** conda `worldscore` + CUDA 12.1 系 PyTorch；子模块 **DROID-SLAM、Grounded-SAM、SAM2**；**VFIMamba**；再 `pip install .`；权重脚本下载 GroundingDINO / SAM / SAM2 / VFIMamba / `droid.pth` 等到 `worldscore/benchmark/metrics/checkpoints/`。
6. **评测入口：** `python worldscore/run_evaluate.py --model_name <name>` → `worldscore_output/worldscore.json`；完整性可用 `worldscore-analysis -cd` / `-cs`。
7. **上榜：** 自测完整后邮件提交 JSON；见 [Leaderboard 归档](../sites/worldscore-leaderboard-hf.md)。

---

## 目录导航（复现相关）

| 路径 | 作用 |
|------|------|
| `download.py` | 拉取 WorldScore-Dataset |
| `config/model_configs/` | 各模型生成配置 |
| `world_generators/` | 视频生成适配与 `generate_videos.py` |
| `worldscore/run_evaluate.py` | 评测主入口 |
| `worldscore/run_analysis.py` / CLI `worldscore-analysis` | 完整性 / 分数校验 |
| `thirdparty/` | DROID-SLAM、Grounded-SAM、SAM2 等 |

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Generative World Models](../../wiki/methods/generative-world-models.md) | 为「3D/4D/视频世界生成」提供统一、可复现、可上榜的评测坐标 |
| [EWMBench](../../wiki/entities/ewmbench.md) | 同属视频世界模型评测，但 EWMBench 锚定 **机器人操纵** 三轴；WorldScore 锚定 **多场景相机可控世界生成** |
| [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md) | 当把视频模型当世界接口时，WorldScore 暴露「跟不住运镜 / 跨场景不一致」等失败 |

---

## 对 wiki 的映射

- 新建 **`wiki/entities/paper-worldscore.md`**：论文 + 基准 + 工具链实体页（流程 mermaid、十项指标、论文表与活榜读法、源码时序图）。
- 轻量交叉更新 EWMBench / generative-world-models / depth-embodied-eval-benchmark / embodied-eval-benchmark-selection-loop / video-as-simulation。

---

## 外部参考（便于复核）

- Duan et al., *WorldScore: A Unified Evaluation Benchmark for World Generation*, ICCV 2025 / [arXiv:2504.00983](https://arxiv.org/abs/2504.00983)
- [haoyi-duan/WorldScore（GitHub）](https://github.com/haoyi-duan/WorldScore)
- [WorldScore Leaderboard（HF）](https://huggingface.co/spaces/Howieeeee/WorldScore_Leaderboard)
