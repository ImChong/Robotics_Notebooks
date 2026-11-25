# NavVerse-Benchmark

> 来源归档

- **标题：** NavVerse-Benchmark — A Physically Grounded Indoor-to-Outdoor Embodied Navigation Benchmark
- **类型：** repo
- **机构：** 密歇根大学（UMich）；CURLY 实验室（UMich-CURLY）
- **链接：** <https://github.com/UMich-CURLY/NavVerse-Benchmark>
- **项目页：** <https://umich-curly.github.io/NavVerse-Benchmark/>
- **数据集：** <https://huggingface.co/datasets/tccoin/navverse-benchmark>
- **论文：** <https://arxiv.org/abs/2607.19695>
- **入库日期：** 2026-09-13
- **最近核对：** 2026-11-21
- **代码 / 开源状态：** **已开源** — 2026-09-30 正式发布代码、基线包装器、Isaac Sim Docker 与 Hugging Face v1 场景及 episode 数据。
- **一句话说明：** NavVerse 官方仓库，提供统一命令行启动器 `navverse-run`、Isaac Sim/Lab 仿真封装、安全/过渡指标统计器与多种零样本基线适配器。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-navverse.md`](../../wiki/entities/paper-navverse.md)
- **交叉归档：** [navverse-benchmark-github-io.md](../sites/navverse-benchmark-github-io.md)、[navverse_arxiv_2607_19695.md](../papers/navverse_arxiv_2607_19695.md)

---

## 仓内结构（2026-11-21 核查，默认主分支）

| 路径 / 入口 | 作用 |
|-------------|------|
| `navverse-run` (`python -m navverse.launcher.cli`) | 统一 CLI 入口（支持 `setup` / `download` / `build` / `benchmark`，含 `--list-tasks`、`--list-presets`、`--dry-run`） |
| `navverse/` | 核心仿真与任务环境：`sim.py`、`vln_env_wrapper.py`、`episode.py`、`task_generator.py`、`path_following.py`、`measures.py`、`navmesh.py`、`robots/` |
| `baselines/` | 基线测评调度与适配器：`eval_navverse.py`、`evaluation_worker.py`、`uninavid`、`streamvln`、`poliformer`、`longnav_r1`、`internvla_n1`、`gpt`、`oracle`、`dummy` |
| `scripts/` | 安装与构建辅助脚本：`setup_navverse.sh`、`download_navverse_data.py`、`navverse_evaluation.py`、`navverse_task_generation.py`、`navverse_rebuild_navmeshes_isaac.py` |
| `configs/` / `docker/` / `episodes/` / `tests/` | 实验预设配置、Isaac Sim 兼容性 Docker 构建上下文、episode 配置及单元测试 |

**数据下载：** `navverse-run download` 从 Hugging Face `tccoin/navverse-benchmark` 拉取 `navverse_v1_july23.tar.gz` 与 `episodes/` 并校验 SHA-256，解压生成 `vc_plus/`、`grscenes_commercial/`、`navmesh/` 与 `episodes/` 并写入 `.env`（`SCENE_FOLDER` / `EPISODE_FOLDER`）。

---

## 对 wiki 的映射

- 实体页：[NavVerse](../../wiki/entities/paper-navverse.md)
- 任务交叉：[VLN](../../wiki/tasks/vision-language-navigation.md)、[零样本物体导航](../../wiki/tasks/zero-shot-object-navigation.md)
- 基准对照：[VLN-CE](../../wiki/entities/paper-vln-02-vln-ce.md)、[ESI-Bench](../../wiki/entities/esi-bench.md)
