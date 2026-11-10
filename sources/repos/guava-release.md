# Guava Release — 官方源码阅读与入口分析

- **仓库：** <https://github.com/hdacnw/guava-release>
- **README：** <https://github.com/hdacnw/guava-release/blob/main/README.md>
- **项目页：** <https://guava-harness.github.io/>
- **论文：** <https://arxiv.org/html/2606.18363v3>
- **学生权重：** <https://huggingface.co/AIcell/guava-v13b-qwen3.5-4b>
- **数据集：** <https://huggingface.co/datasets/AIcell/Guava-4B-SFT-data>
- **核对范围：** main 的 README（安装、15 任务、评测、采集、恢复分支、SFT/GRPO、现实部署说明）、目录树及 guava/inference/run_study.py、guava/inference/benchmark_engine.py、guava/inference/rollout_env.py、guava/scripts/collect_release.py、guava/scripts/collect_branches.py、guava/collection/runner.py、guava/collection/rollout.py、guava/tools/tools.py、scripts/run_sft_full.sh 与 scripts/run_grpo_full.sh 的入口和关键调用关系。
- **通读：** 已完成

## 已公开功能与限制

- 仿真 15 项桌面操作任务及 benchmark、轨迹采集与恢复分支、过滤转换脚本、SFT/GRPO 训练入口、本地/云端模型推理。
- Linux x86-64、Python 3.11、CUDA；README 评测环境使用 RTX 5090。评测、SAM3、vLLM、训练各自有隔离依赖环境；SAM3 第三方 checkpoint 需单独获得访问权限。
- 根目录列出 README、guava、configs、scripts、pyproject.toml、third_party、uv.lock 等，**未找到根 LICENSE**；源码公开不能自动等于商用许可。
- 真机实验在论文/演示中报告，但当前核对的发布 README 核心复现步骤是 MuJoCo Harness；不保证存在开箱即用的 FR3 现场部署配置。

## 源码运行链：评测

1. `python -m guava.inference.run_study` → `run_study.main()`：解析任务/模型，生成 manifest，按需启动本地 vLLM 和 SAM3 server，按 job 调用 benchmark engine。
2. `guava.inference.benchmark_engine.main()` 配置 `RolloutEngine`、task 和模型 client；`run_episode()` 进行 `episode.reset(seed)`、刷新 RGB/夹爪 observation。
3. 每轮 `client.chat.completions.create(messages)` → `decision(text, mode)` 解析单次语义工具 → `registry[name]`，最终调用 `guava.tools.tools` 的感知/夹爪/机械臂操作；执行反馈与新图像加回 history。
4. `state(engine)` 查询任务成功判据，结果写入 `responses.json`、`actions.json`、`result.json`、summary。失败工具动作同样返回反馈；模型仍可重规划。

## 源码运行链：教师数据与恢复

1. `python -m guava.scripts.collect_release` → 读取 `configs/<task>.yaml` → `guava.collection.runner.run_collection`。
2. `_build_components` 创建 MuJoCo task env、RobotArm/IK、感知及 LLM provider。
3. `guava.collection.trial.run_trial` → `guava.collection.rollout.continue_rollout` → `query_checked` → `execute_observe`（语义工具调用、更新 RGB/夹爪状态）；保存每轮工具及 reasoning、checkpoint 和 trajectory。
4. `python -m guava.scripts.collect_branches` 载入记录的 checkpoint，注入 grasp_failure / object_moved_align / drop_during_transport 等扰动，从父轨迹历史继续 rollout 并保存分支。
5. `scripts/filter_trajectories.py` 只生成筛选 manifest，不能自动代表人工视觉审核；`scripts/convert_to_native.py` 将整理后的 JSONL 转成 Qwen3.5 训练格式。
6. `scripts/run_sft_full.sh` 调用 ms-swift 全参 SFT；`scripts/run_grpo_full.sh` 编排 SAM3、sim_server、rollout/vLLM 和 GRPO 训练器。脚本默认超参可与论文表略异，实验须固定实际配置。

## 官方 README 示例（先 dry-run）

```bash
.venv-eval/bin/python -m guava.inference.run_study \
  --provider local --models AIcell/guava-v13b-qwen3.5-4b \
  --tasks can_in_bin --episodes 1 --output runs/guava-check --dry-run
```

`--dry-run` 只核对参数，不启动仿真/模型。实际运行前按 README 安装分离环境、下载模型并确认 SAM3 访问及 CUDA 硬件。此处仅静态阅读源码，**未声称已启动评测或复现论文结果**。

## 关联

- [论文精读](../papers/guava_arxiv_2606_18363.md)
- [项目页核验](../sites/guava-harness-project.md)
- [知识实体](../../wiki/entities/paper-guava-harness.md)
