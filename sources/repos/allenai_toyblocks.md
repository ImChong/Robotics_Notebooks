# allenai/ToyBlocks — Isaac/LeRobot 机器人数据生成与评测开源代码

- **仓库：** https://github.com/allenai/ToyBlocks
- **项目页：** https://allenai.github.io/ToyBlocks
- **官方博文：** https://huggingface.co/blog/allenai/toyblocks
- **许可：** 主仓 Apache-2.0；assets 和子模块的许可单独适用。
- **核查日期：** 2026-11-09
- **核对范围：** README Quick Start、Code Organization，scripts/datagen/run_datagen_pipeline.py 文件头及阶段调用说明，scripts/launch_eval.py 推理评测入口，src/toyblocks 目录结构；未实际执行。

## 源码结构与运行链
| 阶段 | 经官方代码核对的入口 |
|---|---|
| 生成任务 | `scripts/task_gen/stack_three.py` 等 PDDL 任务生成器 |
| 管线启动 | `scripts/datagen/run_datagen_pipeline.py` |
| 物理执行 | `scripts/vecgen/run.py` / `src/toyblocks/vecgen` 批量规划与技能执行 |
| 重渲染 | `scripts/render/render.py`，由 `--render-pass` 显式打开 |
| 合并 | `scripts/datagen/merge_datasets.py`，输出 LeRobot 格式 |
| 评测 | `scripts/launch_eval.py` 启动策略服务、任务代码生成、episode queue 和 GPU eval workers |

## 可复现的最小入口
````bash
git submodule update --init --recursive
uv venv --python 3.11
uv pip install -e '.[dev,isaac,torch-cu128,vecgen]'
get-toyblocks-assets --asset-manifest asset_manifest.json
python scripts/datagen/run_datagen_pipeline.py --experiment-id my_first_run --output-root data --task-gen-script scripts/task_gen/stack_three.py --task-gen-count 2 --env-per-problem 8 --num-envs 8 --exec-num-envs 8 --vecgen-problems-per-env 1 --render-pass src/toyblocks/domain_randomization/flags/render_flags_plain.yaml
````

- 注：文档只验证入口存在；没有执行 GPU benchmark、下载整套数据或复现原始评测。
- [主节点](../../wiki/entities/toyblocks.md)
- [官方博文原文归档](../sites/ai2-toyblocks.md)
