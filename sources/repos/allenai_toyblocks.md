# allenai/ToyBlocks — Isaac/LeRobot 机器人数据生成与评测开源代码

- **仓库：** https://github.com/allenai/ToyBlocks
- **项目页：** https://allenai.github.io/ToyBlocks
- **官方博文：** https://huggingface.co/blog/allenai/toyblocks
- **许可：** 主仓 Apache-2.0；assets 和子模块的许可单独适用。
- **核查日期：** 2026-11-11
- **核对范围：** 官方 README Quick Start / Code Organization；scripts/datagen/run_datagen_pipeline.py 阶段定义与局部执行/重渲染逻辑；scripts/launch_eval.py 参数/模式及 scripts/eval/run_eval_pipeline.sh 服务/队列/worker 运行路径；src/toyblocks/data_annotation/annotate_target_xy.py 标注字段；未执行 Isaac/GPU 训练或真实机械臂测试。

## 源码结构与运行链
| 阶段 | 经官方代码核对的入口 |
|---|---|
| 生成任务 | `scripts/task_gen/stack_three.py` 等 PDDL 任务生成器 |
| 管线启动 | `scripts/datagen/run_datagen_pipeline.py` |
| 物理执行 | `scripts/vecgen/run.py` / `src/toyblocks/vecgen` 批量规划与技能执行 |
| 重渲染 | `scripts/render/render.py`，由 `--render-pass` 显式打开 |
| 合并 | `scripts/datagen/merge_datasets.py`，输出 LeRobot 格式 |
| 评测 | `scripts/launch_eval.py` 启动策略服务、任务代码生成、episode queue 和 GPU eval workers |

## 源码级运行路径（经公开入口核验）

1. `scripts/datagen/run_datagen_pipeline.py` 声明 `task-gen -> env -> execution -> render -> merge -> annotate` 六阶段；`--render-pass` 为**可选**阶段，execution 默认不记录相机视频；使用 `--resume` 从已完成阶段恢复。
2. `scripts/vecgen/run.py` 的执行记录通过 FF/PDDL 规划、技能执行和仿真验证；每个问题单独或多个问题共用 GPU 仿真进程，参数 `--vecgen-problems-per-env` 调整打包方式。官方文档给出最高 256 env/GPU、24× 实时速度，这是发布方硬件/配置结果。
3. `scripts/datagen/merge_datasets.py` 合并为 LeRobot 数据集，`src/toyblocks/data_annotation/annotate_target_xy.py` 写入 `goal.action`（文本动作）与 `goal.position`（目标像素 UV），push/align 另含 `goal.position_target`；若存在 camera pose/intrinsics 优先使用逐帧标定而非默认相机常量。
4. `scripts/launch_eval.py` 将 CLI 参数转换为环境变量并调用 `scripts/eval/run_eval_pipeline.sh`；后者按 **policy service（默认 GPU 0）-> task codegen -> episode queue -> N eval workers** 启动并 teardown。回放模式 `--replay` **跳过模型服务**。
5. 高低层替换：`--policy maniflow` 是高层 oracle + ManiFlow；`--policy molmo2_maniflow --ll-oracle` 是 Molmo2 高层 + 低层 vecgen oracle；使用此对照时请区分模型失败和动作执行失败。

源码入口：
- [datagen pipeline](https://github.com/allenai/ToyBlocks/blob/main/scripts/datagen/run_datagen_pipeline.py)
- [per-frame annotation](https://github.com/allenai/ToyBlocks/blob/main/src/toyblocks/data_annotation/annotate_target_xy.py)
- [eval launcher](https://github.com/allenai/ToyBlocks/blob/main/scripts/launch_eval.py)
- [eval pipeline](https://github.com/allenai/ToyBlocks/blob/main/scripts/eval/run_eval_pipeline.sh)

## 可复现的最小入口
````bash
git submodule update --init --recursive
uv venv --python 3.11
uv pip install -e '.[dev,isaac,torch-cu128,vecgen]'
get-toyblocks-assets --asset-manifest asset_manifest.json
python scripts/datagen/run_datagen_pipeline.py --experiment-id my_first_run --output-root data --task-gen-script scripts/task_gen/stack_three.py --task-gen-count 2 --env-per-problem 8 --num-envs 8 --exec-num-envs 8 --vecgen-problems-per-env 1 --render-pass src/toyblocks/domain_randomization/flags/render_flags_plain.yaml
````

- 注：已静态核对以上源码入口与阶段职责，但**未**运行 GPU benchmark、安装 Isaac 依赖、下载整套数据或复现原始评测；已开放仿真/策略代码≠已发布 FR3 真机控制工具包。
- [主节点](../../wiki/entities/toyblocks.md)
- [官方博文原文归档](../sites/ai2-toyblocks.md)
