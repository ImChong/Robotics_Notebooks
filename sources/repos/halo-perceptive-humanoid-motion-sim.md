# ml-halo（HALO 官方仓库）

> 来源归档

- **标题：** HALO: Perceptive Humanoid Motion from Simulation
- **类型：** repo
- **链接：** <https://github.com/apple-aiml-research/ml-halo>
- **论文：** [halo_perceptive_humanoid_motion_sim_arxiv_2611_06908.md](../papers/halo_perceptive_humanoid_motion_sim_arxiv_2611_06908.md)
- **项目页：** [halo-perceptive-humanoid-motion-sim.md](../sites/halo-perceptive-humanoid-motion-sim.md)
- **核对提交：** `35d8334e9f7dc3e2b7a7737a9a3ec34f170c809f`（2026-11-04，"Add arXiv link (2611.06908) and update citations"）
- **入库日期：** 2026-11-06
- **核查方式：** `git ls-remote` 可达，`git clone --depth 1` 只读静态阅读；**未运行任何仓库代码**
- **沉淀到 wiki：** [paper-halo-perceptive-humanoid-motion-sim](../../wiki/entities/paper-halo-perceptive-humanoid-motion-sim.md)

## 开放状态

| 项 | 状态 |
|----|------|
| 代码 | 已公开（约 1,600 个文件） |
| 权重 | **不随仓库发布**：生成器 `best_model.pt` 用 `generator/scripts/train.py` 自训，跟踪器 ONNX 用 `tracker/training/scripts/rsl_rl/play.py --export_only` 导出 |
| 数据 | **不随仓库发布**；`datagen/` 只产出 `sit_on_chair` 的数据，其余技能需要仓库外数据（`system/README.md`） |
| 许可 | `LICENSE` = Apple 自定义条款（Copyright 2026 Apple Inc.；授予使用 / 复制 / 修改 / 再分发许可，保留声明，免责，不得用 Apple 名称背书）；`ACKNOWLEDGEMENTS` 列第三方许可（含 Kimodo、Infinigen、Qwen3.5）；README 内留有 `TODO(release): license / authors / scope` 注释 |
| 其它 | `CITATION.cff` 与 BibTeX 键 `halo2026` |

## 目录结构

| 目录 | 内容 |
|------|------|
| `datagen/` | 数据管线：`scene_gen/`（Infinigen + 烘焙）、`motion_gen/`（Kimodo + `motion_check` 校验、`tasks/sit.py`）、`render/`（Isaac / MuJoCo 渲染、`stitch.py`）、`launch/run_stage.sh`、`configs/pipeline/{scene_gen,gen,render}.env` |
| `generator/` | 运动生成器：`models/{motion_generator,vision_language_encoder,motion_decoder}.py`、`training/{trainer,flow_matching}.py`、`scripts/{train,eval_openloop}.py`、`configs/qwen3_5_2b_dit_vision_{frozen,finetune,finetune_vdrop}.yaml` |
| `tracker/` | 跟踪器：`training/`（Isaac Lab 任务 `Halo-MotionTracking-*`，PreRL / Distillation / PostRL）、`data/`、`deploy/`（Python）、`deploy_cpp/`（C++ 实时，ONNX Runtime / TensorRT，`src/vision/` 立体深度） |
| `orchestrator/` | `halo_orchestrator`：`orchestrator.py`（`deliberate` / `decide_at_settle`）、`skills.yaml`（11 个技能）、`functions.json`、`prompts.py` |
| `system/` | 闭环：`halo_system/scripts/{run_sim,run_real}.py`、`nodes/{motiongen_zmq,orchestrator_zmq}.py`、`lib/zmq_helper.py`；`comm/`（`halo_comm`，ZeroMQ + protobuf）、`serve/`（`halo_serve`，MLX 调度）、`cpp/nodes/motiontrack_{mujoco,real}_zmq.cpp` |
| `assets/` | Unitree G1 描述（MJCF / URDF / USD）、物体与场景 |

## README 要点

- 单任务配置（`push_box`、`crouch_under_table`、`go_to_man`、`sit_on_chair`、`sidestep_door`）不起编排器，可在 Linux + CUDA 上跑；多任务 `door_then_sit` 需 Apple 芯片 Mac（MLX，≥ 32 GB）。
- 运行入口：`python system/halo_system/scripts/run_sim.py --task_config <task>.yaml --web`；真机先在 Orin NX 启动 `motiontrack_real_zmq`，再在 Mac mini 上 `run_real.py --motiongen --orchestrator --web`；遥控器 Start → A → B，Select 急停。
- 数据生成：`datagen/launch/run_stage.sh {scene_gen|gen|render} --shard i --num-shards n`，全量为 300 个场景分片、250 个运动分片（50,000 个 episode）、250 个渲染分片。
