# MIT HAN Lab / PEAR：Agentic Robotics 后训练引擎

> 来源归档：官方代码仓库及可运行入口核对

- **标题：** PEAR — a Post-training Engine for Agentic Robotics
- **类型：** repo
- **链接：** <https://github.com/mit-han-lab/pear>
- **文档：** <https://github.com/mit-han-lab/pear/blob/main/docs/README.md>
- **论文：** <https://mit-han-lab.github.io/pear-blog/paper.pdf>
- **项目页：** <https://mit-han-lab.github.io/pear-blog/>
- **权重：** <https://huggingface.co/collections/mit-han-lab/pear>
- **许可：** Apache-2.0（官方 LICENSE / pyproject）
- **入库日期：** 2026-11-15
- **回访代码快照：** [`31fd1cf`](https://github.com/mit-han-lab/pear/tree/31fd1cf2052c6fc989579ee77c8046d47fc0a5eb)（北京时间 2026-11-25 05:23:22）。
- **核对范围：** 根目录 README、docs/getting_started/quickstart.md、examples/models/qwen3_8_27b_pear/train/{README.md,run_stage1.sh}、deploy/{README.md,run_server.sh,run_station.sh}、scripts/start_ray_node.sh、pear/deploy/station/cli.py 与 pyproject.toml 的入口定义；回访另查三项提交的变更、站点客户端、模型集合及训练配置。
- **通读：** 已完成

## 实际运行入口（与 wiki 时序图对应）

| 环节 | 入口 | 已核对的作用 |
|------|------|--------------|
| 安装 | `scripts/install.sh libero_run` | 模型、LIBERO 与训练环境 |
| Ray 集群 | `scripts/start_ray_node.sh trainer|rollout` | 为训练器、vLLM rollout、LIBERO 仿真服务器提供资源 |
| Stage 1 | `examples/models/qwen3_8_27b_pear/train/run_stage1.sh` | 读取 LIBERO episode 配置；启动 `python -m pear.trainer.main`；128 组 × 8 rollout |
| Stage 2/3 | `run_stage2.sh` / `run_stage3.sh` | 加 LIBERO-Long、RoboDojo 任务的课程 |
| 评测 | `examples/eval/eval_libero.sh` | 输出 `val_dumps/eval_<step>*.json*` |
| 推理服务 | `examples/models/qwen3_8_27b_pear/deploy/run_server.sh` | 启动 `python -m pear.deploy.server.main` 的 vLLM HTTP 服务 |
| 真机 station | `examples/models/qwen3_8_27b_pear/deploy/run_station.sh` | 启动 `pear.deploy.station.cli gui`，配置 `VLLMClient` 连接服务 |
| 真机检查 | `pear-station yam/so101 preflight` 等 | 校准/静态检查/受控 smoke test；部署文档要求急停和人工监护 |

## 复现边界

- 训练配方示例需要分布式 H100/H200/GB300 集群；单张 RTX 5090 只对应量化推理服务，并非完成论文训练的硬件需求。
- LIBERO 使用 MuJoCo，RoboDojo 使用 Isaac Sim；场景和依赖需先安装。
- 模型已提供标准、LIBERO-only、NVFP4 权重；论文精度结果使用 BF16，不能把量化版结果视为等价。
- 官方 README 明确支持 YAM 与 SO-101 station；需要实体硬件、正确标定和安全互锁。

## 回访核对：训练 rollout 与站点发布分别记录

- [`fd583e9`](https://github.com/mit-han-lab/pear/commit/fd583e9) 在训练 rollout 中引入 MTP speculative decoding；[当前示例训练变量](https://github.com/mit-han-lab/pear/blob/31fd1cf2052c6fc989579ee77c8046d47fc0a5eb/examples/models/qwen3_8_27b_pear/train/README.md#variables) 与论文 v2 附录 A 对应。不能把新的训练吞吐或预计资源降幅当成原模型经过重训。
- [`7054599`](https://github.com/mit-han-lab/pear/commit/70545994c09c97ff2ba02f8c140bddd2fe64ab90) 让站点在单次会话使用一个客户端；例如 `VLLMClient` 构造一次 `AsyncOpenAI`，各轮请求通过该实例开启流并保留会话历史。[`31fd1cf`](https://github.com/mit-han-lab/pear/commit/31fd1cf2052c6fc989579ee77c8046d47fc0a5eb) 改进基于相机录制生成的视频及任务导出轨迹页；与控制动作接口和论文性能测量区分。
- 官方模型集合另列 [DFlash2 辅助 drafter](https://huggingface.co/mit-han-lab/Qwen3.8-27B-PEAR-DFlash2)，不是独立策略权重，也不同于附录 A 使用的内置 MTP。未下载权重、启动模型服务器或进行真机测试。

## 对 wiki 的映射

- [PEAR / RURL](../../wiki/entities/paper-pear-robot-use-rl.md)
- [论文全文核对](../papers/pear_robot_use_rl_mit_2026.md)
- [项目博客归档](../sites/pear_robot_use_rl_blog.md)
