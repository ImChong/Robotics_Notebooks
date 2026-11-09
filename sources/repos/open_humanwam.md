# Open-HumanWAM 官方源码仓库

> 来源归档（repo）：从 README 与**具体源码入口**核查运行时链路；没有执行训练或评测。

- **标题：** Open-HumanWAM: An Open Recipe for Scaling Human and Robot Experience for Embodied Foundation Models
- **类型：** repo
- **官方代码（论文与项目页当前入口）：** <https://github.com/DAGroup-PKU/HumanNet/tree/Open-HumanWAM>
- **历史分支别名：** <https://github.com/DAGroup-PKU/HumanNet/tree/open-humanscale>；2026-11-09 GitHub API 核验两者均指向提交 `4e90ff6d4eafd762ec49eb0e44f2c0f0dab5ad2a`，未来可能独立变化。
- **项目页：** <https://dagroup-pku.github.io/open-humanwam/>
- **arXiv：** <https://arxiv.org/abs/2611.06817>
- **入库日期：** 2026-11-06；2026-11-08 补充源码核查；2026-11-09 校准官方分支
- **核对范围：** 根 README、humanwam_vanilla/README.md、humanwam_goal/README.md；Vanilla 的 scripts/train.sh、train.py、eval_robotwin.sh、eval_robotwin.py、humanwam/runtime.py、trainer.py、robotwin_policy.py；Goal 的 src/goalwam/cli.py、engine.py、server.py、examples/infer.py
- **沉淀到 wiki：** [paper-open-humanwam](../../wiki/entities/paper-open-humanwam.md)

## 已核查的可运行入口

| 模块 | 入口 | 源码确认的调用链 |
|---|---|---|
| Vanilla 训练 | `humanwam_vanilla/scripts/train.sh --config ...` | bash → `train.py`（单 GPU）或 `torch.distributed.run`（多 GPU）→ `humanwam.config.load_config` → `humanwam.runtime.run_training(cfg)` → 训练器/保存权重 |
| Vanilla RoboTwin | `humanwam_vanilla/scripts/eval_robotwin.sh` | 8 shard → `eval_robotwin.py` 启动单独 `robotwin_policy_worker.py` → `RoboTwinPolicy.predict` → `model.infer_action` → 回传 `32×16` chunk → RoboTwin 仿真 |
| Goal 文件推理 | `goalwam infer --checkpoint ... --sample ...` | `goalwam/cli.py` → `io.read_observation` → `GoalWAM.predict` → `ObservationProcessor/RobotProcessor` → `NativeBackend.predict` → `io.save_actions` |
| Goal HTTP | `goalwam serve --checkpoint ...` | `goalwam/cli.py` → `server.serve` → `GoalWAM.predict`，提供 HTTP /infer |
| Goal RoboTwin | `goalwam.robotwin.RoboTwinSession` | 官方目标缓存 `OfficialGoalCache` + `bridge.py` → 执行 session，检查 seed/场景/初始图像匹配，XPolicyLab 适配 |

## 开源状态与复现边界（2026-11-08）

- **部分开源**。官方 README 的 Release Plan 显示 Goal RoboTwin 与部分 Vanilla checkpoint 已发布，**完整 Open-HumanWAM-50K 语料尚在上传中**，Goal 的机器人后训练代码待发布；Vanilla 其余复现资源/探索权重亦未完全齐备。
- **Vanilla 脚本协议差异**：`eval_robotwin.sh` 的 `{0..7}` shard 合计 50 episodes/task/phase，论文的官方成绩用 100 episodes/task/phase；先标明评测协议再比较数字。
- **环境约束**：Goal README 要求 Python 3.13、PyTorch 2.10、CUDA 12.8、Hopper (SM90)、GLIBC ≥ 2.34；`PhyAIBackend` 可选但相应固定版本依赖并非完整公开。
- **许可证**：根目录 Apache-2.0；`humanwam_goal/README.md` 对 Cosmos 衍生代码单列 NVIDIA OpenMDW-1.1，模型和数据需看对应独立条款。
- 本次只进行**源码静态阅读和 README 交叉核验**，不声称 GPU 训练/评测已通过。
