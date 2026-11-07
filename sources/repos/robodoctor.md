# RoboDoctor 官方实现：代码、运行入口与可复现边界

> 来源归档（repo；2026-11-07，官方 main commit `e3ba0eaaf1fdf0ffb244c9b2919abaa19544eac9`）

- **标题：** RoboDoctor: Training-Free Recursive Self-Improvement for Robot Policies
- **类型：** repo
- **仓库 / 代码：** <https://github.com/THUSI-Lab/RoboDoctor>
- **项目页：** <https://robodoctor-rsi.github.io/>
- **论文：** <https://arxiv.org/abs/2610.35604>
- **权重：** <https://huggingface.co/THUSI-Lab/PrismWAM/tree/main/robodojo>
- **许可证：** Apache-2.0（仓库 LICENSE；三方依赖另有其许可证）
- **入库 / 更新：** 2026-11-07
- **核对范围：** 完整 README（安装、支持 workflow、运行与可用资源）、`run_rsi.sh`、`run_pipeline.sh`、`robodoctor/pipeline.py`（Pipeline.command/run）、`robodoctor/launch.py`（主入口/进程组织）、`robodoctor/rsi.py`（RSITools.start/rollout/continue_policy/act）、`robodoctor/experience.py`（ExperienceStore）、`robodoctor/prismwam_policy.py`（策略加载/推理适配器）、`robodoctor/harness.py`（evolve_harness）、`robodoctor/runner.py`（Agent 运行时/工具）；未遍历所有依赖及全部 tests，也未运行 GPU 仿真。
- **通读：** 未完成

## 已验证的真实运行路径

1. `bash run_rsi.sh` 调用 `python -m robodoctor.launch --evaluation-method rsi_policy_doctor`；`bash run_pipeline.sh` 调用 `python -m robodoctor.pipeline`。
2. `Pipeline.run` 生成 iter0–2 的 `rsi_policy_doctor` 运行命令，传入前回合经验；`launch` 管理 simulator、policy service 与 Codex agent 进程。
3. `RSITools.start` 通过 RoboDojo RPC reset 并记录真实回合；`RSITools.rollout` 调用 `student.infer` 获取 H×14 chunk，通过 `fk_preview` 获取零物理步的预览。
4. Agent 可以 `continue_policy` 放行前缀：先检查当前 `prediction_id`，再 `chunk_step` 提交动作；或使用 `RSITools.act` 的 grounded 末端/关节动作作 1–5 步介入。执行过的逐帧观测交给 `student.observe`；原提案后缀废弃。
5. `ExperienceStore` 保存诊断/操作方法及证据；回合结束输出 `intervention_experience.md`；pipeline 根据前三次得分和阈值选择后续 RSI+Harness 修改或 Direct 分支。
6. `robodoctor/prismwam_policy.py` 检查官方 RoboDojo 模型配置、14 维动作、32 步预测/16 步重规划和 VAEs/归一化资源；README 的模型和依赖必须单独下载/配置。

## 安装、复现与边界

- README 要求 Linux、NVIDIA GPU、Vulkan/EGL、RoboDojo、Isaac Sim 5.1、策略推理环境以及 Codex app-server 和模型 API。先运行环境检查、pytest 与 `bash run_pipeline.sh --dry-run`；dry-run **不代表** GPU 渲染/策略推理已验证。
- 官方发布的是运行时源代码和测试，包含在 `third_party/prismwam/` 的推理依赖，但模型和大体量机器人仿真资产不在该仓库内。无需重新训练 WAM 来部署该系统。
- 本次仅源码静态核对；**没有**下载模型、获取私有凭据或运行 RoboDojo，不能声称成功复现任务评分。
- 早期 `THUSI-Lab/RoboDoctor-Preview` 空仓库（2026-10-22）是旧记录；请以现在的正式实现为准。

## Wiki 对应

- [RoboDoctor 论文与项目统一节点（含源码时序图）](../../wiki/entities/paper-robodoctor.md)
- [原论文通读归档](../papers/robodoctor_arxiv_2610_35604.md)
- [官网](../sites/robodoctor.md)
