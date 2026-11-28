# CLARE 官方 LeRobot / PEFT 实现 / Official CLARE Implementation

- **类型：** repo
- **代码：** <https://github.com/learnsyslab/clare>
- **固定版本：** [ecb6600dd4052bffd3b78a961da644d191083801](https://github.com/learnsyslab/clare/tree/ecb6600dd4052bffd3b78a961da644d191083801)（2026-07-17）
- **项目页：** <https://tum-lsy.github.io/clare/>
- **论文：** <https://arxiv.org/abs/2601.09512v2>
- **许可证：** README 指向上游许可；`lerobot_lsy/LICENSE` 与 `peft_lsy/LICENSE` 均为 Apache-2.0。数据和模型的使用条款分别看其卡片，不从代码许可外推。
- **核对日期：** 2026-11-28

## 入口与发布边界

- CLI 入口 [`lerobot_lsy/src/lerobot/scripts/clare.py`](https://github.com/learnsyslab/clare/blob/ecb6600dd4052bffd3b78a961da644d191083801/lerobot_lsy/src/lerobot/scripts/clare.py) 加载当前任务数据、冻结策略和（若有）前阶段 PEFT 权重，处理分布漂移、适配器/判别器训练、检查点及 rollout。
- [`bash/clare/`](https://github.com/learnsyslab/clare/tree/ecb6600dd4052bffd3b78a961da644d191083801/bash/clare) 里实际脚本名是 `clare_libero_10.sh`、`clare_libero_goal.sh`、`clare_libero_spatial.sh`、`clare_libero_40_10_goal_spatial_object.sh`，不是 README 旧的 `dit_dec_libero_*.sh`。原样 `bash -n` 被未引用的 W&B 尖括号占位符阻断；用普通占位名替换后这 4 个脚本及 `pretrain.sh` 的语法检查通过，未实际执行训练。LeRobot 的包元数据要求 Python ≥3.10。
- 论文式 (7) 写带符号的标准化残差，而公开源码的 `discriminator.py::compute_z_score` 取绝对值；公开 LIBERO 脚本默认阈值 1.0，与论文表 I 仿真阈值 2.5 不同。
- 推理用 [`peft_lsy/src/peft/tuners/clare/layer.py`](https://github.com/learnsyslab/clare/blob/ecb6600dd4052bffd3b78a961da644d191083801/peft_lsy/src/peft/tuners/clare/layer.py) 的评估分支：批量计算判别器损失，按样本取最小重构误差并沿关联索引转发到适配器。DiTFlowMT 策略仅在动作队列需要重新填充时生成下一块，不能把模块路由耗时视作整轮闭环耗时。
- Hugging Face [资源主页](https://huggingface.co/continuallearning) 列出 [LIBERO 第 0 任务数据](https://huggingface.co/datasets/continuallearning/libero_10_image_task_0)、[真机 Lego 示范](https://huggingface.co/datasets/continuallearning/real_4_put_lego_into_drawer_filtered) 及 [真机条件投影适配器示例](https://huggingface.co/continuallearning/dit_posttrainv2_clare_dit_cond_proj_real_4_put_lego_into_drawer_filtered_consolidated_seed1000)；后者文件清单有 `adapter_model.safetensors`，并非自包含基础模型。
- README 和 shell 路径所用的 `dit_flow_mt_libero_90_pretrain_new` 公开 API 请求未返回可读条目（401）；可核实近名 [`dit_flow_mt_libero_90_pretrain`](https://huggingface.co/continuallearning/dit_flow_mt_libero_90_pretrain) 含 `model.safetensors`。未验证两者权重、处理器与模型配置等价，避免直接替换并声称精确复现。已看到仿真入口、模型与数据，不宣称一键真机复现。

- **对 wiki 的映射：** [CLARE](../../wiki/entities/paper-clare-continual-vla.md)
