# AutoVLA-D 官方代码（ShivaTeam/AutoVLA-D）

> 论文配套 GitHub 仓库的代码、工程文档与开放程度记录。

- **项目：** AutoVLA-D / TrajCond-VLA
- **代码：** <https://github.com/ShivaTeam/AutoVLA-D>
- **论文：** <https://arxiv.org/abs/2611.12382>
- **相关论文归档：** [AutoVLA-D](../papers/autovla_d_arxiv_2611_12382.md)
- **许可证：** Apache-2.0（仓库 LICENSE）
- **读取版本：** 2026-11-19 公开 master；未锁定到永久 SHA，复现时须自行钉住提交。
- **核对范围：** 根目录 README、`data_engine/README.md`、`data_engine/main.py`、`data_engine/src/pipelines/runner.py`、`data_engine/src/prompts/trajcond_stage1.py` / `trajcond_stage2.py` / `trajcond_common.py`、`codebook/build_codebook_from_gt.py`、`sft/` 训练目录与 README 列出的配置入口。
- **通读：** 已完成
- **wiki 映射：** [AutoVLA-D](../../wiki/entities/paper-autovla-d.md)

## 开源状态（2026-11-19）

| 组件 | 状态 / 可核查入口 |
|---|---|
| 轨迹 codebook | **已开源** `codebook/build_codebook_from_gt.py`；Brohan、KMeans、KDisks 及已有 codebook 文件 |
| NAVSIM 数据引擎 | **已开源** `data_engine/main.py` 与 `src/pipelines/runner.py`；三类 tokenizer × 两阶段配置 |
| SFT 训练 | **已开源配置/词表扩展** `sft/extend_action_vocab.py`、`sft/configs/*.yaml`；依赖外部 LLaMA-Factory 和 DeepSpeed |
| CoT 标注与训练数据 | README 链接 [Google Drive](https://drive.google.com/drive/folders/1T5kgpoeLcoKxoguCWJprfR3JP3WVrEt6?usp=sharing)，实际下载权限和文件版本需自行验证 |
| GRPO / RFT | **主仓没有独立完整训练实现**；README 指向 [Curious-VLA](https://github.com/Mashiroln/curious_vla) 的 GRPO workflow，需修改 rollout 和轨迹 parser |
| 权重和车端部署 | README 未公开可直接使用的 AutoVLA-D checkpoint，也未展示实车推理 / 上车测评入口 |

## 真实入口与代码调用关系

1. `python codebook/build_codebook_from_gt.py --method brohan --size 768`：从 GT pickle 的 `[x,y,yaw]` 轨迹求前一时刻自车系下的差分动作，生成分轴 codebook。README 的 768 意味着每轴 256 token。
2. `python data_engine/main.py --config data_engine/configs/pipelines/navsim_brohan_stage1.yaml --check-config`：加载 YAML 合并默认配置、构建 PromptBuilder、实例化 Enrichers 并执行 `preflight`；**仅检查配置，不抽取 NAVSIM**。
3. 去掉 `--check-config` 后，`main.py` 调 `pipelines.runner.run_pipeline(config,builder)`，使用 NAVSIM Adapter → `build_sample_ir` / `compute_sample_trajectory` → Enrichers 校验/CoT 关联/动作编码/误差过滤 → `TrajcondStage1Builder` 或 `TrajcondStage2Builder` 输出训练 QA → JSON writer。
4. `stage1` 和 `stage2` 使用不同 YAML 和 prompt schema。Stage-2 还需要 `normalized_target_trajectory`，其目标由归一化的真值连续轨迹生成。
5. `python sft/extend_action_vocab.py ... --num-action-tokens 768` 扩充 Qwen3-VL 词表，再按 README 依次运行 `llamafactory-cli train sft/configs/navsim_brohan_stage1.yaml` 和 `stage2.yaml`，第二阶段从第一阶段 checkpoint 初始化。

**论文 / 开源实现差异：** 论文 §3.2 采用局部差分的 1% / 99% 分位裁切；当前公开 `build_codebook_from_gt.py` 函数 `build_codebook` 按每轴 `min/max` 用 `np.linspace` 生成 inclusive 栅格。论文使用逐步角点候选搜索，实际数据引擎也含 trajectory search/filter；但这处边界差异应在复现时核对输出统计与版本，不能把代码与论文简单画等号。

## 复现边界

- 需要独立准备 NAVSIM v2、本地路径、图片、CoT annotation、GT trajectory、训练好的基础模型和 LLaMA-Factory 环境；`--check-config` 不能代替数据完整性验证。
- 主仓已发布训练工程入口，但本次只做源码静态核验，**没有运行训练、车载推理或基准复现**。

## 2026-11-25 官方发布与配置复核

- [README](https://github.com/ShivaTeam/AutoVLA-D/blob/master/README.md) 的标注及数据入口已更新为上表地址；未验证实际文件下载权限。README 的 arXiv 徽章文字仍显示 `2607.01658`，但链接目标为正确的 `2611.12382`，引用以论文 URL 为准。
- [Codebook 文档](https://github.com/ShivaTeam/AutoVLA-D/blob/master/codebook/README.md) 与根 README 确认默认 `trajectory.pkl` 为 BON 优化轨迹；如需原始轨迹，作者指示使用 `trajectory-gt.pkl` 替换。实验复现需记录使用哪一份轨迹与码本，避免混用数据版本。
- [Stage 1 数据配置](https://github.com/ShivaTeam/AutoVLA-D/blob/master/data_engine/configs/pipelines/navsim_brohan_stage1.yaml) 和 [Stage 2 数据配置](https://github.com/ShivaTeam/AutoVLA-D/blob/master/data_engine/configs/pipelines/navsim_brohan_stage2.yaml) 均设置每轴候选数 4、beam size 1、重建误差过滤比例 0.01；第二阶段额外有 `merge_gt_trajectories`（读取 `codebook/trajectory.pkl` 的 `gt` 字段）与 `normalize_trajectory`（处理 8 个时刻的 `target_trajectory`）。
- [Stage 1 SFT](https://github.com/ShivaTeam/AutoVLA-D/blob/master/sft/configs/navsim_brohan_stage1.yaml) 与 [Stage 2 SFT](https://github.com/ShivaTeam/AutoVLA-D/blob/master/sft/configs/navsim_brohan_stage2.yaml) 均使用全参数微调、BF16、3 epochs、学习率 4e-5、每卡 batch 2、梯度累积 1、DeepSpeed ZeRO-2；Stage 2 从 `./models/brohan_stage1` 加载。按 README 的 16 卡启动方式，全局 batch 为 32。
- 本次仅复核官方 README、codebook 文档和上述四份配置，未重新运行源码、下载数据或复现论文实验；不改变原有论文实体及其流程图、运行时序图和路线挂载。
