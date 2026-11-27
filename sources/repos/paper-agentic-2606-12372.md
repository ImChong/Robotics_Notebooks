# Denghaoyuan123/UniIntervene — UniIntervene 官方开源仓库档案

- **GitHub 地址：** https://github.com/Denghaoyuan123/UniIntervene
- **默认分支：** `main`
- **开源协议：** MIT License
- **关联论文：** *UniIntervene: Agentic Intervention for Efficient Real-World Reinforcement Learning*（[arXiv:2606.12372](https://arxiv.org/abs/2606.12372)，CoRL 2026 Accepted）
- **核对边界：** 静态代码核对，未本地运行（基于仓库元数据、`README.md` 与 `files_sample` 文件清单核查；仓库仅提供离线训练、记忆构建与数据校验管线，不包含 UR7e 真机部署、SpaceMouse 接入与 HIL-SERL 在线训练代码，亦不含轨迹数据、训练权重或记忆库文件）。

## 核心入口与目录结构（已核实）

- **统一调度与数据校验入口：**
  - `run.py`：管线统一启动入口，支持 `--stage all` 或按顺序分阶段运行 `nttg`、`vf`、`annotate`、`mining`、`vq1`、`memory`、`vq2`，支持 `--dry-run` 仅打印命令不启动训练。
  - `validate_data.py` & `docs/DATA_FORMAT.md`：数据契约文档与校验脚本，检查 episode pickle（`list[dict]`）中的 `observations`（双视角 RGB 与机器人状态）、`next_observations`、7 维 `actions`（第 7 维为夹爪指令）、奖励/终止标记及任务元数据。
- **配置与发布清单：**
  - `configs/fold_towel.json`、`research_impl/config_loader.py`、`research_impl/configs/vq1/full_tvr_history.yaml`、`research_impl/configs/vq2/shared_residual.yaml`：Fold Towel 端到端示例配置及 VQ1 / VQ2 模型配置（`README.md` 指出 Fold Towel 示例配置使用 20 个进度分桶、每桶 12 条、span 32 与 0.40 价值提升阈值）。
  - `CODE_PROVENANCE.md`、`THIRD_PARTY_NOTICES.md`、`third_party_revisions.json`（记录所使用的 FAST 分词器版本）、`requirements.txt`、`requirements-optional.txt`。
- **离线训练、标注与挖掘模块（`research_impl/`）：**
  - `research_impl/nttg_label_buffer.py`：NTTG 进度标注，为轨迹转移生成 `raw_value` 与 `raw_norm`。
  - `research_impl/JEPA/train_vf_stitched.py` & `research_impl/scripts/run_vf_stitched.sh`：基于 SigLIP-SO400M 与 Gemma-3-270M 的拼接代理价值函数（VF）训练脚本。
  - `research_impl/scripts/annotate_buffer_vf_value_pred.py`：使用冻结 VF 对回放缓冲区逐转移标注 `vf_value_pred`。
  - `research_impl/scripts/mining_multitask_full_3rules.py`：基于值轨迹执行 decline / plateau / abrupt-drop 三规则干预点挖掘。
- **VQ1、恢复记忆与 VQ2 模块（`research_impl/`）：**
  - `research_impl/hil_vlm_train.py`、`research_impl/hil_vlm_dataset.py`、`research_impl/hil_vlm_eval_vf.py`、`research_impl/hil_vlm_action_parse.py`、`research_impl/vq1_heads.py`、`research_impl/episode_split.py`：基于 `Qwen3-VL-2B-Instruct` 与冻结 V-JEPA2 目标的 VQ1（未来、Twin-Q、风险与当前价值头）以及 VQ2 训练与划分逻辑。
  - `research_impl/memory/memory_bank.py`、`research_impl/memory/query_builder.py`、`research_impl/memory/recover_target_mining.py`、`research_impl/memory_constraints.py`、`research_impl/scripts/build_vf_delta_bank_fast.py`：在确定性 episode-level 训练集划分上筛选满足 `end_vf - start_vf >= delta_threshold` 与 `end_vf >= target_vf_min` 的恢复目标，并用同一 `vq1_epoch30` 检查点生成查询嵌入（验证/测试集轨迹不入库）。
  - `research_impl/fast_decoder.py`、`research_impl/fast_decoder_v2.py`、`tools/smoke_fast_overfit.py`、`reports/fast_overfit_fold_towel.json`：基于 `physical-intelligence/fast`（正交 DCT + 量化 + BPE + EOS/PAD 掩码 + 逐 token 交叉熵）的 8 步×7 维恢复动作块解码器与组件冒烟测试。
- **测试套件：**
  - `tests/test_data_contract.py`、`tests/test_launch_config.py`：数据格式契约与启动配置单元测试。

## 复现与运行依赖边界

- **软件与基础模型快照：** 需 Python 3.10+、PyTorch 2.4+、Transformers 4.57+，以及本地审校后的基础模型快照：`Qwen3-VL-2B-Instruct`、`SigLIP-SO400M`、`Gemma-3-270M`、`physical-intelligence/fast`（及 V-JEPA2 目标编码器）。所有启动脚本默认关闭 W&B，VF/VQA 路径默认禁用远程模型代码执行。
- **冒烟测试边界：** `tools/smoke_fast_overfit.py` 仅在小批量（16 chunks / 300 steps）上验证 FAST 分词器往返与解码器优化（`reports/fast_overfit_fold_towel.json` 记录 token CE 从 `7.6259` 降至 `0.00351`、token 准确率 `1.0`、重建 MAE `0.00404`），不代表留出集策略性能。
- **真机闭环边界：** 仓库明确说明机器人真机部署与 HIL-SERL 在线控制集成不在本代码包范围内，用户需自行提供与 `docs/DATA_FORMAT.md` 兼容的轨迹 pickle 数据。

## 与 wiki 的映射

- 对应实体页：[UniIntervene: Agentic Intervention for Efficient Real-World Reinforcement Learning](../../wiki/entities/paper-agentic-2606-12372.md)
