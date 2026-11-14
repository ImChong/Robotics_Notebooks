# Being-H（BeingBeyond）开源仓库与家族论文精读归档

- **类型：** 开源仓库精读（配合论文核对）
- **仓库：** <https://github.com/BeingBeyond/Being-H>
- **本地检出提交：** `27ec67b`（`main` 分支）
- **所属机构：** BeingBeyond（智在无界）
- **通读：** 已完成
- **核对范围：** 仓库 `BeingBeyond/Being-H@27ec67b`（`README.md`、`Being-H05/README.md`、`Being-H07/README.md`、`tutorials/Being-H-EDU/README.md`、`Being-H05/docs/unified_action_space.md`、`Being-H05/scripts/train/train_libero_example.sh`、`Being-H05/scripts/eval/eval-libero.sh`、`Being-H05/BeingH/train/train.py`、`Being-H05/BeingH/model/beingvla.py`、`Being-H05/BeingH/inference/run_server_vla.py`、`Being-H05/BeingH/inference/beingh_policy.py`、`Being-H05/BeingH/inference/beingh_service.py`、`Being-H05/BeingH/benchmark/libero/run_libero_eval_fast.py`、`Being-H05/BeingH/benchmark/utils/policy.py`），并对照三篇核心论文原文：`Being-H0`（`arXiv:2507.15597`）、`Being-H0.5`（`arXiv:2601.12993`，另见 [being-h05_arxiv_2601_12993.md](../papers/being-h05_arxiv_2601_12993.md)）与 `Being-H0.7`（`arXiv:2605.00078`）。

---

## 一、仓库结构与三代模型开源状态核对（`main@27ec67b`）

1. **`Being-H0`（arXiv:2507.15597）**：聚焦大规模第一视角人类手部操作视频预训练（UniHand，11 个数据集聚合数千万级手部运动帧，弱透视投影物理空间对齐 + 深度缩放/平面旋转视角均衡 + MANO 运动分词），在主仓库以独立 `being-h0` 分支托管。
2. **`Being-H0.5`（`Being-H05/`，arXiv:2601.12993）**：当前 `main` 分支完整可运行实现。核心架构为 Mixture-of-Transformers（MoT，采用 `InternVL3.5-2B` 理解主干 + `Qwen3-0.6B` 动作专家，另支持单 VLM 变体），采用 **200 维统一状态/动作空间（Unified Action Space, UAC）**、`ActionChunkLength=16` 与 **Flow Matching** 连续动作生成；支持 **MPG（Manifold-Preserving Gating）** 隐空间流形门控对齐与 **Training-Time RTC（Real-Time Chunking）** 前缀延时条件化。提供 LIBERO / RoboCasa / Cross-Embodiment 训练与评测脚本，以及 ZMQ 推理服务端 `BeingH.inference.run_server_vla`。
3. **`Being-H0.7`（`Being-H07/`，arXiv:2605.00078）**：提出无像素想象的双分支 **Latent World-Action Model**——在单个打包序列中并行构造使用可学习 latent queries 的 `prior` 分支与使用未来视觉表征 `z^post` 的 `posterior` 分支（共享上下文、双分支因果隔离 attention mask），通过表征对齐损失与反坍缩正则（范数下界 + 谱熵分散）让推理期仅运行可部署 prior 分支即可获得未来感知能力。本地检出中 `Being-H07/` 目前仅包含 `README.md`，训练与模型代码待后续发布。
4. **`tutorials/Being-H-EDU/`**：面向高校教学与入门用户的精简复现教程，覆盖从零微调到真机部署流程。

---

## 二、Being-H0.5 核心源码链路核对

1. **训练主链路（`BeingH/train/train.py` + `BeingH/model/beingvla.py`）**：
   - `main()` 初始化分布式环境并用 `HfArgumentParser` 解析 `ModelArguments`/`DataTrainingArguments`/`TrainingArguments`；载入 `InternVL3.5-2B`（ViT+LLM+Connector）与 `Qwen3-0.6B` 专家模型，向 tokenizer 添加动作特殊 token，构建 `BeingHConfig`（统一动作维 `200`、Flow Matching、可选 `use_mpg` 与 `use_training_time_rtc`）。
   - 若开启 `use_expert=True` 且 `use_mpg=True`，`BeingH.forward()` 在 `language_model.forward_train()` 前对状态/动作投影后缀与 VLM 上下文计算门控传输代价（`last_mpg_gate`、`last_mpg_transport_cost`），随后在 MoT 稀疏块注意力掩码下联合回归理解交叉熵损失 `und_loss` 与 Flow Matching 速度场损失 `action_loss`，由 `FSDPCheckpoint.fsdp_save_ckpt` 定期落盘并同步保存归一化元数据。
2. **ZMQ 服务与 LIBERO 评测闭环（`BeingH/inference/` + `BeingH/benchmark/libero/`）**：
   - `run_server_vla.py` 实例化 `BeingHPolicy`（加载模型权重与数据集 `unified_mapping`/归一化统计）与 `BeingHInferenceServer`（监听 `get_action` 与 `get_modality_config` 端口，默认 `5555`）。
   - `run_libero_eval_fast.py` 使用 `ProcessPoolExecutor` 并发启动试验，各 worker 创建 `Policy_Libero`（底层封装 `ExternalRobotInferenceClient`），通过 ZMQ 调用 `BeingHPolicy.get_action()`：依次经 `_modality_transform`、`_prepare_packed_inputs`（按 `unified_mapping` 填充状态张量并构造视觉-文本-动作序列）、`model.get_action()`（Flow Matching ODE 积分 `num_inference_timesteps` 步；可选 RTC 前缀锁定 `prev_chunk` 与两阶段 MPG 迭代细化），再按 `unified_mapping` 切片反归一化返回环境执行。
3. **开源边界**：`Being-H05/README.md` 明确标注大规模跨本体人类数据预训练（UniHand 2.0）脚本仍在整理（TODO），真机硬件控制部分需依实体机械臂与灵巧手配置独立适配。
