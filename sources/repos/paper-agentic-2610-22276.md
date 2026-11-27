# Fr0zenCrane/MJev-Spatial 仓库核查归档

- **项目名称：** MJev-Spatial（`Fr0zenCrane/MJev-Spatial`）
- **仓库地址：** <https://github.com/Fr0zenCrane/MJev-Spatial>
- **关联权重：** <https://huggingface.co/mindlab-research/MJev-Spatial-4B>
- **默认分支：** `main`
- **开源协议：** Apache-2.0（含 `LICENSE` 与 `NOTICE`）
- **开源状态：** **已开源**（模型运行时、训练与评测脚本、微调配置、合成请求示例及测试已公开；`MJev-Spatial-4B` 检查点托管于 Hugging Face；论文中的 12.3M 转换数据管线未打包进该推理/微调公共仓）
- **核查边界：** 基于缓存的仓库目录树（`files_sample`）与 `README.md` 完成静态代码与接口结构核查，**未在本机下载权重或运行 CUDA 推理/训练**

## 已核查目录与核心文件

| 路径 | 角色与功能 |
|---|---|
| `pyproject.toml` | 包定义与可选依赖分组（`pip install -e '.[inference,train]'`，Python $\ge 3.10$ + CUDA PyTorch） |
| `configs/finetune.json` | 微调配方配置（默认冻结视觉编码器与投影器，更新语言侧决策参数） |
| `scripts/predict.py` | 批量推理 CLI 入口（读取 `--checkpoint`、`--input examples/requests.jsonl`、`--output`） |
| `scripts/train.py` | 有监督微调 CLI 入口（读取 `--checkpoint`、`--data examples/train.jsonl`、`--config configs/finetune.json`、`--output`） |
| `scripts/evaluate.py` | 公共 JSONL 评测 CLI 入口（输出 choice 准确率、scalar 误差、point 距离/命中率） |
| `src/spatial_jev/schema.py` | 请求与目标契约校验（`choice` / `scalar` / `point` 的 `answer_space` 与 `target`） |
| `src/spatial_jev/qwen35.py` | Qwen3.5-4B 骨干适配与冻结视觉分支封装 |
| `src/spatial_jev/kev_native_qk.py` | 最终全注意力层原生 Q/K 读出（`N_in`、头归一化、GQA 对齐、mRoPE、均值缩放内积） |
| `src/spatial_jev/kev.py` / `kev_prefix.py` | KEV 指针读出与前缀相关实现 |
| `src/spatial_jev/unified.py` | 统一多阶段决策调度与前向打分接口 |
| `src/spatial_jev/hierarchy.py` | 粗到精 3×3 空间网格划分与边界坐标映射 |
| `src/spatial_jev/numeric_tasks.py` | 长度 / 计数 / 角度码本区间构造与中点解码 |
| `src/spatial_jev/point_probability_map.py` | 多点条件概率树扩展与连通域峰值解码 |
| `src/spatial_jev/roi_replay.py` / `visual_token_pointing.py` | 消融对比模块：ROI 特征/K-V 重放与直接视觉 token 指向 |
| `src/spatial_jev/training_alignment.py` | 多阶段 Teacher-Forcing 联合监督与训练对齐逻辑 |
| `examples/` | 含 `requests.jsonl`、`train.jsonl`、`eval.jsonl`、`images/scene.png` 等无私有数据依赖的合成示例 |
| `tests/test_public_contract.py` | 公共输入/输出契约自动化测试 |

## 静态接口与运行边界

- **输入契约（JSONL）：** 每条记录包含 `media`（图像 URI）、`question` 与 `answer_space`（`choice` 传入 `options`、`scalar` 声明 `quantity` 与 `unit`、`point` 声明 `normalized_xy` 与 `num_points`）；训练/评测记录额外附带 `target`（`option_id` / `value` / `points`）。
- **推理命令：** `python scripts/predict.py --checkpoint "$MJEV_CHECKPOINT" --input examples/requests.jsonl --output predictions.jsonl`，返回解码答案、决策路径与各阶段候选分数。
- **训练与评测边界：** 公共训练脚本面向用户自备 JSONL 标注做有监督微调并保存可重载 checkpoint；仓库未包含论文中 12.3M 规模训练的分布式长短分组调度集群脚本或完整原始训练集打包。
