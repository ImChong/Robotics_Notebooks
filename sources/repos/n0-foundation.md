# N0-Foundation（neoteai/N0-Foundation）

> 来源归档

- **标题：** 𝒩₀-Foundation — Towards the Age of Tactile Intelligence
- **类型：** repo
- **组织：** NeoteAI × Fudan TEAI
- **代码：** <https://github.com/neoteai/N0-Foundation>
- **项目页：** <https://research.neoteai.com/n0-foundation/>
- **技术报告：** <https://research.neoteai.com/assets/n0-foundation-report.pdf>
- **数据集：** <https://huggingface.co/datasets/NeoteAIEmbodied/OpenNeoData>（ModelScope：<https://modelscope.cn/datasets/NeoteAI/OpenNeoData>）
- **许可：** OpenNeoData 数据集为 CC-BY-NC-SA-4.0；`neoforce/` 与 `neosim/` 代码目录各有 MIT License，须分别遵循。
- **入库日期：** 2026-07-26
- **复核日期：** 2026-11-19
- **arXiv：** <https://arxiv.org/html/2608.29601v2>（v2）
- **NeoForce 权重：** <https://huggingface.co/NeoteAI/NeoForce>
- **一句话说明：** 𝒩₀-Foundation 官方仓库，当前已包含 NeoForce 可运行训练/评测、NeoSim 仿真代码，OpenNeoData 与 NeoForce 权重可获取；外置仿真资产仍须另行配置。
- **沉淀到 wiki：** [𝒩₀-Foundation（论文实体）](../../wiki/entities/paper-n0-foundation.md)

## 开源状态（步骤 2.5 · 2026-11-19 核验）

| 项 | 状态 |
|---|---|
| [OpenNeoData](https://huggingface.co/datasets/NeoteAIEmbodied/OpenNeoData) | **已发布** 5,000 h 子集；遵循数据平台访问与许可要求 |
| [NeoForce 代码](https://github.com/neoteai/N0-Foundation/tree/main/neoforce) | **已发布**：训练与评测源码、脚本、配置、数据处理 |
| [NeoForce 权重](https://huggingface.co/NeoteAI/NeoForce) | README 已列模型权重获取地址 |
| [NeoSim 代码](https://github.com/neoteai/N0-Foundation/tree/main/neosim) | **已发布**：12 个基准任务、Isaac Lab/TacEx 仿真采集与策略服务评测 |
| NeoSim 外置 USD 资产与 checkpoints | README 说明需另外下载，仓内仍有资产链接 TODO；不可视作即取即运行 |
| 全量 NeoData / 真机 NeoReal | 未发现全量 30k h 数据或真机成套开源交付 |

## 实际代码入口（非模板推测）

1. **NeoForce 训练**：`neoforce/scripts/train.sh` 使用 `torchrun --nproc_per_node=8 -m neoforce.train`，读取 `configs/neoforce.json`、`splits/train.jsonl` / `val.jsonl`、OpenNeoData，`train.py` 训练共享特征编码器和 EMA teacher，写 checkpoint。
2. **NeoForce 评测**：`neoforce/scripts/evaluate.sh` 调用 `python -m neoforce.evaluate`，评测测试分割上的接触像素 MAE / RMSE 与 contact mask mIoU。
3. **NeoSim 数据采集**：`neosim/scripts/collect_data.py` 读取 `task_config/<name>.yml`，启动 `isaaclab.app.AppLauncher`，加载 `envs.<task>`，逐 seed `reset → play_once → check_success → save_to_hdf5`。
4. **NeoSim 服务器策略评测**：`neosim/eval/run_eval.sh` 等待远程 policy server 端口，再运行 `eval/eval_official_ee.py`；simulator observation → WebSocket 推理 → action chunk 的协议见 [NeoSim README](https://github.com/neoteai/N0-Foundation/blob/main/neosim/README.md)。

- [NeoForce README](https://github.com/neoteai/N0-Foundation/blob/main/neoforce/README.md)
- [NeoSim README](https://github.com/neoteai/N0-Foundation/blob/main/neosim/README.md)
- **旧资料修正：** 2026-07-26 的「NeoForce 待发布、仓内仅 README」是历史状态，不代表 2026-11-19。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md) | **力场统一表示**（三轴力场）替代设备相关触觉外观 |
| [接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md) | NeoReal / NeoSim 标准化接触评测套件 |
| [VLA](../../wiki/methods/vla.md) | 在固定 π₀.₅ 骨干上对照触觉接入方式 |
| [𝒩₀-VTLA](../../wiki/entities/paper-n0-vtla.md) / [𝒩₀-TWAM](../../wiki/entities/paper-n0-twam.md) | 下游策略与世界–动作模型共用 NeoData |
