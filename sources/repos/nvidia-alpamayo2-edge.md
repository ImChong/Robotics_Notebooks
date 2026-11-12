# NVlabs/alpamayo2-edge — 推理与部署源码

> 来源归档

- **标题：** NVIDIA Alpamayo 2 Edge: standalone PyTorch inference and post-training/deployment recipes
- **类型：** repo
- **组织：** NVlabs / NVIDIA
- **代码：** <https://github.com/NVlabs/alpamayo2-edge>
- **配方：** <https://github.com/NVlabs/alpamayo-recipes>
- **权重：** <https://huggingface.co/nvidia/Alpamayo2-Edge>
- **入库日期：** 2026-11-12
- **版本：** 上游 main 分支截至 2026-11-12；PyTorch 推理 README、src/alpamayo2_edge 主路径及 alpamayo-recipes 相应配方 README。
- **核对范围：** 通读 NVlabs/alpamayo2-edge README 全文，逐段阅读 src/alpamayo2_edge/inference.py、checkpoint.py、helper.py、load_physical_aiavdataset.py 与 models/alpamayo2_edge.py 的实际加载、输入处理、共享 prefill、CoC 生成、diffusion expert 采样和轨迹恢复主路径；通读 alpamayo-recipes 的 README 与 recipes/alpamayo2edge_sft/README.md、recipes/alpamayo2edge_edgellm/README.md。未逐文件审计所有测试与 CUDA 图 kernel，也未实际安装或运行代码。
- **通读：** 已完成
- **一句话说明：** 开放 BF16 PyTorch 模型推理及输入规范，另有联合 SFT 与 TensorRT Edge-LLM 的 DRIVE AGX Thor FP8 导出/部署配方。
- **沉淀到 wiki：** [Alpamayo 2 Edge](../../wiki/entities/alpamayo2-edge.md)

## 开源状态和运行边界

| 资产 | 状态 | 依据 |
|---|---|---|
| 模型权重 | 已发布，OpenMDW-1.1 | [HF 模型卡](https://huggingface.co/nvidia/Alpamayo2-Edge) |
| 独立推理 | 已开源，Apache-2.0 | [NVlabs/alpamayo2-edge](https://github.com/NVlabs/alpamayo2-edge) |
| 联合 SFT | 已开放配方 | [alpamayo2edge_sft](https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo2edge_sft) |
| Thor FP8 部署 | 已开放配方 | [alpamayo2edge_edgellm](https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo2edge_edgellm) |
| PhysicalAI-AV 样例数据 | 有访问条款 | [数据集](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles) |
| 整套生产级车辆安全系统 | 未提供此保证 | 模型卡的限制声明 |

## 官方可执行入口和调用链

README 的基础运行路径是 Linux + Python 3.12 + 支持 BF16 的 NVIDIA GPU，推荐至少 16 GB 显存，模型与数据集均需 HF 访问权限：

```bash
uv sync --locked --extra dataset --dev
hf auth login
python -m alpamayo2_edge.inference \
  --model-id nvidia/Alpamayo2-Edge \
  --manifest examples/nav_samples.json \
  --output outputs/edge
```

执行链（已按源码核对）：

1. `inference.main` 校验 manifest/clip/time/navigation、设备与采样参数；
2. `load_physical_aiavdataset` 提取六相机历史帧、时间戳与 20 帧自车位姿；
3. `checkpoint.load_model` 读取完整 HF checkpoint，拒绝缺 shard 的模型；
4. `helper.prepare_model_inputs` 组装多模态消息、image tokens 与历史轨迹；
5. `Alpamayo2Edge.sample_trajectories_from_data` 复用 VLM prompt prefill，生成 CoC（或 NOMINAL）；用截止到未来轨迹锚点的 KV cache 条件化 expert；
6. `_sample_actions` 调用 `expert.diffusion.sample` 生成加速度和曲率，`action_space.action_to_traj` 转换成 XYZ/rotation；
7. `run_data` 检验 64 轨迹点 shape 与有限值，写出 `output.safetensors`、`packed.safetensors`、`prompt.txt`、`output.json`，最终 `report.json`。

### 后训练与车端部署

- SFT：`recipes/alpamayo2edge_sft/` 提供 VLM 未来轨迹离散 token 与专家 acceleration/curvature flow matching 的联合训练，expert 对 VLM conditioning 使用 detach；原生任务应使用 `sft_joint_nav_cot`。
- DRIVE AGX Thor：`recipes/alpamayo2edge_edgellm/` 使用 `host/shim.py`、`host/quantize.py`、`host/export.py`、`host/action_graph.py` 等导出 FP8 TensorRT Edge-LLM 引擎，经 `thor/run.py` 与 `thor/runtime/` 执行；包括共享 prefill、DFlash2 speculative decoding、expert 访问 LM KV pool。README 测试环境 DRIVE OS 7.2.7.0 / CUDA 13.4 / TensorRT 11.3.1.9。

## 对 wiki 的映射

- 主实体：[Alpamayo 2 Edge](../../wiki/entities/alpamayo2-edge.md)
- 官方发布归档：[产品页、技术博客与模型卡](../sites/nvidia-alpamayo2-edge.md)
- 基础模型：[Cosmos 3](../../wiki/entities/cosmos-3.md)
- VLA 方法：[视觉—语言—动作](../../wiki/methods/vla.md)
