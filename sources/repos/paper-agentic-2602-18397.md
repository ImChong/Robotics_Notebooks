# VLA-Perf 官方代码：VLA 推理性能解析建模工具

> 来源归档（repo）；核查日：2026-11-27；**已开源**（Apache 2.0 协议）

- **仓库：** <https://github.com/NVlabs/vla-perf>
- **默认分支：** `main`（快照可见 104 Stars / 10 Forks）
- **论文：** <https://arxiv.org/abs/2602.18397>（arXiv HTML 全文 <https://arxiv.org/html/2602.18397>）
- **许可：** Apache 2.0 License（仓库根目录 `LICENSE` 与 `vla-perf/LICENSE`；内置扩展的 `genz/` 子目录保留其 `genz/LICENSE`）
- **核对范围：** 基于仓库主页、`README.md` 预览及文件清单（`files_sample`）进行静态结构与入口核对；未本地克隆及运行 Python 脚本（`静态代码核对，未本地运行`）。
- **通读：** 已完成

## 开源范围与关键路径

| 路径 | 职责与内容 |
|---|---|
| `vla-perf/pi0_perf.py` | 论文核心入口脚本：执行全部 7 组 $\pi_0$ 性能实验，结果输出至 `vla-perf/perf_results/` |
| `vla-perf/openvla_perf.py` | OpenVLA（双视觉编码器 DINOv2 + SigLIP 与 Llama 2 7B 自回归解码）性能评估脚本 |
| `vla-perf/perf_utils.py` | VLA 性能评估公共工具函数与指标汇总逻辑 |
| `vla-perf/network_latency.py` | WiFi 5/6/6E/7、4G/5G、Ethernet（1G–400G）及云链路网络传输延迟计算 |
| `vla-perf/plot_scripts/` | 论文图表生成脚本（如 `print_1_base_pi0.py`、`print_2_scale_model.py`、`print_3_long_context.py` 等），输出至 `vla-perf/paper_figures/` 与 `vla-perf/paper_tables/` |
| `genz/GenZ/Models/Model_sets/vla_models.py` | 面向 VLA 工作负载扩展的模型定义（$\pi_0$ 视觉/VLM/Action Expert、$\pi_{0.6}$、OpenVLA、SigLIP/DINOv2） |
| `genz/Systems/system_configs.py` | 扩展的硬件加速器规格配置（新增 NVIDIA B100、RTX 4090 及 Jetson AGX Thor / Orin / Xavier 等，并保留 A100/H100） |
| `genz/GenZ/LLM_inference/llm_parallel_decode.py` | 新增并行解码建模模块：模拟扩散式 Action Expert 每步去噪对多动作 token 并行解码及对 VLM KV cache 的交叉注意力 |
| `genz/GenZ/LLM_inference/llm_prefill.py` / `llm_decode.py` | Prefill 建模（视觉编码器与 VLM 主干）与逐 token Decode 建模（自回归动作头，如 OpenVLA） |
| `genz/GenZ/analyse_model.py` / `genz/GenZ/system.py` | GenZ 算子分解、Roofline 计算与系统并行策略（TP/PP）抽象核心 |

## README 给出的复现入口与依赖关系

```bash
# 1. 创建 Python 3.10 环境并安装扩展版 GenZ 后端
conda create -n vla-perf python=3.10
conda activate vla-perf
cd genz && pip install -e .

# 2. 运行论文全部 7 组 Pi0 实验（结果写入 vla-perf/perf_results/）
cd ../vla-perf
python pi0_perf.py

# 3. 生成论文图表与 LaTeX 表格
cd plot_scripts
python print_1_base_pi0.py
python print_2_scale_model.py
python print_3_long_context.py
```

- **上游依赖说明：** `genz/` 目录基于 `GenZ-LLM-Analyzer`（<https://github.com/abhibambhaniya/GenZ-LLM-Analyzer> / <https://github.com/astra-sim/GenZ-LLM-Analyzer>）扩展而来，但仓库已将其内嵌修改并打包为本地可安装后端；`code:` 归属应指向 `NVlabs/vla-perf` 本身，而非上游 GenZ 仓库。
- **复现边界：** 静态代码核对，未本地运行。该工具为解析性能预测器，不实际加载 VLA 权重或连接真机/仿真器执行控制，因此不依赖高显存 GPU 即可计算，但预测值为理论 Roofline 上界（论文实测对比 RTX 4090 Triton 实现的保真度为 73.3%–82.6%）。

## 映射

- [原论文归档](../papers/paper-agentic-2602-18397.md)
- [唯一主 Wiki 节点](../../wiki/entities/paper-agentic-2602-18397.md)
