# nv-tlabs/ardy

> 来源归档

- **标题：** ARDY — Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation
- **类型：** repo
- **组织：** NVIDIA Toronto AI Lab（nv-tlabs）
- **代码：** <https://github.com/nv-tlabs/ardy>
- **论文：** <https://arxiv.org/abs/2607.08741> · [PDF](https://research.nvidia.com/labs/sil/projects/ardy/assets/ardy_paper.pdf)
- **项目页：** <https://research.nvidia.com/labs/sil/projects/ardy/>
- **模型集合：** <https://huggingface.co/collections/nvidia/ardy>
- **Stars：** ~885（2026-09-06）
- **入库日期：** 2026-09-06
- **一句话说明：** SIGGRAPH 2026 **交互式自回归扩散** 人体运动官方实现：`ardy_demo` / `generate.py`、TensorRT 可选、G1/Core 多 checkpoint；Apache-2.0 代码 + NVIDIA Open Model 权重。
- **沉淀到 wiki：** [`wiki/entities/ardy.md`](../../wiki/entities/ardy.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **代码** | **已开源** — [nv-tlabs/ardy](https://github.com/nv-tlabs/ardy)（Apache-2.0） |
| **权重** | **已发布** — Hugging Face [nvidia/ARDY-*](https://huggingface.co/collections/nvidia/ardy)（NVIDIA Open Model License） |
| **文本编码器** | 依赖 **gated** [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)（需 HF token） |
| **约束 Demo 数据** | 可选 [Bones SEED](https://huggingface.co/datasets/bones-studio/seed) CSV（G1 约束采样） |

## README 要点（2026-09-06）

### 环境与安装

- 测试环境：**Ubuntu 22.04 · RTX 4090 · driver 575 · Python 3.11**
- `pip install torch>=2.4`（按 CUDA 自选）→ `pip install -e ".[all]"`（含 demo + TensorRT + C++ motion-correction 扩展，需 CMake/C++17）
- 部分安装：`pip install -e .`（推理核）/ `.[demo]` / `.[trt]`

### 已发布 Checkpoint（HF 自动下载）

| 模型 | 骨架 | 数据 | FPS | Horizon |
|------|------|------|-----|---------|
| ARDY-Core-RP-20FPS-Horizon40 | Core 27-joint | Bones Rigplay 1 | 20 | 40 |
| ARDY-Core-RP-20FPS-Horizon8 | Core | Rigplay 1 | 20 | 8 |
| ARDY-G1-RP-25FPS-Horizon52 | Unitree G1 | Rigplay 1 | 25 | 52 |
| ARDY-G1-RP-25FPS-Horizon8 | G1 | Rigplay 1 | 25 | 8 |

**Coming soon：** SOMA skeleton 变体（[SOMA-X](https://github.com/NVlabs/SOMA-X)）。

### 入口脚本

| 脚本 | 用途 |
|------|------|
| `scripts/run_demo.py` | 浏览器交互 Demo（`http://localhost:2333`）：流式文本、路点/键盘 locomotion、约束采样 |
| `scripts/generate.py` | CLI 批量文生运动 → `.npz`（G1 另导出 MuJoCo qpos `.csv`） |
| `scripts/visualize.py` | 回放生成结果（`:2334`） |
| `scripts/run_text_encoder_server.py` | 后台 LLM2Vec 服务，避免重复加载 |

### 运行时 API（Demo 集成）

- **加载：** `ardy/model/load_model.py` + `load_text_encoder()`；registry 昵称 `core`/`g1`/`core8`/`g152` 等
- **生成：** `Ardy.autoregressive_step()` — text embedding + `motion_mask`/`observed_motion` 约束 → 解码 `motion_rep.inverse(...)`

### 延迟（论文 / README 语境）

- Demo 工作站 RTX 4090：**4-step 扩散 ~33 ms**；10-step ~63 ms（窗口 G=40 @ 20fps）

## 对 wiki 的映射

- 实体：[`wiki/entities/ardy.md`](../../wiki/entities/ardy.md)
- 论文：[`sources/papers/ardy_siggraph_2026.md`](../papers/ardy_siggraph_2026.md)
- 项目页：[`sources/sites/ardy-project.md`](../sites/ardy-project.md)
