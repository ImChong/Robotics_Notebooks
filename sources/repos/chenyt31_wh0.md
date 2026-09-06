# chenyt31/Wh0

> 来源归档

- **标题：** Wh0
- **类型：** repo
- **组织：** chenyt31（南京大学 Yang Gao 组相关）
- **代码：** <https://github.com/chenyt31/Wh0>
- **项目页：** <https://chenyt31.github.io/wh0.github.io/>
- **论文：** <https://arxiv.org/abs/2606.22136>
- **Stars：** ~24（2026-09-06）
- **入库日期：** 2026-09-06
- **一句话说明：** **WM-H 合成 + VITRA 共训** 官方实现：`WM-H/` 生成 Wan/Qwen 管线，`vitra-wh0/` 策略微调/评测；`scripts/agent_run.sh` 一键 quick 流程（10 视频 + 100 step smoke）。
- **沉淀到 wiki：** [`wiki/entities/paper-sa-2606-22136-wh0-generative-world-models-as-scalable-sources.md`](../../wiki/entities/paper-sa-2606-22136-wh0-generative-world-models-as-scalable-sources.md)

## 开源边界（步骤 2.5）

| 项 | 结论 |
|----|------|
| **状态** | **已开源**（Research use only；上游模型许可另计） |
| **代码** | <https://github.com/chenyt31/Wh0> |
| **依赖权重** | `weights/` 管理 VITRA、HaWoR、Wan、Qwen、PaliGemma、MANO 等；`scripts/download_weights.sh` |
| **G1 遥操作** | 集成 Unitree [`xr_teleoperate`](https://github.com/unitreerobotics/xr_teleoperate)（`vitra-wh0/thirdparty/`） |

## README 要点（2026-09-06）

### 两仓结构

| 目录 | 角色 |
|------|------|
| **`WM-H/`** | 合成操纵视频：指令 → 场景编辑 → Wan I2V → HaWoR 标注 → 可选 robot-hand 编辑 |
| **`vitra-wh0/`** | 基于 [Microsoft VITRA](https://github.com/microsoft/VITRA) 的训练 / 推理 / 评测 |

### 端到端脚本顺序

1. `configs/project_request.yaml` — 路径、权重、WM-H profile、训练步数  
2. `scripts/run_wmh.sh` / `run_all.sh --stage wmh` — 生成视频  
3. `scripts/run_annotate.sh` — HaWoR 标注  
4. `run_all.sh --stage hand_edit` — Qwen-Image-Edit 机器人手（默认每 4 帧）  
5. `run_all.sh --stage prepare_data` — VITRA 训练树（默认 **20%** 链到 edited 视频）  
6. `scripts/run_train.sh` — Co-FT / 微调  
7. `scripts/run_eval_pipeline.sh` — pred vs GT 可视化  

输出根：`WM-H/database/wm-h/instr_first/streaming_runs/<run_id>/`；多卡自动 merge 为 `<run_id>_merged/`。

### 环境门槛

- Linux + CUDA + [uv](https://docs.astral.sh/uv/)  
- 主流程需 **≥1× 80GB GPU**（WM-H 生成 + 训练）  
- 策略栈测试：**torch==2.6.0+cu124**  
- Qwen3-VL FP8 建议独立 **vLLM** 环境（`WMH_PYTHON`）；`vllm==0.8.5` 不可靠  

### Quick start

```bash
bash scripts/agent_run.sh --reconfigure
bash scripts/agent_run.sh --run quick   # 10 videos + 100 train steps
bash scripts/agent_run.sh --run all
```

## 对 wiki 的映射

- 实体页：[paper-sa-2606-22136-wh0-generative-world-models-as-scalable-sources.md](../../wiki/entities/paper-sa-2606-22136-wh0-generative-world-models-as-scalable-sources.md)
- 论文：[wh0_arxiv_2606_22136.md](../papers/wh0_arxiv_2606_22136.md)
- 项目页：[wh0-project.md](../sites/wh0-project.md)
