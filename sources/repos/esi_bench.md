# ESI-Bench（官方实现与数据）

> 来源归档

- **标题：** ESI-Bench: Towards Embodied Spatial Intelligence that Closes the Perception-Action Loop
- **类型：** repo + benchmark toolkit + HF dataset
- **组织：** Stanford / UCLA / Northwestern 等（见论文）
- **代码：** <https://github.com/ESI-Bench/ESI-Bench>
- **论文：** <https://arxiv.org/abs/2605.18746>
- **项目页：** <https://esi-bench.github.io/>
- **Hugging Face 数据：** <https://huggingface.co/datasets/esi-bench/ESI-Bench>
- **许可证：** MIT（README badge）
- **入库日期：** 2026-05-22
- **核查日期：** 2026-11-07（官方 README / 运行入口 / 任务注册表 / 关键模型接口）
- **开源状态：** 已开源（官方 GitHub 仓库 MIT License；Hugging Face 题目与被动单视图数据公开）
- **一句话说明：** 开源 **主动探索** 运行器 + **dataset/json_clean** 任务 JSON + **dataset_generation** 构造脚本；依赖 **`behavior` conda 环境** 与本地 **OmniGibson / BEHAVIOR-1K** 资产，逐步截图后调用 **OpenAI / Gemini** API 写 `answer.json`。

## 仓库结构（README 归纳）

```
esi-bench/
├── dataset/json_clean/          # 按任务类分目录的 question JSON
├── src/active_explore/          # OmniGibson 场景 + 逐步图像 + MLLM + answer.json
├── src/dataset_generation/      # 数据集构造
└── outputs/                     # 结果与 step 图像（gitignore）
```

**任务 JSON 顶层目录名（与论文类对应）：** Action Sequencing、Cognitive Mapping、Enumerative Perception、Metric Comparison、Perceptual Grounding、Physical Dynamics、Physical Structure（站页 Physical Capacity）、Spatial Relations、Specular Reflection、Temporal Understanding（站页 Temporal Scene）。

## 环境与运行（摘要）

1. `conda activate behavior`（README 假定已有 BEHAVIOR / OmniGibson 安装）。
2. 设置 `OPENAI_API_KEY` 或 `GEMINI_API_KEY`。
3. 官方示例：`python src/main.py --task spatial_segmentation --metadata "dataset/json_clean/Enumerative Perception/Spatial Segmentation/Merom_0_int/living_room_0/q_000.json" --provider gemini --model gemini-3.1-pro-preview --max-steps 30 --min-steps 1 --threshold 0.9 --results-root outputs/results --step-image-root outputs/steps --overwrite`。`counting` 是旧版兼容别名，实际映射到 `category_ambiguity`；新任务请使用 29 子任务标准名。

## 对 wiki 的映射

- 实体页：[ESI-Bench](../../wiki/entities/esi-bench.md) — 复现门槛、目录与评测范式。
- 交叉： [3D 空间 VQA](../../wiki/concepts/3d-spatial-vqa.md)、[视觉–语言导航](../../wiki/tasks/vision-language-navigation.md)、[WEM / HTEWorld](../../wiki/entities/paper-wem-world-ego-modeling.md)（同属 BEHAVIOR-1K / OmniGibson 生态）。

## 2026-11-07 源码核查：真实执行路径

1. [`src/main.py`](https://github.com/ESI-Bench/ESI-Bench/blob/main/src/main.py) 将 `src/active_explore` 放入导入路径并调用 `pipeline.main(argv)`。
2. [`pipeline.py`](https://github.com/ESI-Bench/ESI-Bench/blob/main/src/active_explore/pipeline.py) `parse_args` / `select_question` 解析单题 JSON 或 `json_paths` 列表；`load_task_module_for_payload` 使用 [`tasks/_registry.py`](https://github.com/ESI-Bench/ESI-Bench/blob/main/src/active_explore/tasks/_registry.py) 分发到对应任务模块。
3. `run_one` 按任务模块创建 `og.Environment(...)`（可由 task hook 修改物体与相机），获取当前 RGB 截图；`models/gemini.py` 或 `models/gpt.py` 的 `generate_json` 返回动作、答案、置信度，部分模块有自定义响应 schema、停止规则和操作 hook。
4. 未停止时由 `execute_action` 修改虚拟相机位置或执行任务特定仿真操作，进入下一步观测；停止后通过 `task_score` 评分，写入每题 `*_answer.json` 和 `step_*.png`。
5. **验证边界：** 官方 README 说明 `max_steps=1` smoke test 验证了环境、渲染、模型调用和结果写入，**不等于**多步物理操作/全基准准确率已通过；本次仅完成静态源码核对，**没有实际执行 OmniGibson/GPU benchmark**。

### 运行环境风险

- 需要已配置的 BEHAVIOR / OmniGibson 资产与 `behavior` conda 环境、模型 API key；需按 README 的 `make_maps.py` 指南使地图结构类别仅含 floor，以符合该评测设置。
- 官方 README 提醒其环境和 BEHAVIOR-1K 在 50 系 / Blackwell GPU 上可能出现严重渲染问题；这是**作者给出的兼容性提示**，不应延伸为独立硬件性能实测。
- 论文项目页两处旧分类命名 `Physical Capacity / Temporal Scene` 与数据目录 `Physical Structure / Temporal Understanding` 存在差异，按注册表和 JSON 路径运行任务。

### 上游核验材料

- [论文 arXiv v2](https://arxiv.org/html/2605.18746) · [项目站](https://esi-bench.github.io/) · [官方 README](https://github.com/ESI-Bench/ESI-Bench/blob/main/README.md) · [任务注册表](https://github.com/ESI-Bench/ESI-Bench/blob/main/src/active_explore/tasks/_registry.py) · [Gemini 适配器](https://github.com/ESI-Bench/ESI-Bench/blob/main/src/active_explore/models/gemini.py)。
