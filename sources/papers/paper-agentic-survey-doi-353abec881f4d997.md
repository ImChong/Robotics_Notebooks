# SCOPE: Real-Time Natural Language Camera Agent at the Edge: A Sim-to-Real Benchmark and Analysis of Open-Source Vision and Language Agents for PTZ Camera Tasks

- **作者：** Nikolaj Hindsbo, Sina Ehsani, Pragyana Mishra
- **机构：** Armada AI（Bellevue, Washington, USA）
- **版本 / 载体：** Proceedings of the 21st ACM/IEEE International Conference on Human-Robot Interaction (HRI '26), March 16–19, 2026, Edinburgh, Scotland, UK；DOI: `10.1145/3757279.3785641`；ISBN: `979-8-4007-2128-1/2026/03`；预印本 `arXiv:2606.02951v1 [cs.RO]`（2026-06-01，Received 2025-12-01，CC BY 4.0）
- **原文：** [ACM DOI 入口](https://doi.org/10.1145/3757279.3785641) / [arXiv:2606.02951](https://arxiv.org/abs/2606.02951)（核读文本为 arXiv LaTeXML 全文归档）
- **项目页 / 代码：** **已开源**（官方 GitHub 仓库 [HindsboNikolaj/SCOPE](https://github.com/HindsboNikolaj/SCOPE)，MIT 协议；含完整智能体运行时 `src/scope/`、536 任务评测集 v1/v2、Docker 镜像及 2026 年 11 月发布的 v0.2.0 并行 GPU 评测工具；4 个 Blender 场景文件托管于 [Hugging Face 数据集 `HindsboNikolaj/scope-benchmark`](https://huggingface.co/datasets/HindsboNikolaj/scope-benchmark)）
- **核对范围：** 通读论文全文 1–8 节（Abstract、1 Introduction、2 Related Work、3 System Architecture 3.1–3.4、4 Benchmark Construction 4.1–4.4、5 Evaluation Methodology 5.1–5.2、6 Results 6.1–6.4、7 Limitations、8 Discussion and Future Work 与 References），完整核对 Table 1–Table 6 与 Figure 1–Figure 4；同时对照官方仓库 `README.md` 与核心源码入口。
- **通读：** 已完成

## 原文核对笔记

- **研究动机与背景：** 将自然语言指令（如“去公路预设位，然后每步向右旋转 15° 直到看到至少 6 个锥桶”）转化为云台变焦相机（PTZ）的闭环感知-规划-执行动作，需要在边缘侧受限显存与延迟预算下协调语言规划、视觉感知与机械执行。论文前作 OPUS（Buynitsky et al., 2025）依赖少样本工具调用和 YOLO 检测器，新增工具需重新生成数据并微调且查询受限；一般多模态或工具调用基准则缺乏闭环 PTZ 动作空间与边缘延迟指标。
- **解耦架构与状态机（第 3 节）：**
  - **规划-感知解耦：** 由紧凑型小语言模型（SLM）担任高层规划器，仅通过 OpenAI 兼容 JSON Schema 调用工具，不直接向多轮对话流式塞入图像 Token（避免单帧数百至数万 Token 随轮次累积），仅在需要时调用轻量级视觉-语言模型（VLM）工具并接收简短结构化文本结果（计数、OCR 文本、属性或定位动作反馈）。
  - **离散时间递归状态机（3.1 节）：** 状态 $S_t = (U, H_t, O_t)$，每步规划器输出最终文本回答或结构化工具调用 $\tau_t$，获得结果 $R_t$ 后更新历史 $H_{t+1} = H_t \,\|\, [\tau_t, R_t]$ 与观测日志 $O_{t+1} = O_t \,\|\, [R_t]$，直至输出答案或达到步数上限 $T_{\max}$。
  - **9 类工具接口（Table 1）：**
    - 直接相机控制（5 个）：`ADJUST_PTZ`（相对 pan/tilt/zoom）、`GO_TO_PRESET`、`GO_HOME`、`GET_PRESETS`、`TAKE_IMAGE`。
    - 视觉感知工具（4 个）：`COUNT_OBJECTS`、`QUERY_ANSWER`、`ZOOM_TO_OBJECT`、`TRACK_OBJECT`。
- **真机与 Blender 数字孪生（3.3–3.4 节）：**
  - 真机后端封装 AXIS M5075-G PTZ 相机并配备 Streamlit 界面；仿真后端在 Blender 中固定相机位置 $(x, y, z)$，开放 yaw/pitch 旋转与焦距调节，暴露与真机 1:1 一致的 JSON Schema 与工具名。
  - 为避免多帧时序差异与视角重叠导致重复计数，实现确定性全景扫描拼接工具（panorama tool），支持 `current`（当前视野）与 `full`（360° 拼接全景）两种观测范围。
- **536 任务基准构建与评测协议（第 4–5 节）：**
  - 基于 4 个公开城市 Blender 场景（`book-nook` 170 题、`city-street` 120 题、`postwar-city` 109 题、`whitechapel` 137 题）及多组固定预设视角构建标注，并显式记录“不存在的对象/属性”以支持否定存在性提问；结合真机日志从初版 541 题经三轮审查精炼为 **536 题**（Table 2 分 8 类：Counting 95、Descriptor 89、Single Call 72、Comparative Relational 62、Multi-step Command 57、Multi-step Reasoning 54、OCR Identification 54、Location / Spatial 53）。
  - 采用 `gpt-oss-120B` 作为统一 LLM-as-Judge 对完整执行轨迹判定二值正确性，并在错误时归因至 Table 3 的 8 类互斥错误模式（`VLM – Query`、`VLM – Counting`、`Reasoning`、`Hallucination`、`Lack of Tool Call`、`Tool Args`、`Scope`、`Tool Routing`）。
- **核心实验结论（第 6 节，Table 5–Table 6）：**
  - **19 组配对评测最高分：** `Qwen3-30B-A3B` + `Moondream3` 在 8 类平均准确率达到 **73.8%**（Single Call 98.6%、Multi-step Command 98.2%、Multi-step Reasoning 79.6%、OCR 77.8%、Counting 76.8%、Location/Spatial 66.0%、Comparative Relational 56.5%、Descriptor 37.1%）。
  - **瓶颈转移与错误拆解（6.1 节）：** 弱规划器（如 4B）常因幻觉与工具参数/路由错误失败；MoE 规划器显著降低工具与推理错误后，视觉感知成为首要瓶颈。在同一 VLM 下切换 SLM 带来 5.6–12.4% 波动，跨 VLM 最大波动 6.9%；若将 `VLM - Counting` 与 `VLM - Query` 视为 Oracle 正确，平均准确率从 **66.2%** 跃升至 **82.0%（+15.8 个百分点）**，表明视觉感知失败平均占总错误的 **46.7%**。
  - **全景范围影响（6.2 节）：** 当前视野查询平均准确率 **66.9%**，全场景拼接全景查询降至 **58.3%**，且拼接图输入使中位延迟从 **6.07 s** 升至 **7.43 s（+22.4%）**。
  - **延迟与架构权衡（6.3–6.4 节，Table 6）：** MoE 规划器（`Qwen3-30B-A3B` 1.39 s、`Qwen3-Next-80B-A3B` 1.42 s）在保持接近 `Qwen3-4B`（1.94 s）且显著快于稠密 `Qwen3-32B`（3.00 s）的推理耗时下获得更高精度；VLM 占主导耗时（5.46–9.18 s，含共享集群与未优化自托管开销；本地独立 A100-80GB 上轻量 VLM 单次推理达 200–500 ms，满足约 2 Hz 实时门槛）。FP8/INT4 量化在多数配置上保持准确率，但 A100 推理路径受限于算子支持可能出现延迟反升（如 `Qwen3-30B-A3B-FP8` 2.62 s vs FP16 1.39 s）。
- **仓库演进补充（README v0.2.0，2026 年 11 月）：** 公开仓库新增修正版数据集 v2（`benchmark/scope_v2.csv`，仍为 536 题）、三评委投票脚本（`scripts/13_judge_panel.py`）与支持 6 并发 GPU 离屏渲染的 `scripts/run_fast.py`（A100 80GB 全量 536 题耗时由 0.1.0 的 157 min / 单帧 14.6 s 降至 30 min / 单帧 0.2 s）。
- **对 wiki 的映射：** [SCOPE 论文实体页](../../wiki/entities/paper-agentic-survey-doi-353abec881f4d997.md)
