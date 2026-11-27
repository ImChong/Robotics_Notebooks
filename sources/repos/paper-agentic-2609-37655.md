# yingjiayu12/Exemplar2VQA 官方代码核查

- **代码仓库：** <https://github.com/yingjiayu12/Exemplar2VQA>
- **开源协议与分支：** `CC-BY-4.0` 协议，默认分支 `main`
- **对应论文：** [Exemplar2VQA: A Scalable Exemplar-Driven Visual Question Answering Generation Framework via Multi-Agent Coding (arXiv:2609.37655v1)](../papers/paper-agentic-2609-37655.md)
- **核对范围：** 静态核对 GitHub 仓库元数据、`README.md` 预览及 `files_sample` 目录文件清单（覆盖 `run_pipeline.sh`、`shared/`、`Track1/` 与 `Track2/` 全部核心入口及工具模块）。静态代码核对，未本地运行。
- **通读：** 未完成
- **开源范围判断：** **已开源** 完整的双轨多智能体代码生成与数据流水线源码，包括端到端脚本 `run_pipeline.sh`、Track 1 AI2-THOR 相机采集管道、Track 2 数据桥接与 QA 生成管道、几何工具集及 3 个 Holodeck 兼容示例场景（`scene_0`–`scene_2`）；运行时所需的 `Qwen3-Coder-30B-A3B-Instruct` 权重（或 OpenAI 兼容 API）、AI2-THOR 特定提交版本及 Holodeck/Objaverse 3D 资产包需按安装说明另行拉取。

## 已核对目录与入口

1. **端到端与公共后端入口：**
   - `run_pipeline.sh`：串联 Track 1 与 Track 2 的总入口脚本，支持传入自定义场景目录、Track 1 输出目录、Track 2 数据目录与任务键（如 `A`）。
   - `shared/llm_backend.py`（及 `Track2/Track2_pipeline/llm_backend.py`）：统一模型推理后端封装，通过环境变量 `EXEMPLAR2VQA_LLM_BACKEND` 在 `vllm`（默认多卡 `Qwen/Qwen3-Coder-30B-A3B-Instruct`）、`transformers`（单卡开发）与 `openai`（兼容 `/v1/chat/completions` 接口）之间切换，无需改动提示词。
2. **Track 1 相机生成轨道（`Track1/`）：**
   - `Track1/Track1_Pipeline/run.sh`：Track 1 入口，默认读取 `Track1/data/scenes` 并输出至 `Track1/data/collected`。
   - `Track1/Track1_Pipeline/Instructions.py` 与 `Prompt.py`：定义自然语言相机采集任务指令与各智能体提示词。
   - `Track1/Track1_Pipeline/camera_skills.py` 与 `camera_skills_reference.py`：封装 AI2-THOR 仿真器加载、去重物体元数据提取、位姿采样与可见物体记录接口。
   - `Track1/Track1_Pipeline/run_camera_agent.py`：执行 Architect 语义规划与 Coder 相机脚本文档生成阶段。
   - `Track1/Track1_Pipeline/run_camera_reviewer.py`：执行物理引擎碰撞/越界回溯诊断（Reviewer）与代码修补循环（Refiner）。
   - `Track1/data/scenes/scene_{0,1,2}/scene_{0,1,2}.{json,png}`：随仓附带的 3 个 Holodeck 格式示例室内场景。
3. **Track 2 QA 生成轨道与跨轨道桥接（`Track2/`）：**
   - `Track2/Track2_pipeline/run.sh`：Track 2 主入口，自动发现 Track 1 产物、执行规范化迁移、提取 schema 并逐场景运行四阶段流水线，最终写出 `qa_<task_name>_holodeck.json`。
   - `Track2/Track2_pipeline/prepare_data.py`：Track 1 → Track 2 数据契约桥接脚本，将嵌套输出整理为 `<scene_name>_bbox.json`、`<scene_name>_visible_views.json` 及 `image_path` 引用的 RGB 图像。
   - `Track2/Track2_pipeline/1_extract_schema.py`：提取紧凑的场景与可见视角 JSON Schema（参考示例 `data/scene_0/schema_scene_0_bbox.json`、`schema_scene_0_visible_views.json`）。
   - `Track2/Track2_pipeline/2_architect_plan.py`、`3_coder_generate.py`、`4_code_refiner.py`：对应 QA Architect（从 `Example.py` 读取示例抽象为 `tasks.json`）、QA Coder（调用 `Utils.py` / `base_qa_generator.py` 合成代码）与 QA Reviewer/Refiner（Python 运行时自纠错）。
   - `Track2/question_templates.py`、`Track2/format_qa.py`、`Track2/base_processor.py` 及 `Track2/utils/`（含 `common_utils.py`、`trajectory_utils.py`、`vsibench_utils.py`、`arkitscenes_utils.py`、`scannet200_utils.py`、`scannetpp_utils.py`、`scannetv2_utils.py`）：面向多基准格式转换与空间几何处理的辅助工具集。

## 边界说明

- 本归档仅基于公开仓库文件结构与 `README.md` 完成静态代码核对，**未在本地配置 Linux CUDA 环境、vLLM、AI2-THOR（`ai2thor==0+6f165fdaf3cf2d03728f931f39261d14a67414d0`）或 Holodeck/Objaverse 资产执行实际生成**。
- `README.md` 明确说明该流水线面向 Linux CUDA 环境（Windows 需通过 WSL2），且下游微调训练（`swift` / DeepSpeed）与评测脚本需另行对接。
- 论文同时提及的 `allenai/Holodeck`、`QwenLM/Qwen3-Coder` 与 `vllm-project/vllm` 属于上游依赖场景生成器、基座模型及推理引擎，并非本项目的自有源码仓。

## 映射

- [Exemplar2VQA 论文实体与源码运行时序图](../../wiki/entities/paper-agentic-2609-37655.md)
