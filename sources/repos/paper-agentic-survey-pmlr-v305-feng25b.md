# ReflectVLM 官方代码仓归档（yunhaif/reflect-vlm）

- **官方仓库：** https://github.com/yunhaif/reflect-vlm
- **关联论文：** [Reflective Planning: Vision-Language Models for Multi-Stage Long-Horizon Robotic Manipulation（CoRL 2025 / PMLR v305 / arXiv:2502.16707）](../../wiki/entities/paper-agentic-survey-pmlr-v305-feng25b.md)
- **开源协议：** MIT License（见仓库 `LICENSE`；内含 LLaVA 代码遵循 Apache 2.0 协议，仿真环境部分改编自 `Farama-Foundation/Metaworld` 与 `kevinzakka/mjctrl`）
- **默认分支与体量：** `main`（截至采集记录共 7 commits，182 stars，4 forks）
- **开源状态判定：** **部分开源（推理评测、MuJoCo 仿真环境与三套模型权重已发布；后训练脚本待补充）**
  - 已发布内容：Python 包结构与依赖声明、基于 MuJoCo 的程序化桌面装配仿真与交互工具、基线与反思策略评测脚本（支持 `sim` 与 `diffusion` 两种前瞻动力学模式）、多类智能体封装（`llava`、`diffuser`、`gemini`、`gpt`、`oracle`、`random`），以及 Hugging Face 合集 `yunhaif/reflectvlm-67b95e4316ab2d5f71ad4b25` 下的三个检查点（`ReflectVLM-llava-v1.5-13b-base`、`ReflectVLM-llava-v1.5-13b-post-trained`、`ReflectVLM-diffusion`）。
  - 待发布内容：README 中 `## Policy training` 明确标注为 `Coming soon...`，虽仓库捆绑了上游 `llava/train/` 训练源码，但尚未提供论文对应的 3 轮 DAgger 交互式反思后训练与 DDM 数据/训练完整流水线脚本。
- **核对边界：** 静态代码核对，未本地运行（依据仓库主页目录结构、`files_sample` 清单及 `README.md` 全文核对入口脚本与模块分工）。

## 仓库结构与核心模块（基于实测文件清单）

- **安装与根入口：**
  - `pyproject.toml`：通过 `pip install -e .` 安装运行环境（Python 3.9），可选 `pip install -e ".[train]"` 与 `flash-attn`。
  - `run.py`：根目录运行入口脚本。
- **交互仿真与评测脚本（`scripts/`）：**
  - `scripts/interact.py`：启动 MuJoCo 交互式可视化环境，支持通过 `--env_seed`（如 `--env_seed 1000001`）按种子程序化生成不同积木装配任务，并通过命令行输入动作原语（macOS 提示使用 `mjpython`）。
  - `scripts/eval_base_vlm.sh`：在 100 个程序化生成测试任务上评测固定专家数据集训练的基础 VLM 策略（自动从 Hugging Face 拉取 `ReflectVLM-llava-v1.5-13b-base`）。
  - `scripts/eval_reflect_vlm.sh {sim|diffusion}`：评测反思后训练 VLM 策略（`ReflectVLM-llava-v1.5-13b-post-trained`），通过参数选择使用 MuJoCo 仿真器（`sim`）或扩散动力学模型（`diffusion`，对应 `ReflectVLM-diffusion`）执行前瞻想象。
  - `scripts/diffusion_demo.py`：测试扩散动力学模型图像预测生成，样例输出图像存于 `assets/images/diffusion_examples/`。
- **智能体与仿真环境包（`roboworld/`）：**
  - `roboworld/agent/`：包含 `llava.py`（VLM 策略封装）、`diffuser.py`（扩散动力学前瞻封装）、`gemini.py` 与 `gpt.py`（商用 VLM 零样本评测接口）、`oracle.py`（特权符号专家策略）、`random.py` 与 `utils.py`。自定义智能体只需在 `act(img, goal_img, inp)` 接口中接收当前图、目标图与提示文本并返回动作字符串。
  - `roboworld/envs/`：MuJoCo 环境构建模块（`__init__.py`、`asset_path_utils.py`）与 Franka 机械臂模型资产（`roboworld/envs/assets/franka/`，含 `.obj`/`.stl` 网格与独立 `LICENSE`），常量配置位于 `roboworld/constants.py`。
- **底层 VLM 底座（`llava/`）：**
  - 完整嵌入 `haotian-liu/LLaVA`（`llava/model/`、`llava/mm_utils.py`、`llava/conversation.py`、`llava/train/train.py`、`llava/serve/` 等），用于加载 `llava-v1.5-13b` 及其 LoRA 适配权重。

## 对 wiki 的映射

- [ReflectVLM：面向多阶段长程机器人操作的视觉-语言模型反思规划](../../wiki/entities/paper-agentic-survey-pmlr-v305-feng25b.md)
