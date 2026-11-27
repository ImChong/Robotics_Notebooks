# sygkounas/MEMENTO — MEMENTO 官方开源仓库档案

- **GitHub 地址：** https://github.com/sygkounas/MEMENTO
- **默认分支：** `main`
- **开源协议：** 仓库元数据未标明开源协议（`license` 字段为空）
- **关联论文：** *MEMENTO: Memory-Guided Memetic Code-as-Policy Evolution*（[arXiv:2607.22832](https://arxiv.org/abs/2607.22832)）
- **核对边界：** 静态代码核对，未本地运行（基于 `context.json` 中采集的仓库元数据、`README.md` 与 `files_sample` 文件列表核查，未在本地执行仿真或真机试验）。

## 核心入口与目录结构（已核实）

- **主演化管线与核心代码（`MEMENTO/`）：**
  - `MEMENTO/main.py`：运行 MEMENTO 演化流水线的主入口。
  - `MEMENTO/utils.py`：策略与评估器的执行、评估及辅助工具函数。
  - `MEMENTO/inference.py`：大模型查询与代码生成推理工具。
  - `MEMENTO/env_tower_heavy_dr.py`：Robosuite Franka Tower-of-Hanoi 重度域随机化仿真环境。
  - `MEMENTO/cfg/config.yaml`、`MEMENTO/cfg/generate.yaml`：管线与生成配置文件。
  - `MEMENTO/prompts/`：各阶段提示词模板，包括 `policy_init.txt`（初始策略）、`fitness_prompt.txt`（评估器生成/变异）、`majority_prompt.txt`（评估器多数投票判断）、`hill_climb_prompt.txt`（爬山微变异）、`macro_mutation_prompt.txt`（宏变异）、`crossover_prompt.txt`（交叉重组）。
- **物理 Franka 真机部署模块（`hanoi/`）：**
  - `hanoi/hanoi_fsm.py`：Tower-of-Hanoi 有限状态机控制逻辑。
  - `hanoi/hanoi_node.py`：物理 Franka 机械臂执行节点。
  - `hanoi/objectgrounding.py`：基于 RGB-D 与分割模型的桌面物体定位（Object Grounding）。
- **最优演化策略产物（`best_policies/`）：**
  - `best_policies/franka_robotsuite_policy.txt`：Robosuite Tower-of-Hanoi 最优演化策略程序。
  - `best_policies/thor_policy.txt`：AI2-THOR 家居交互任务最优演化策略程序。
- **演示素材与项目页（`assets/`、`videos/`、`docs/`）：**
  - `assets/`：`physical_franka_teaser.gif`、`robosuite_teaser.gif`、`thor_demo1.gif`、`thor_demo2.gif`。
  - `videos/`：完整演示视频（`Robot_Demo_6x_github_24MB.mp4`、`robotsuite_demo_cropped_10x.mp4`、`thor_demo1`、`thor_demo2`）。
  - `docs/`：静态展示页面及结果图表（含 `docs/index.html` 与 `docs/assets/`）。
- **依赖配置：**
  - `requirements.txt`（README 安装命令：`pip install -r requirements.txt`）。

## 复现与运行边界

- **大模型 API 依赖：** 论文实验使用 GPT-5.2 生成评估器与策略代码提案，运行演化流水线需配置对应模型推理接口（`MEMENTO/inference.py`）。
- **仿真与真机环境依赖：** Robosuite 实验依赖 MuJoCo/Robosuite 及 Franka Panda 模型；AI2-THOR 实验依赖 AI2-THOR 仿真器及 `FloorPlan1`–`FloorPlan29`；真机部署依赖 Franka 机械臂控制栈、RGB-D 相机及 SAM3 物体分割管线。
- **未核实范围：** 仅核对了仓库目录清单与 README 说明，未拉取完整源码内容逐行执行单元测试或实测复现。

## 与 wiki 的映射

- 对应实体页：[MEMENTO: Memory-Guided Memetic Code-as-Policy Evolution](../../wiki/entities/paper-agentic-2607-22832.md)
