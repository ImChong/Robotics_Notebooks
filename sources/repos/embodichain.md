# DexForce/EmbodiChain

- **标题：** EmbodiChain
- **类型：** repo
- **来源：** 灵巧智能（DexForce Technology Co., Ltd.）
- **链接：** https://github.com/DexForce/EmbodiChain
- **文档：** https://dexforce.github.io/EmbodiChain/main/index.html
- **官网：** https://dexforce.com/embodichain/index.html#/EmbodiChain
- **Stars：** ~224（2026-09-18）
- **版本：** v0.2.4（`VERSION` 文件）
- **许可：** Apache 2.0
- **入库日期：** 2026-09-18
- **一句话说明：** 端到端 GPU 加速具身智能平台：DexSim 物理渲染引擎 + Gym 任务环境 + 数据管线 + IL/RL 训练 + LeRobot 数据集互操作 + 可选 Sim2Real 部署。
- **代码：** https://github.com/DexForce/EmbodiChain（**已开源**）
- **沉淀到 wiki：** 是 → [`wiki/entities/embodichain.md`](../../wiki/entities/embodichain.md)

---

## 核心定位

- **引擎：** [DexSim](https://pypi.org/project/dexsim_engine/)（PyPI 包名 `dexsim_engine`，Python 导入 `dexsim`；经 DexForce 私有 index 分发）
- **阶段：** **Alpha**（README 声明持续开发；API 可能变动）
- **栈：** 高保真 GPU 刚体/可变形物理、光线追踪传感器、批量仿真、Viser 浏览器远程可视化
- **学习：** 统一 Gymnasium 接口；内置 IL/RL 训练（`train-rl`、`eval-policy`）；依赖 `lerobot>=0.4.4,<0.5`
- **数据：** `data` CLI 下载资产；LeRobot 格式录制与 `preview_lerobot_data` 校验
- **生成式仿真（可选 `gensim`）：** SimReady 资产管线、Scene Engine（图像→场景）、Blender `bpy` 网格处理

---

## 模块结构（仓库根）

| 目录 | 职责 |
|------|------|
| `embodichain/` | 核心：CLI、`lab` 环境运行、`learning` 训练、`data_pipeline`、`gen_sim`、`toolkits` |
| `embodichain_tasks/` | 官方任务包（manipulation / classic-control 等）；随主 wheel 安装，**勿单独 editable 安装** |
| `examples/` | 学习、策略评测、数据生成示例 |
| `scripts/tutorials/` | 仿真、可视化、数据管线教程 |
| `docker/` | `dexforce/embodichain:ubuntu22.04-cuda12.8` 镜像与 `docker_run.sh` |

---

## 统一 CLI（`embodichain`）

| 命令 | 用途 |
|------|------|
| `run-env` | 从 gym JSON/YAML 启动任务（数据生成或预览）；别名 `run-task` |
| `list-task` | 按类别/能力列出已注册任务 |
| `train-rl` | RL 训练（JSON/YAML 配置） |
| `eval-policy` | 策略评测（headless 或 viewer） |
| `data` | 列出/下载 EmbodiChain 数据资产 |
| `simready` / `scene-engine` | 生成式资产与场景管线 |
| `preview-asset` / `preview_lerobot_data` | 资产与 LeRobot 数据集预览 |
| `annotate-grasp` / `decompose-urdf` | 抓取标注、URDF 凸分解 |

任务注册：`@register_env` + `embodichain.tasks` entry point；第三方包可声明 entry point 自动发现。

---

## 安装要点（文档 2026-09）

**系统：** Linux x86_64、NVIDIA GPU（CC 7.0+）、驱动 ≥535、CUDA 12.x、Python 3.10–3.12（`gensim` 需 3.11）。

**DexForce index（必需）：**

```bash
pip install embodichain \
  --extra-index-url http://pyp.open3dv.site:2345/simple/ \
  --trusted-host pyp.open3dv.site
```

**Docker（首推首次体验）：**

```bash
docker pull dexforce/embodichain:ubuntu22.04-cuda12.8
./docker/docker_run.sh <container_name> <data_path>
```

**可选 extras：** `[policy-deploy]`（ONNX Runtime GPU）、`[gensim]`（Blender bpy + pyrender）；cuRobo V2 需单独从 NVlabs 源码安装。

**验证：**

```bash
python -c "import embodichain, dexsim; print(embodichain.__version__, dexsim.__version__)"
embodichain run-env --gym_config embodichain_tasks/configs/tasks/manipulation/repeated_pick_place/task.franka.yaml
```

部分任务支持 `--physics newton`（Newton 后端）。

---

## 生态与下游

- **RoboSynChallenge：** [EDEM-AI/RoboSynChallenge](../repos/robosynchallenge.md) 以 EmbodiChain 为安装/数据/训练栈（PI0/PI0.5/Motus + ACT/DP）
- **任务模板：** [embodichain_task_template](https://github.com/DexForce/embodichain_task_template) — 自定义 `@register_env` 或 YAML Task Program
- **LeRobot：** 核心依赖；数据集预览与训练互操作
- **DexForce 硬件：** FiveAges 仿真描述等页提及 DexForce W1 轮式人形

---

## 开源核查（步骤 2.5，2026-09-18）

| 项 | 结论 |
|----|------|
| GitHub 仓 | **已开源**（Apache 2.0） |
| 项目页 / 文档 | 链到同一 GitHub；无「待发布代码」表述 |
| 可运行 | 需 DexForce PyPI index 拉 `dexsim_engine`；Docker 镜像可复现 GPU 栈 |
| 仿真后端 | `dexsim_engine` **非** PyPI 默认 index 公开包，须 `--extra-index-url` |
| 权重/数据 | `embodichain data` CLI；RoboSynChallenge 另托管 HF org |
| 成熟度 | **Alpha**；roadmap 见官方文档 |

---

## 对 wiki 的映射

- [EmbodiChain](../../wiki/entities/embodichain.md)
- [RoboSynChallenge](../../wiki/entities/paper-robosynchallenge.md) — 下游挑战赛栈
- [LeRobot](../../wiki/entities/lerobot.md) — 数据集与训练互操作
- [Isaac Lab-Arena](../../wiki/entities/isaac-lab-arena.md) — GPU 通才评测对照
- [RoboCasa](../../wiki/entities/robocasa.md) — 厨房 manipulation benchmark 对照
