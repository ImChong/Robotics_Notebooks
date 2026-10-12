# Evo-World（MINT-SJTU/Evo-World）

> 来源归档

- **标题：** Evo-World 官方实现
- **类型：** repo
- **来源：** 上海交通大学 MINT 实验室（MINT-SJTU）
- **代码：** https://github.com/MINT-SJTU/Evo-World
- **论文：** https://arxiv.org/abs/2610.12936
- **许可：** 仓库根目录未见 LICENSE 文件（2026-10-12 克隆核查）
- **核查：** 2026-10-12 `git clone --depth 1`，HEAD `0b3e0b7`（2026-10-06 “Update project files”）
- **沉淀到 wiki：** [paper-evo-world-optical-flow-vla.md](../../wiki/entities/paper-evo-world-optical-flow-vla.md)
- **论文来源：** [evo_world_arxiv_2610_12936.md](../papers/evo_world_arxiv_2610_12936.md)

## 结论：已开源（代码 + 两个基准检查点）

## 目录与入口

| 路径 | 作用 |
|------|------|
| `model/internvl3/`、`model/action_head/flow_matching.py`、`model/world_model/evo_world.py` | InternVL3 嵌入、flow-matching 动作头、FiLMGenerator / SIGReg / DinoTextWorldModel |
| `scripts/precompute_flow.py` | 离线 RAFT 光流预计算（`flow_gap`，按数据集标定幅值） |
| `scripts/train.py`、`launch/train/flow_stage1_*.sh`、`flow_stage2_*.sh` | 两阶段训练（accelerate + DeepSpeed） |
| `scripts/EvoWorld_server.py`、`run_metaworld_server.sh`、`run_robotwin_server.sh` | websocket 推理服务，返回反归一化动作块 |
| `eval/metaworld/`、`eval/robotwin/EvoWorld/` | Meta-World 客户端；RoboTwin 策略插件 |
| `dataset/` | LeRobot 格式配置、`compute_normstats.py` |

README 声明预训练检查点在 HuggingFace：`MINT-SJTU/Evo-World-Metaworld`、`MINT-SJTU/Evo-World-Robotwin`（本次未下载权重）。仓库未见 AgileX 真机数据或真机部署脚本。
