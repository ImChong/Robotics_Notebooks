# NJU-R-L-Group-Embodied-Lab/uni-lavira-code

> 来源归档

- **标题：** Uni-LaViRA（官方实现）
- **类型：** repo
- **组织 / 作者：** NJU-R-L-Group-Embodied-Lab（南京大学具身实验室等）
- **代码：** <https://github.com/NJU-R-L-Group-Embodied-Lab/uni-lavira-code>
- **论文：** <https://arxiv.org/abs/2605.27582>
- **项目页：** <https://xetroubadour.github.io/Uni-LaViRA/>
- **License：** CC BY-NC-SA 4.0
- **入库日期：** 2026-07-22
- **一句话说明：** Uni-LaViRA **training-free** 统一具身导航官方仓：`sim-code/`（Habitat + AirSim 评测）与 `real-world-code/`（四本体部署）自包含；LA/VA 经 API 调用，无机器人轨迹微调。

## 入口速查（对齐 README）

| 路径 / 命令 | 作用 |
|-------------|------|
| `sim-code/habitat/README.md` | 室内 VLN-CE / ObjectNav / EQA 环境与数据 |
| `sim-code/habitat/eval_scripts/vlnce_r2r.sh` 等 | 默认 100-episode 分层子集评测（`NPROC` 并行） |
| `sim-code/habitat/run_mp.py` · `vlnce_baselines/ZS_Evaluator_mp.py` | Habitat 多进程评测入口 |
| `sim-code/airsim/scripts/unilavira_eval.sh` · `unilavira_evaluator.py` | OpenUAV / Aerial-VLN 评测 |
| `real-world-code/cobot_magic/main.py` | Agilex Cobot Magic 真机 |
| `real-world-code/unitree_g1/main.py` · `scripts/run_navigation.sh` | Unitree G1 |
| `real-world-code/unitree_go1/main.py` | Unitree Go1 |
| `real-world-code/self_built_uav/` | 自研 UAV（ROS launch / indoor_eval） |
| `.env`（`LA_*` / `VA_*`） | Language / Vision Action 模型 API |

**依赖提示：** Habitat-Sim/Lab v0.1.7、GroundingDINO + SAM（语义图）、NavDP 局部规划、Matterport/HM3D 场景许可；LA 推荐 Gemini、VA 推荐 Qwen3.5-27B。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Uni-LaViRA](../../wiki/entities/paper-uni-lavira.md) | 实体归纳：三层翻译、TDM/SCB、零训练跨任务 |
| [视觉–语言导航](../../wiki/tasks/vision-language-navigation.md) | VLN-CE / Aerial-VLN 任务坐标 |
| [VLN 四范式复现](../../wiki/overview/vln-open-source-repro-paradigms.md) | 相对 Uni-NaVid 等「训练式导航 VLA」的 **零样本 agentic** 对照栈 |
| [VLA](../../wiki/methods/vla.md) | 导航是否必须走大规模轨迹 VLA 的反例立场 |
| [WorldVLN](../../wiki/entities/paper-worldvln-aerial-vln-wam.md) | 同属空中 VLN，但 WorldVLN 为训练式 WAM |

## 对 wiki 的映射

- 论文摘录：[`sources/papers/uni_lavira_arxiv_2605_27582.md`](../papers/uni_lavira_arxiv_2605_27582.md)
- 项目页：[`sources/sites/xetroubadour-uni-lavira-github-io.md`](../sites/xetroubadour-uni-lavira-github-io.md)
- 沉淀 **[`wiki/entities/paper-uni-lavira.md`](../../wiki/entities/paper-uni-lavira.md)**
