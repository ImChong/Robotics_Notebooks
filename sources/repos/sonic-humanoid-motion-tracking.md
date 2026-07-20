# SONIC（规模化人体运动跟踪驱动的人形全身控制）

- **标题**: SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control
- **论文**: https://arxiv.org/abs/2511.07820
- **项目页（主站）**: https://nvlabs.github.io/GEAR-SONIC/ （与 `https://nvlabs.github.io/SONIC/` 为同一套公开材料，后者为别名）
- **代码**: https://github.com/NVlabs/GR00T-WholeBodyControl （GEAR-SONIC 训练 / 评测 / C++ 部署均在此单仓）
- **权重 / 模型卡**: https://huggingface.co/nvidia/GEAR-SONIC
- **动捕数据**: https://huggingface.co/datasets/bones-studio/seed （BONES-SEED，G1 重定向轨迹）
- **文档站**: https://nvlabs.github.io/GR00T-WholeBodyControl/
- **类型**: paper / foundation-controller
- **机构**: NVIDIA、CMU 等（论文与官网作者列表为准）
- **收录日期**: 2026-05-07
- **最近对照官网整理**: 2026-07-20（对照 GEAR-SONIC 项目页 + GR00T-WholeBodyControl README / 训练与 Quick Start 文档；确认官方源码与权重已发布）

## 一句话摘要

将 **运动跟踪（motion tracking）** 作为可规模化监督信号，用海量高质量动捕帧训练通用人形策略，把「跟踪参考运动」学到表征里，从而支持 VR、视频、文本、音乐等多接口输入，并作为下游任务的通用低层执行器。

## 为何值得保留

- **范式**: 用密集轨迹监督替代大量手工任务奖励工程，与 BeyondMimic / DeepMimic 族思路相承但在 **数据量、模型规模与接口多样性** 上强调 scaling。
- **系统集成**: 被视频驱动人形流水线用作「物理过滤器」——把估计的人体运动映射到真实机器人可行域（见 ExoActor 等）。
- **已开源**: 论文与文档站写明训练/部署代码在 [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)；权重在 HF `nvidia/GEAR-SONIC`；大规模动捕子集以 BONES-SEED 发布。仓内 `gear_sonic/`（Isaac Lab 训练）与 `gear_sonic_deploy/`（C++ / TensorRT 真机推理）分工清晰。

## 技术要点（论文 + 官网公开描述对齐）

### Scaling 与数据

1. **三轴扩展**：同时增大 **网络容量**（公开区间约 **1.2M→42M** 参数量级叙述）、**数据集规模**（约 **1 亿+ 帧**、约 **700 小时** MoCap）与 **算力**（约 **2.1 万 GPU 小时** 量级叙述）；报道中性能随规模与数据多样性稳步改善。
2. **单一统一策略**：官网强调主要结果由 **同一套统一控制策略** 产生（具体训练与部署细节以论文为准）。

### 统一表示与下游

3. **通用 token / 控制接口**：多种上游（机器人运动、人体运动、混合指令）经 **专用编码器** 汇入 **共享潜空间 / token 空间**，再驱动同一 tracking policy；便于替换上游而不重写整套奖励。
4. **实时运动学规划桥**：官网展示 **实时运动学规划器** 与 tracking 衔接，用于 **导航式交互**（如手柄驱动多样步态、蹲跪爬、拳击等风格化运动），把「跟踪参考」接到任务层交互。
5. **与 VLA 堆叠**：公开演示将 **GR00T N1.5** 类 VLA 基础模型经 **同一通用控制接口** 接入，强调 **高层推理 + 低层快速全身反应** 的组合；策略在展示中为全自主（以论文与视频说明为准）。

### 遥操作与多模态条件

6. **视频遥操作**：以视频为输入，配合 [GEM](https://research.nvidia.com/labs/dair/genmo/) 做姿态估计，使人形实时跟踪复杂人体动作。
7. **VR 遥操作**：含 **头+双手三追踪点** 驱动上身、**运动学规划器生成下身** 的混合模式，以及 **全身 VR 追踪** 模式。
8. **音乐 / 文本条件**：在统一接口下，由 GEM 生成编舞或语言→人体运动，再由策略跟踪（属条件运动合成 + tracking 的部署形态）。

### 鲁棒性叙事

9. **扰动下跟踪**：官网单独展示在外部扰动下仍维持全身稳定的跟踪片段（具体实验协议见论文）。

### 与 BeyondMimic 关系

10. **生态位**：同属 Isaac / 人形模仿生态中的高性能跟踪路线；SONIC 侧重「规模化跟踪即基础能力」与多接口产品化叙事。

## 工程入口（以仓库文档为准）

| 阶段 | 入口 |
|------|------|
| 拉权重 | `python download_from_hf.py`（部署 ONNX）或 `--training`（PyTorch + SMPL） |
| 数据 | Bones-SEED G1 CSV → `gear_sonic/data_process/convert_soma_csv_to_motion_lib.py` → `filter_and_copy_bones_data.py` |
| 训练 | `pip install -e "gear_sonic/[training]"` → `gear_sonic/train_agent_trl.py +exp=manager/universal_token/all_modes/sonic_release` |
| 仿真评测 | `gear_sonic/eval_agent_trl.py +checkpoint=sonic_release/last.pt` |
| Sim2Sim | Terminal1 `gear_sonic/scripts/run_sim_loop.py`；Terminal2 `gear_sonic_deploy/deploy.sh sim` |
| 真机 | `gear_sonic_deploy/deploy.sh real` |

完整归档见 [gr00t_wholebodycontrol.md](./gr00t_wholebodycontrol.md)。

## 对 Wiki 的映射

- **wiki/methods/sonic-motion-tracking.md**：人形通用动作跟踪基础模型方法页（含流程图、源码运行时序图与接口总览）。
- **wiki/entities/gr00t-wholebodycontrol.md**：官方代码仓实体页。
- **wiki/methods/beyondmimic.md**：历史与技术脉络对齐。
- **wiki/methods/exoactor.md**：作为「SMPL 估计轨迹 → 机器人执行」的执行层实例。
- **wiki/methods/vla.md**、**wiki/concepts/foundation-policy.md**：VLA + 低层 tracking 分层的公开案例引用。
- **wiki/tasks/teleoperation.md**：VR / 视频遥操作部署参考。
