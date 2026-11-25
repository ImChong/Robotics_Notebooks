# ruoyiqiao/mjlab_hand — Dexterous Hand Tasks in mjlab

> 官方仓库来源归档（repo；不是论文实验结果转载）

- **标题：** mjlab_hand: Dexterous Hand Tasks in mjlab（mjlab 灵巧操作任务库）
- **作者 / 维护者：** Ruoyi Qiao（仓库 README 的 BibTeX 引用）
- **类型：** repo / dexterous manipulation / reinforcement learning
- **原始仓库：** <https://github.com/ruoyiqiao/mjlab_hand>
- **核对版本：** `main` @ `9161811fa5486b50011b993f85cbf3a30fbfd1c1`（2026-11-22 读取）
- **许可：** 原创代码 Apache-2.0；机器人模型按各自 BSD / MIT / Apache 协议，ContactDB 资产遵循其单独条款（参见仓库 README / NOTICE）
- **环境：** Python 3.13、mjlab 1.6.0、rsl-rl-lib 5.4.2；`uv sync --locked`
- **代码：** <https://github.com/ruoyiqiao/mjlab_hand>
- **相关论文：** ContactExplorer（<https://arxiv.org/abs/2603.10971>）；mjlab（<https://arxiv.org/abs/2601.22074>）
- **沉淀到 wiki：** [mjlab_hand 详情](../../wiki/entities/mjlab-hand.md)
- **核对范围：** 完整阅读官方 `README.md` 的 Tasks、Installation、Quick Start、Custom Objects、Evaluation & Playback、Acknowledgements、Citation、License；检查 `pyproject.toml`、`src/mjlab_hand/__init__.py`、`registration.py`、`robot/hand_spec.py`、`grasp/grasp_env_cfg.py`、`in_hand_rotation/rotation_env_cfg.py`、`cli/train_object.py`、`cli/play_object.py`、`cli/eval.py`、`eval/__init__.py`、`eval/base.py`，核对 CLI 注册、奖励、观测、评测执行流。未逐行通读仓库全部子模块。
- **通读：** 已完成

## 项目定位

基于 [mjlab](https://github.com/mujocolab/mjlab) 的多机器人手强化学习任务集合，目标是用统一的注册任务、训练命令与指标接口覆盖多种灵巧操作，而不只是单一抓取场景。提供六种手型：Allegro、LEAP、Shadow、Sharpa、Wuji、XHand。

## README 明确列出的任务族

| 任务 | README 示例 ID | 说明 |
|---|---|---|
| Grasp | `Grasp-Shadow` | 抓取、抬升物体 |
| In-hand rotation | `InHand-Rotation-Allegro` | 手内旋转并跟踪目标朝向 |
| Grasp and throw | `Grasp-Throw-XHand` | 抓取后将方块投向目标 |
| Object reorientation | `Object-Reorientation-Wuji` | 将长方体转至随机目标位姿 |
| Regrasp | `Regrasp-LEAP` | 松手后重新抓取 |
| Two-hand handover | `DualHand-Reorientation-Allegro-TrackSlew` | 双 Allegro 交接物体 |
| Plate spinning | `PlateSpin-Shadow` | 指尖转盘 |
| Baoding balls | `Baoding-Sharpa-TiltThumb` | 两颗保定球掌内旋转 |
| Juggle 1 / 2 | `Juggle1-OneHand-Allegro`、`Juggle2-OneHand-Allegro-Entry6` | 单手一球 / 两球 |
| Juggle 3 | `Juggle3-LEAP-Cascade-Residual`、`Juggle3-Sharpa-Tennis`、`Juggle3-Shadow-HalfShower`、`Juggle3-Wuji-Box`、`Juggle3-Allegro-Shower` | 三球多种轨迹模式 |

任务 ID 仅列官方 README 中的实例；实际可注册 ID 以本地 `uv run list-envs` 输出为准，不能推导所有手型都实现全部任务。

## 运行入口与源码证据

- `pyproject.toml` 使用 `[project.entry-points."mjlab.tasks"]` 指向 `mjlab_hand`；`src/mjlab_hand/__init__.py` 调用 `register_tasks()`，后者在 `registration.py` 中导入各任务模块产生注册副作用。
- 抓取任务 `grasp/grasp_env_cfg.py` 将手/物体状态、关节状态、目标位姿、上次动作组成观测；奖励包含接近、目标进度、平滑度与 `contact_explorer_reward`，并配置摩擦随机化。
- 手内旋转任务 `in_hand_rotation/rotation_env_cfg.py` 使用 `JointPositionEMAActionCfg`、朝向对齐/成功奖励与可选 ContactExplorer，含失手掉落终止。
- `cli/train_object.py` 和 `cli/play_object.py` 解析 `--object` / `--scale`，修改注册表中抓取/旋转环境配置后分别移交上游 `mjlab.scripts.train.main` / `play.main`。
- `cli/eval.py` 解析检查点、创建评测环境、根据任务选择 evaluator；`eval/base.py` 的 `run_eval` 在仿真步前 snapshot，随后 policy → env.step → on_step / on_reset → finalize，避免终止自动重置污染指标。
- ContactExplorer 的 README 说明了 **接触前面向未探索物体表面区域的引导**、**接触后奖励覆盖新接触区域**。奖励具体适用范围以各任务配置是否启用为准。

## 官方可复现命令

```bash
git clone https://github.com/ruoyiqiao/mjlab_hand.git
cd mjlab_hand
uv sync --locked
uv run list-envs
uv run train Grasp-Allegro --env.scene.num-envs 2048
uv run train Juggle3-LEAP-Cascade-Residual --env.scene.num-envs 2048
uv run train-object Grasp-Allegro --object hammer --env.scene.num-envs 2048
uv run play-object Grasp-Shadow --checkpoint-file /path/to/model.pt
uv run eval-policy --task Grasp-Allegro --checkpoint /path/to/model.pt --output eval_metrics.json
```

README 为三球杂耍、转盘和宝鼎球提供 `tools/allegro4/` 下的专门评测入口；`eval-policy` 不能替代这些任务专用脚本。评测包括成功率、旋转成功次数、连续接球数、转数、掉落和落点误差等，**因任务族不同而口径不同**。

## 开放状态与边界

- **已开源：** 任务配置、资产对接、训练包装器、策略回放及评测脚本。
- **不应扩写为：** 已验证所有任务/所有自定义物体都稳定收敛；README 明确提示 ContactDB 对象未全部测试，部分需要奖励和超参调节。
- **未由本次核实：** 统一预训练权重下载、真机部署脚本、所有任务的对照实验成功率、不同灵巧手在相同协议下的定量排行榜。本次没有运行训练或评测。
- **依赖与资产：** Python 严格 3.13，特定 mjlab / rsl-rl 版本；资产并非统一 Apache-2.0。
- **对应论文关系：** ContactExplorer 与 mjlab 为项目引用研究；`mjlab_hand` 是独立任务代码仓库，不等同这两篇论文的唯一官方实现身份。

## 对知识库的映射

- [mjlab_hand 实体](../../wiki/entities/mjlab-hand.md)：任务族、输入输出、ContactExplorer、运行路径与复现风险。
- [mjlab 框架](../../wiki/entities/mjlab.md)：manager-based 仿真训练底座。
- [Contact-Rich Manipulation 路线](../../roadmap/depth-contact-manipulation.md)：作为仿真强化学习与多手型灵巧操作实作资源。

## 一手链接

- [README.md（已核对版本）](https://github.com/ruoyiqiao/mjlab_hand/blob/9161811fa5486b50011b993f85cbf3a30fbfd1c1/README.md)
- [任务注册](https://github.com/ruoyiqiao/mjlab_hand/blob/9161811fa5486b50011b993f85cbf3a30fbfd1c1/src/mjlab_hand/registration.py)
- [抓取环境配置](https://github.com/ruoyiqiao/mjlab_hand/blob/9161811fa5486b50011b993f85cbf3a30fbfd1c1/src/mjlab_hand/grasp/grasp_env_cfg.py)
- [手内旋转配置](https://github.com/ruoyiqiao/mjlab_hand/blob/9161811fa5486b50011b993f85cbf3a30fbfd1c1/src/mjlab_hand/in_hand_rotation/rotation_env_cfg.py)
- [训练包装入口](https://github.com/ruoyiqiao/mjlab_hand/blob/9161811fa5486b50011b993f85cbf3a30fbfd1c1/src/mjlab_hand/cli/train_object.py)
- [评测循环](https://github.com/ruoyiqiao/mjlab_hand/blob/9161811fa5486b50011b993f85cbf3a30fbfd1c1/src/mjlab_hand/eval/base.py)
