# 机器人操作数据集与基准：18 个官方入口核对（2026-11-07）

- **类型：** 官方项目、代码仓库、数据卡、技术报告的导航与核验记录
- **核对时间：** 2026-11-07
- **核对范围：** 18 个链接的公开页面与可见 README / dataset card；核对数据访问边界、部分数据规模与 benchmark 定位。未逐一通读全部论文 PDF、全部训练/评测源码及完整数据文件。
- **通读：** 未完成
- **使用边界：** 本文不是 18 个来源全文精读；论文、代码深读以对应项目独立归档为准，本文只记录本次可证实事实及待核查项。

## 1. 仿真基准与数据（5 个入口）

| 项目（中英双语） | 官方资源 | 本次能核实的事实 | 既有 wiki |
|---|---|---|---|
| LIBERO：终身机器人学习知识迁移基准 / Benchmarking Knowledge Transfer in Lifelong Robot Learning | [代码](https://github.com/Lifelong-Robot-Learning/LIBERO) · [数据卡](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets) | 官方数据卡列出 `libero_object`、`libero_spatial`、`libero_goal`、`libero_90`、`libero_10`；每任务 HDF5 演示，页面当前不提供自动数据预览 | [LIBERO](../../wiki/entities/libero-benchmark.md) |
| RoboTwin 2.0：可扩展双臂数据生成与评测 / RoboTwin 2.0 | [代码](https://github.com/RoboTwin-Platform/RoboTwin) · [预采集数据](https://huggingface.co/datasets/TianxingChen/RoboTwin2.0) | GitHub README 指明 `main` 为 2.0；官方 HF 数据卡存在，但 README 当前为空且预览出错，不能从数据卡推断字段/schema | [RoboTwin](../../wiki/entities/robotwin.md) |
| CALVIN：语言条件长程机器人操作基准 / Composing Actions from Language and Vision | [代码](https://github.com/mees/calvin) | 官方 README 描述语言驱动的多步模拟操作评测，仓库含 `calvin_env`、`calvin_models`、`dataset` | [CALVIN](../../wiki/entities/calvin-benchmark.md) |

## 2. 真实机器人与跨本体数据（12 个入口）

| 项目（中英双语） | 官方资源 | 本次能核实的事实 | 既有 wiki |
|---|---|---|---|
| Open X-Embodiment：跨机器人数据与 RT-X / Open X-Embodiment | [代码](https://github.com/google-deepmind/open_x_embodiment) · [项目](https://robotics-transformer-x.github.io/) | 官方项目页介绍跨本体数据与 RT-X；统一数据集接口不意味着所有机器人的动作/动力学都相同 | [OXE](../../wiki/entities/paper-open-x-embodiment.md) |
| DROID：野外机器人操作示范 / Distributed Robot Interaction Dataset | [项目](https://droid-dataset.github.io/) | 官方页报告 76k 条轨迹、350 小时、564 场景、86 任务；提供 Dataset Colab、Hardware Code、Policy Learning Code 链接 | [DROID](../../wiki/entities/paper-rcl-ref-aa9c28fbf4242ea03696-droid-a-large-scale-in-the-wild-robot-manipulati.md) |
| BridgeData V2：跨环境机器人操作数据 / A Dataset for Robot Learning at Scale | [项目](https://rail-berkeley.github.io/bridgedata/) · [训练代码](https://github.com/rail-berkeley/bridge_data_v2) | 官方项目页报告 60,096 轨迹、24 个环境、13 类技能，包含示范与脚本抓放 rollout | [BridgeData V2](../../wiki/entities/painode-085-bridgedatav2.md) |
| RoboSet：厨房多任务真实机器人数据 / RoboSet Dataset | [项目和数据入口](https://robopen.github.io/roboset/) | 官方页报告 28,500 轨迹（9,500 遥操作 + 19,000 动作示教回放），四路相机；项目页有 HDF5 的 `h5py` 读取样例 | [RoboSet](../../wiki/entities/roboset-dataset.md) |
| RH20T：多模态接触丰富操作数据 / A Comprehensive Robotic Dataset for Learning Diverse Skills in One-Shot | [项目及下载](https://rh20t.github.io/) · [API](https://github.com/rh20t/rh20t_api) | 官方页报告超过 110,000 条真实操作序列；RGB(D)、关节、力/力矩、音频以及部分配置的触觉；公开下载说明与解析 API | [RH20T](../../wiki/entities/painode-092-rh20t.md) |
| AgiBot World Colosseo：大规模操作平台 / AgiBot World Colosseo | [代码](https://github.com/OpenDriveLab/AgiBot-World) · [Alpha](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Alpha) · [Beta](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) · [技术报告](https://arxiv.org/abs/2503.06669) | Alpha 数据卡记录更正后保留 36 任务、595.31 小时；Beta 声明 100 万+ 轨迹、2976.4 小时。**两张数据卡均要求登录同意共享信息及协议**；Beta 页面下载示例仍指 Alpha，实际下载请复核目标路径。技术报告 arXiv 于 2026-10-20 更新至 v5 | [Colosseo](../../wiki/entities/paper-sa-2503-06669-agibot-world-colosseo-a-large-scale-manipulation.md) |

## 3. 机器人学习工具链（1 个入口）

- **LeRobot：开放机器人学习工具链 / LeRobot** — [官方源码](https://github.com/huggingface/lerobot)。README 将模型、数据、工具组合在 PyTorch 生态中；它**不是与 LIBERO、DROID 同类的单一原始数据集**。对应 [LeRobot 实体](../../wiki/entities/lerobot.md)，LeRobot 的具体内置模型须按仓库 README 与独立模型节点维护。

## 工程选型读法

- **评测环境**：LIBERO 看任务知识迁移；CALVIN 看语言条件长程组合；RoboTwin 2.0 看双臂仿真、专家数据与评测协议。
- **离线真实数据**：DROID、BridgeData V2、RoboSet、RH20T、AgiBot World 在本体、场景、传感器与许可上各不相同；不能直接拼接动作字段后视作同分布。
- **聚合与工具**：OXE 是跨数据集的汇集研究，LeRobot 是数据/策略/机器人训练工具链；二者与单一采集型数据集不是相同层级。
- **获取限制**：HF 有数据页不等于能在浏览器预览或免登录下载；尤其 AgiBot Alpha/Beta 需明确遵守门控协议。

## 待完成核验

1. 全文精读各项目主论文（含 RoboTwin 2.0、DROID、BridgeData V2、AgiBot World Colosseo）及附录。
2. 逐一阅读训练/评测/解析代码入口，验证时序与版本；不能根据 README 的文件名推断调用栈。
3. 对照数据集实际 schema、物理动作空间及许可条款；RoboTwin 2.0 HF 当前数据预览错误、AgiBot Alpha/Beta 数据门控，不能假装完成样本级校验。
