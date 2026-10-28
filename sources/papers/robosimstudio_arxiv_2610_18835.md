# RoboSimStudio: An Interactive Toolkit for Rapid Real2Sim of Robot Manipulation Tasks

- **类型：** 技术报告 / arXiv 预印本
- **作者：** Kaiyuan Chen、Shuangyu Xie、Uksang Yoo、Xinyue Zhu、Cael Magner、Alvin Zhu、Daehwa Kim、Eric Heiden、Erwin Coumans、Miles Macklin、Spencer Huang、Ken Goldberg
- **机构：** 加州大学伯克利分校 AUTOLab、NVIDIA
- **年份：** 2026
- **arXiv：** <https://arxiv.org/abs/2610.18835>
- **项目页：** <https://robosimstudio.github.io/>
- **代码：** <https://github.com/robosimstudio/robosimstudio>
- **入库日期：** 2026-10-15；补充核查：2026-10-16
- **一句话摘要：** 提出一个开放交互式 GPU 仿真工具包，让人和编码 Agent 显式选择并迭代调整求解器、接触模型及物理参数，以快速重建真实机器人操作任务。
- **对 wiki 的映射：** [wiki/entities/robosimstudio.md](../../wiki/entities/robosimstudio.md)

## 论文线索

论文针对机器人与操作物体间的接触、耦合动力学、形变、刚度和摩擦难以准确建模的问题，提出让仿真器选择和物理参数可见、可编辑，并让人和编码 Agent 基于观察到的仿真行为共同迭代。

工具建立在 Newton 物理仿真系统上。代表性双臂海绵擦拭场景中，Newton 内的 MuJoCo Warp 负责机械臂和刚体，SuperDex 有限元方法（FEM）模拟海绵形变。论文还报告 18 项接触丰富操作任务仿真，涵盖刚体、关节机构、线缆、布料与软体对象。

Agent 仅凭手机视频构建擦拭任务仿真，论文报告其中一个案例约 56 分钟完成。Graph-as-Policy 利用空间—时间仿真反馈，将模拟擦拭面积指标从 0% 提高到 100%（14 次迭代）；改进策略随后部署到真实机器人并完成擦拭。真实运行中支撑杆发生移动，而原仿真将其固定；团队据此修改约束，使仿真能表现支撑杆移动和倾倒。该证据展示单项任务上的 Real2Sim2Real 迭代，不等于跨任务的精度基准或普遍成功率保证。

## 论文局限

作者将真机迁移描述为 pilot study，评估集中在单一操作任务，仿真与真机的一致性主要通过定性观察判断。报告指出系统能力受已支持的物理求解器、现有材料模型、求解器耦合和机器人资产限制；与外部物理引擎协同仿真还会带来计算开销。跨任务仿真精度、接触力、物体轨迹、建模时间和真机成功率等系统定量评估留待后续工作。

## 补充精读（2026-10-28）

- §3：同一场景显式指定对象、求解器和材料参数；Newton 内求解器可交换接触力，外部引擎可交换位姿及接触力和力矩。交互界面向人展示实时场景，向编码智能体提供视觉与时空状态反馈。
- §4.1：18 个任务以视觉对照真实运动与形变，定性调整物理参数；例如 USB-C 插接涉及亚毫米间隙，叶片涉及弯曲恢复，抛光以接触力和相对运动建模材料去除。
- §4.2：Gello 主臂驱动仿真机器人，并依据指令与仿真关节构型差提供反馈；速度相关死区和暂停或落后时释放反馈用于降低延迟造成的误感。
- §4.3：两种编码智能体分别根据手机视频构建场景；一个案例 56 分钟，生成 30 个空间布局并改变海绵刚度。GaP 的 0%→100% 是论文定义的模拟擦拭面积，14 次为策略改进迭代；真机失败后模型修改呈现杆移动和倾倒，文中未报告修订后跨任务真机成功率。

## 来源链接

- arXiv 摘要与 PDF：<https://arxiv.org/abs/2610.18835>
- 项目页：<https://robosimstudio.github.io/>
- 官方仓库：<https://github.com/robosimstudio/robosimstudio>
