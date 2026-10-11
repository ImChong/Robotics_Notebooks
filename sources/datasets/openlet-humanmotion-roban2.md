# OpenLET LET-HumanMotion-Roban2（真机小脑动作数据）

> 来源归档（官方文档与数据申请入口）

- **类型：** 机器人动作数据集
- **官方说明：** <https://www.lejubfm.lejurobot.com/docs/datasets/openlet>
- **申请入口：** <https://openlet.openatom.tech/data/let-humanmotion-roban2>
- **技术仓库（需已有访问权限）：** <https://atomgit.com/OpenLET/LET-HumanMotion-Roban2.git>
- **配套运控代码：** <https://atomgit.com/OpenLET/Lejulab_ARM>
- **制作方：** 乐聚机器人、青瞳视觉
- **一句话说明：** 面向鲁班人形运控训练与评测的全身动作轨迹集；官方文档称五类动作已重定向到鲁班机型并按 120 fps 重采样，获取需申请。

## 官方公开信息

- 五类：非交互动作、环境交互、物体交互、武术、舞蹈；演示包括后空翻、360°旋转跳、蝴蝶旋转与现代舞。
- 轨迹统一重定向至鲁班机型，并以 120 fps 重采样；官方定位为运控能力训练与评测。
- 数据采取申请制。申请要求、获取方式和使用条款以 OpenLET 申请页与审批结果为准。
- AtomGit 技术仓库与数据申请是两个入口；能访问代码仓不代表已获数据集权限。
- 提供的“近四万条”规模描述未在当前公开文档页找到可核验计数，故不作为已确认的精确规模记录。

## 可用方向与边界

1. **动作模仿 / 运控策略训练：** 将机器人坐标系中的全身动作轨迹作为参考运动或监督信号，训练鲁棒的动作跟踪能力。
2. **运动技能覆盖评测：** 按动作族构造训练/留出切分，评估非交互、环境交互、物体交互及武术舞蹈动作的跟踪成功率与稳定性。
3. **全身 VLA 数据研究：** 可研究动作先验与视觉任务策略结合；但本数据集文档没有承诺自带语言标注或相机观测，不能直接假定它是端到端 VLA 数据。
4. **策略推理部署：** 配套 LejuBFM 提供鲁班 2.2 上的运控模型部署，LeTools 教程给出全身 VLA 从 rosbag 数据转换到训练和 HEFT 真机推理的路径。教程所述输入流程不等同于该数据包的官方转换配方，需取得数据后核验字段与版本。

## 对 wiki 的映射

- [OpenLET LET-HumanMotion-Roban2 实体](../../wiki/entities/openlet-humanmotion-roban2.md)
- [OpenLET 社区](../../wiki/entities/openlet.md)
- [LeTools 工具链](../../wiki/entities/letools.md)
