# HiPHI-Hand 官方数据卡归档 | Dataset card archive

- **类型：** dataset
- **发布方：** Noitom Robotics（ModalityNet）
- **数据集：** <https://huggingface.co/datasets/noitomrobotics/HiPHI-Hand>
- **发布：** 2026-11-09；[发布方公告](https://www.globenewswire.com/news-release/2026/11/09/3398459/0/en/noitom-robotics-releases-hiphi-hand-precision-hand-capture-data-used-to-train-nvidia-egohand-and-demonstrates-real-time-hand-reconstruction-at-corl-2026.html)
- **许可：** [ModalityNet Open Research License v1.0（随库条款）](https://huggingface.co/datasets/noitomrobotics/HiPHI-Hand/blob/main/LICENSE.md)，申请访问；商用另行授权，不能镜像或转发原始数据。
- **核对日期：** 2026-11-25
- **核对范围：** 公开数据卡全文、文件树及 dataset_01 目录、随库许可全文、发布公告与 NVIDIA 模型卡。原始视频、CSV 与 `visualize.py` 正文未获访问，未实跑。
- **通读：** 已完成
- **对应节点：** [HiPHI-Hand](../../wiki/entities/dataset-hiphi-hand.md)

1. **打包。** 数据卡列出 5 组、208/156/120/206/104 段，合计 794 段。发布公告称 11 小时。片段号按组重置，使用组名加片段号定位。每组有自己的标定；同一片段的两路视频与两份 CSV 按帧号对应。
2. **格式。** 双路硬件同步鱼眼 1360×1360、30 fps、H.264。人体手部真值用同一头显 IMU 坐标系（米）表达：MediaPipe21 每帧 42 行，Vicon25 每帧 50 行并带全局取向 `xyzw`；两套 schema 的腕点可交叉核验。两路相机分别应用该组 IMU→camera 变换和等距鱼眼模型。
3. **质量与许可。** 文件树显示 `visualize.py`，数据卡有运行命令；匿名读取脚本返回 401，未把文档说明当成源码验证。许可把非商业科研与产品相关研发分开，禁止转载原始文件；符合条款的非商业成果仍需署名。
4. **与模型的关系。** 发布方称这是提供给 NVIDIA Isaac 团队的训练原始采集之一；[EgoHand-v1 模型卡](https://huggingface.co/nvidia/EgoHand-v1)描述更大的混合训练语料及单目模型，不能把其训练总量、精度或模型许可归给此数据集。全身 [HiPHI](../../wiki/entities/paper-hiphi.md) 是同系列另一数据发布。
