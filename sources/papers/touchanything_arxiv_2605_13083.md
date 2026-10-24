# TouchAnything 论文来源归档

## 元数据

- **题目：** TouchAnything: A Dataset and Framework for Bimanual Tactile Estimation from Egocentric Video
- **作者：** Jianyi Zhou, Ziteng Gao, Feiyang Hong, Zirui Liu, Guannan Zhang, Weisheng Dai, Ruichen Zhen, Chuqiao Lyu, Haotian Wu, Yinian Mao, Xushi Wang, Yuxiang Jiang, Wenbo Ding, Shuo Yang
- **单位：** 哈尔滨工业大学（深圳）、美团机器人研究院、清华大学深圳国际研究生院
- **预印本：** https://arxiv.org/abs/2605.13083v1
- **项目页：** https://jianyi2004.github.io/TouchAnything-Website/
- **代码：** https://github.com/Jianyi2004/TouchAnything （MIT；见 [仓库归档](../repos/touchanything.md)）
- **数据集：** https://huggingface.co/datasets/zhouzhoujy/EgoTouch
- **入库日期：** 2026-10-24

## 论文要解决的问题

第一人称人类操作视频易于规模化采集，但通常缺少真实触觉监督。论文研究能否仅从视频视角估计手物接触压力，以获得更易扩展的触觉学习数据。

## 方法与数据

- **EgoTouch 数据集：** 208 项操作任务、1,891 段 episode，覆盖室内外场景；同步头戴第一人称相机、左右腕相机、双手 3D 姿态（42 个关节）和可穿戴触觉手套的连续压力图。
- **TouchAnything 模型：** 以第一人称视角为主输入，按需使用腕部视角。项目页描述的模型包含共享 DINOv2 视觉编码器、可学习视角嵌入、跨视角 Transformer 注意力和 view dropout，可在推理时灵活缺失视角。
- **数据格式：** 仓库 README 将 episode 存为 30 FPS 的 HDF5；包含多视角图像、手姿、压力、相机位姿、掩码、元数据及时间戳。README 中 pressure grid 为 21×21，项目页采集说明为 16×16，复现前应对照对应数据版本与预处理实现。

## 论文报告结果

项目页报告：与仅使用第一人称视角相比，加入腕部视角通常提高触觉预测；最高相对提升为 Contact IoU 5.0%、Volumetric IoU 6.1%。这里是相对提升，不是百分点；具体任务和视角配置应以论文评测表为准。

## 开源与复现入口

- 代码仓库公开，README 提供从原始数据转换、分布式训练到推理评测的入口：scripts/run_convert_to_hdf5.sh、scripts/run_train_ddp.sh、scripts/run_inference.sh。
- 官方数据集托管在 Hugging Face。2026-10-24 页面文件列表显示约 88.5 GB；仓库 README 中 2026-05 的更新仍称上传进行中，因此下载时需确认所需分片和 split 文件齐全。
- 复现通常需要较大的本地存储和视频预处理能力；公开实现是人类手部视触觉预测基线，不能直接等同于机器人触觉力控策略。

## 对 wiki 的映射

- 详情节点：[wiki/entities/paper-touchanything-egotouch.md](../../wiki/entities/paper-touchanything-egotouch.md)
- 概念关联：[wiki/concepts/visuo-tactile-fusion.md](../../wiki/concepts/visuo-tactile-fusion.md)、[wiki/concepts/tactile-sensing.md](../../wiki/concepts/tactile-sensing.md)

## 一手来源

- 论文：https://arxiv.org/abs/2605.13083v1
- 项目页：https://jianyi2004.github.io/TouchAnything-Website/
- 代码：https://github.com/Jianyi2004/TouchAnything
- 数据集：https://huggingface.co/datasets/zhouzhoujy/EgoTouch
