# TouchAnything 项目页

> 来源归档（site / project-page）

- **标题：** TouchAnything: Multi-View Tactile Prediction
- **类型：** site / project-page
- **官方入口：** <https://jianyi2004.github.io/TouchAnything-Website/>
- **论文：** <https://arxiv.org/abs/2605.13083>
- **代码：** <https://github.com/Jianyi2004/TouchAnything>（公开；MIT）
- **数据集：** <https://huggingface.co/datasets/zhouzhoujy/EgoTouch>
- **机构：** 哈尔滨工业大学（深圳）、美团机器人研究院、清华大学深圳国际研究生院
- **核查日期：** 2026-10-24

## 页面公开信息

项目页称 EgoTouch 包含 208 项操作任务、1,891 段 episode，使用头戴相机与双腕相机采集，并同步双手 42 关节姿态和连续压力图。TouchAnything 以第一人称视角为主，结合可用的腕部视角预测触觉压力。

项目页描述传感器采集的压力图为 16×16；仓库 README 的 HDF5 输出为 21×21 归一化压力网格。二者可能对应采集端与转换后的数据表达，使用前应查代码确认映射方式。

## 开源核查（步骤 2.5）

- **代码已公开：** 官方项目页 GitHub 链接指向 Jianyi2004/TouchAnything；仓库含配置、源码和运行脚本，许可证为 MIT。
- **数据已托管：** 项目页链接至 Hugging Face zhouzhoujy/EgoTouch。截至核查日，数据集页面显示约 88.5 GB 文件。
- **版本提醒：** README 的 2026-05 News 曾称上传仍在进行。当前页面已显示多组数据文件，但复现前仍应核对各 split 是否完整。

## 对 wiki 的映射

- [wiki/entities/paper-sa-2605-13083-touchanything-a-dataset-and-framework-for-bimanu.md](../../wiki/entities/paper-sa-2605-13083-touchanything-a-dataset-and-framework-for-bimanu.md)
- [sources/papers/touchanything_arxiv_2605_13083.md](../papers/touchanything_arxiv_2605_13083.md)
- [sources/repos/touchanything.md](../repos/touchanything.md)
