# Jianyi2004/TouchAnything（TouchAnything 官方实现）

> 来源归档（repo）

- **标题：** TouchAnything: A Dataset and Framework for Bimanual Tactile Estimation from Egocentric Video
- **代码：** <https://github.com/Jianyi2004/TouchAnything>
- **项目页：** <https://jianyi2004.github.io/TouchAnything-Website/>
- **配套论文：** <https://arxiv.org/abs/2605.13083>
- **数据集：** <https://huggingface.co/datasets/zhouzhoujy/EgoTouch>
- **License：** MIT
- **首次入库：** 2026-10-24

## 一句话摘要

从同步第一人称与双腕 RGB 视频估计双手连续触觉压力的研究代码，配套 EgoTouch 人手操作视触觉数据集；README 给出了数据转换、训练及推理评测脚本。

## 公开运行入口

1. 创建 Conda 环境：conda env create -f environment.yaml。
2. 下载 EgoTouch 原始数据，并运行 bash scripts/run_convert_to_hdf5.sh 转为 HDF5。
3. 运行 bash scripts/run_train_ddp.sh 进行分布式训练。
4. 运行 bash scripts/run_inference.sh 生成预测、指标和可视化。

脚本假设从仓库根目录启动。代码许可为 MIT；数据集由单独的 Hugging Face 页面托管，README 所述上传状态较早，使用前应核实数据文件与 split 完整性。

## 数据接口注意事项

README 的 HDF5 样例记录左右压力网格为 (T, 21, 21)，值归一化至 [0,1]，并注明数据为 30 FPS。项目页采集介绍写的是每掌 16×16 压力图。不要把转换后网格尺寸直接当作传感器原生分辨率；应沿数据转换脚本核对处理实现。

## 对 wiki 的映射

- [wiki/entities/paper-sa-2605-13083-touchanything-a-dataset-and-framework-for-bimanu.md](../../wiki/entities/paper-sa-2605-13083-touchanything-a-dataset-and-framework-for-bimanu.md)
- [sources/papers/touchanything_arxiv_2605_13083.md](../papers/touchanything_arxiv_2605_13083.md)
- [sources/sites/touchanything-github-io.md](../sites/touchanything-github-io.md)
