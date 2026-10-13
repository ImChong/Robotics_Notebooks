# facebookresearch/GenIA

- **类型：** repo
- **标题：** GenIA: Generative Reconstruction with Test-Time Input Alignment
- **链接：** https://github.com/facebookresearch/GenIA
- **项目页：** https://facebookresearch.github.io/GenIA/
- **论文：** https://arxiv.org/abs/2610.12388
- **机构：** Meta Reality Labs；Tübingen AI Center
- **许可证：** CC BY-NC 4.0；第三方组件有各自许可，部分亦为非商业用途
- **核查日期：** 2026-10-13
- **开源状态：** **已开源**；公开仓库含 demo/run_demo.py、genia.core.download_weights 和配置文档。
- **运行要求：** Conda、NVIDIA GPU、CUDA Toolkit 12.8；安装脚本初始化 SAM 3D Objects、MapAnything、ActionMesh 等子模块。需先在 Hugging Face 接受 SAM 3D Objects 许可后下载权重。
- **一句话说明：** 官方 GenIA 实现将冻结的 SAM 3D 生成先验与按场景几何/图像观测优化结合，支持 image、multiview、dynamic 三个 demo 模式。

## 最小官方运行入口

```bash
git clone https://github.com/facebookresearch/GenIA
cd GenIA
bash install.sh
conda activate genia
python -m genia.core.download_weights
python demo/run_demo.py image
python demo/run_demo.py multiview
python demo/run_demo.py dynamic
```

安装可能需要编译 CUDA 扩展；不同输入类型需要按仓库 demo/data 约定准备场景目录。这里只归档 README 入口，未在本库环境实测。

## 对 wiki 的映射

- [GenIA 论文与代码实体页](../../wiki/entities/paper-genia-arxiv-2610-12388.md)

## 交叉归档

- 论文摘录：[genia_arxiv_2610_12388.md](../papers/genia_arxiv_2610_12388.md)
- 项目页核查：[genia-facebookresearch.md](../sites/genia-facebookresearch.md)

## 参考来源

- [官方 GitHub 仓库](https://github.com/facebookresearch/GenIA)
- [仓库 README](https://github.com/facebookresearch/GenIA#readme)
- [项目页](https://facebookresearch.github.io/GenIA/)
