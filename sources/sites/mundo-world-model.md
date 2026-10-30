# Mundo: Customized World Model

> 来源归档（site / project page）

- **项目主页：** <https://mundo-world-model.github.io/>
- **技术博客：** <https://huggingface.co/blog/Luo-Yihang/mundo-world-model>（2026-10-23）
- **官方仓库：** <https://github.com/Luo-Yihang/Mundo>
- **核查日期：** 2026-10-30
- **核查范围：** 项目主页及技术博客全文；官方仓库 README 与文件列表。
- **开放状态：** 仓库目前只有 README 和 assets；README 将代码和权重列为 11 月底待发布。未发现可运行实现或公开训练数据集。

## 内容与映射

项目将输入场景图片重建为显式 3D 代理，通过场景程序与仿真更新物理状态，再由视频扩散模型根据代理视频和上下文视图生成图像。详见 [Mundo 实体](../../wiki/entities/mundo-world-model.md)。
