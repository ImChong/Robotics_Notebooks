# FAH 官方项目主页：像素、交互与世界坐标的手部估计

> 项目页一手来源归档（site）。

- **项目：** FAH — Fast and Accurate Hand Estimation
- **类型：** site
- **链接：** <https://www.jinkuncao.com/FAH/>
- **项目备用网址：** <https://noahcao.github.io/FAH/>（官方 README 链接）
- **论文：** <https://arxiv.org/abs/2611.07711>
- **代码：** <https://github.com/noahcao/FAH>
- **权重：** <https://huggingface.co/noahcao/FAH>
- **核对日期：** 2026-11-10
- **核对范围：** 项目主页的一手论文与 Code 入口，并同官方 README、模型表和 LICENSE / WEIGHTS_LICENSE 核对发布范围。
- **通读：** 已完成

## 开放程度与资源

- **已开源（需区分许可证）**：作者仓库公开推理、训练/评测工具与模型下载方法；权重公开但限研究/非商业使用，适用第三方许可另行约束。
- **结果口径：** HOT3D 上 FAH-L 4.31 mm PA-MPJPE、65 FPS 是包含网格拟合的 TensorRT 配置；分割/接触可选模块会降速。
- **前置资源：** SAM 3D Body 相关配置需接受许可证，MANO 模型需另行注册取得；不能将 GitHub 公开等同为免授权的一键商业部署。
- **源码与实验边界：** 训练脚本和测评工具存在，但本次未下载数据集或实跑性能测试。

## 关联

- [论文归档](../papers/fah_arxiv_2611_07711.md)
- [源码归档](../repos/fah.md)
- [统一知识实体](../../wiki/entities/paper-fah.md)
