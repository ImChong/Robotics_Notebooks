# R2RI 数据集与仓库（GitHub / Hugging Face）

> 来源归档

- **标题：** R2RI: A Multi-View Event and RGB Dataset for Robot-to-Robot Interaction
- **类型：** site（数据集页 + README 仓库）
- **GitHub：** <https://github.com/MagriniGabriele/R2RI>
- **Hugging Face 数据集：** <https://huggingface.co/datasets/r2ri/R2RI>（2026-10-12 核查可访问）
- **论文：** <https://arxiv.org/abs/2610.07117>（归档见 [r2ri_arxiv_2610_07117.md](../papers/r2ri_arxiv_2610_07117.md)）
- **代码：** 无训练/评测代码（仓库仅 README 与 assets，论文称基线代码录用后发布）
- **入库日期：** 2026-10-12
- **一句话说明：** 机器人间交互事件 + RGB 数据集，约 5,000 序列 / 650 万帧，4 种人形 × 20 类交互 × 6 个场景。

## 核查结论（步骤 2.5）

| 项 | 结论 |
|----|------|
| 数据集 | 已公开（Hugging Face） |
| 标注 | 论文称随数据发布；README 未描述文件结构与标注格式 |
| 代码 | 仅 README 与 assets，1 次提交；基线代码待发布 |
| 许可 | README 与论文均未单独给出数据集许可 |
| 联系人 | Gabriele Magrini（README） |

README 说明模态为 RGB、Depth、Event，光照含标准与降低亮度条件；构建流程为 INTER-X 人类交互重定向到机器人 URDF，渲染 RGB 后由高帧率 RGB 模拟事件流。

## 对 wiki 的映射

- [paper-r2ri-robot-interaction-dataset.md](../../wiki/entities/paper-r2ri-robot-interaction-dataset.md)
