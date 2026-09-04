# EasyMocap 文档站（chingswy.github.io/easymocap-public-doc）

> 来源归档（ingest 配套站点）

- **URL：** <https://chingswy.github.io/easymocap-public-doc/>
- **对应仓库：** [zju3dv/EasyMocap](https://github.com/zju3dv/EasyMocap)
- **机构：** 浙江大学 CAD&CG 三维视觉组
- **入库日期：** 2026-09-04
- **一句话说明：** EasyMocap 官方公开文档：安装、Quick Start、多 setting 动捕与 Neural Body / MultiNeuralBody 工作页。

## 开源核查（步骤 2.5，2026-09-04）

文档站头部与 GitHub README 一致指向 **已开源代码仓**。页内同时给出：

- **代码：** <https://github.com/zju3dv/EasyMocap>
- **ZJU-MoCap 申请：** 协议 PDF + 邮件 Qing Shuai（`s_q@zju.edu.cn`）并 cc Xiaowei Zhou
- **数据集展示页：** <https://chingswy.github.io/Dataset-Demo/>
- **分仓 / 工作页：** Neural Body、Mirrored-Human、mvpose、iMocap、MultiNeuralBody

人体模型权重不在文档站分发，安装页要求从 SMPL / MANO / FLAME 官方站点下载后放入 `data/bodymodels/`。

## 页面要点（2026-09 快照）

### 核心 setting（文档站 Core features）

| 板块 | 文档主张 |
|------|----------|
| MoCap Anywhere | 少量标定相机即可在户外消费级手机阵列上恢复困难动作（示例：9 部智能手机） |
| Internet video | 对 YouTube 等单目野外视频拟合 SMPL |
| Internet video with a mirror | 镜面人（CVPR 2021） |
| Multiple Internet videos of a specific action | iMocap，文档仍标 Coming soon |
| Multiple views of multiple people | 8 路消费级相机多人重建 |
| Novel view synthesis | Neural Body / 人人交互 MultiNeuralBody |

### 安装页摘要

- 快捷路径钉死 **Python 3.9 + 指定 CUDA/PyTorch wheel**，再 `pip install -r requirements.txt` 与 `python setup.py develop`。
- 2023-06-30 更新示例改为 `cu116` + `torch 1.12.0` + `spconv-cu116`。
- Neural Body 另装 `requirements_neuralbody.txt` 与 `spconv`。

## 对 wiki 的映射

- 仓库归档：[easymocap.md](../repos/easymocap.md)
- 实体页：[wiki/entities/easymocap.md](../../wiki/entities/easymocap.md)
