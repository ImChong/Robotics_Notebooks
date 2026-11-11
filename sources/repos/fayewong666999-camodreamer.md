# CamoDreamer 官方项目仓库（开放状态审读）

- **地址：** https://github.com/fayewong666999/CamoDreamer
- **论文：** https://arxiv.org/abs/2607.17768
- **项目身份：** 论文脚注把该 GitHub 地址列为 Project page，README 标注 ACMMM2026，并自述 Oral
- **核对快照：** `main@159921e0ce2c380428ac51bbe796e2e531b607b4`（2026-09-05 最近可见提交）
- **核对日期：** 2026-11-11
- **核对范围：** 完整阅读 README（论文身份、作者及机构、News、生成结果下载链接），核查默认分支根目录与对应提交。仅见 `README.md`、`asset.jpg`，未见训练脚本、推理脚本、模型权重、配置、许可文件或可运行示例；未检查第三方镜像与隐藏分支。
- **通读：** 已完成
- **代码状态：** **尚未公开可运行实现**。虽然仓库标题为 Implementation，README 的 News 只写预期在 11 月左右开放代码，实际默认分支尚无实现，不能判定已经开源。
- **生成图像：** README 提供 Google Drive 下载链接（作者称为 generated results）：https://drive.google.com/file/d/1Hl7cGwEGPnTnE4lj5okkl9Hf--cbDcZb/view?usp=sharing 。本次未独立下载/校验内容，不把生成结果等同训练集或 checkpoint。
- **时间说明：** README 的部分 News 条目标注 2025 年而论文/仓库为 2026 年；本归档按实际检查的仓库状态记录，不将这些日期作为发布时间证据。

## 已验证边界

当前无法得到实际 CLI 入口、模块类名、参数文件与函数调用栈，所以不绘制伪造的“源码运行时序图”。Wiki 的 Mermaid 主流程图是依据论文 §3 的**算法结构**绘制，不能充当源代码证明。

未来若官方放出实现，先核对许可证、安装说明、checkpoint 与训练/推理入口，再按真实函数和文件补充 sequenceDiagram。

## 对应知识库

- [CamoDreamer 论文实体](../../wiki/entities/paper-camodreamer-context-decoupled-camouflage-generation.md)
- [论文全文审读归档](../papers/camodreamer_arxiv_2607_17768v2.md)
