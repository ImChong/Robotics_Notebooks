# HumanoidTTT 项目页来源归档

> 来源归档（site / project-page）

- **标题：** HumanoidTTT: Test-Time Capability Reuse for Efficient Humanoid Control
- **类型：** 项目主页
- **机构：** 北京大学计算机学院（Peking University）
- **官方入口：** https://aigeeksgroup.github.io/HumanoidTTT/
- **论文：** https://arxiv.org/abs/2610.00198
- **代码：** https://github.com/AIGeeksGroup/HumanoidTTT
- **模型权重：** https://huggingface.co/AIGeeksGroup/HumanoidTTT
- **入库日期：** 2026-10-16
- **一句话说明：** 通过入口状态认证复用已验证的人形完整动作，并用部署后的复用反馈管理有限容量动作库。

## 页面与资源核查

项目主页本次抓取工具未能读取正文；主页地址由 arXiv 全文和官方仓库 README 相互印证。以下开源状态依据官方 arXiv HTML 与仓库 README / 文件目录核实：

- **代码：** 官方 GitHub 仓库公开，包含 src/ Python 包和安装说明。组件覆盖入口特征、适用性证书、能力整合策略、动作特征、存储辅助、签名、类型及哈希。
- **权重：** Hugging Face 发布初始能力整合策略 checkpoint、config.json 和 SHA256SUMS。
- **边界：** 仓库提供方法组件；运行应用需自行接入动作生成器、HoloMotion 跟踪运行时、动作档案、入口几何和执行循环。README 指向外部 OMG 与 HoloMotion 项目及相应 checkpoint。
- **许可：** 本次查看的仓库根目录未见 LICENSE 文件；复用代码前应先确认许可条款。

## 对 wiki 的映射

- [HumanoidTTT 实体页](../../wiki/entities/paper-humanoidttt.md)
- [论文来源归档](../papers/humanoidttt_arxiv_2610_00198.md)
- [代码来源归档](../repos/humanoidttt.md)

## 一手来源

- [HumanoidTTT 项目主页](https://aigeeksgroup.github.io/HumanoidTTT/)
- [官方 GitHub README](https://github.com/AIGeeksGroup/HumanoidTTT)
- [官方模型仓库](https://huggingface.co/AIGeeksGroup/HumanoidTTT)
- [arXiv HTML](https://arxiv.org/html/2610.00198v1)
