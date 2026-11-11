# TacitVLA 官方项目页

> 来源归档（site · 作者项目主页）

- **标题：** TacitVLA: Reasoning Vision-Language-Action Models Without Reasoning Latency
- **类型：** site / project-page
- **官网：** <https://tacitvla.github.io/>
- **论文：** <https://arxiv.org/abs/2610.17540>（v1，2026-10-13）
- **代码：** 官方网站截至 2026-11-11 标记“Code soon”，未展示 GitHub / Hugging Face / 运行入口
- **权重与数据：** 网站未提供 TacitVLA 模型权重或自有训练数据下载链接
- **入库日期：** 2026-11-11
- **核对范围：** 通读主页 Abstract、Method、Results、Latency、Benchmarks、BibTeX 与资源入口，并对照 arXiv v1 正文和附录。
- **通读：** 已完成
- **沉淀到 wiki：** [TacitVLA](../../wiki/entities/paper-tacitvla.md)

## 项目定位与机制

- 作者团队来自 NVIDIA、NYU、UC Berkeley、Johns Hopkins 和 Stanford。
- 关键对照只有同次文本推理 token 对动作专家的可见性：TacitVLA 继续接受 reasoning 训练监督，但动作不读当前 r_t。
- 无记忆任务部署时省略推理文本输出；记忆任务先输出动作，动作执行期间生成账本条目供后续调用读取。
- 主页列举 RoboMME、RMBench、LIBERO (+Plus)、PhysicalAI-AV 的对照指标以及单张 H100 的调用延迟。

## 可复现性与限制

- **状态：待发布代码**，既不是“已开源”，也不是“已验证不能复现”。
- 不建立虚构的 sources/repos/ 页面，官方出现有效源码仓库后再核查 README、权重、许可证与可运行入口。
- 论文驾驶训练使用 380 万条私有数据；公开基准结果不等于全部训练材料已公开。
