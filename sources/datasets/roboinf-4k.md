# RoboInF-4K：操作指令保真度仿真数据 / RoboInF-4K Simulated Instruction Fidelity Dataset

- **类型：** dataset
- **数据：** [Hugging Face 数据卡与下载](https://huggingface.co/datasets/JGuo666/RoboInF-4K)
- **核验版本：** [1893df2](https://huggingface.co/datasets/JGuo666/RoboInF-4K/tree/1893df2369cd9316f4ef56928252ae309d90471d)，公开、未设访问门禁；作者发布内容 CC BY 4.0，第三方模拟资产见许可说明。
- **核对范围：** 阅读数据卡及 Hub 文件清单，下载固定版本四个评测 JSONL 验证三域，未下载全部分片或逐帧核验。
- **通读：** 已完成
- **论文与代码：** [论文归档](../papers/roboinf_arxiv_2611_30555.md) · [官方评测实现](../repos/roboinf.md)

4K 指论文中的 4,242 个原始训练、验证、清晰测试分支；发布的有标签专家示范共 3,381 条（训练 2,520 + 验证 861），包含多视角图像、状态/动作、相机参数、几何与接触信息。测试提供 861 个指令与场景配置，无专家轨迹、参考动作或成功标注。随机外观复用同一批案例。仿真记录时间可能重复且采样间隔不固定。

- **对 wiki 的映射：** [RoboInF](../../wiki/entities/paper-roboinf.md)
