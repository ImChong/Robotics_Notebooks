# Xbotics 第18讲：VLN 实操与评测 / Xbotics Lecture 18: Vision-and-Language Navigation: Methods and Practice

- **类型：** 官方教程章节 / course
- **维护者：** Xbotics 具身智能社区
- **官方章节：** https://xbotics-embodied-ai-club.github.io/Xbotics-Embodied-AI-Handbook/part5-vln/18-vln-practice.html
- **固定版本 Markdown：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/docs/part5-vln/18-vln-practice.md
- **核对范围：** 官方章节完整正文及参考资料列表；不把课程所引第三方工作计为独立通读。
- **通读：** 已完成
- **代码入口：** https://github.com/cshizhe/VLN-DUET/blob/93e8b233164bc079a6db48b8a0a78d123ec8de41/map_nav_src/r2r/main_nav.py
- **验证边界：** 文稿核验与源码静态走读，不代表本次重训、复现实验或真机运行。

## 编译线索

讲解 VLN 从 Seq2Seq 和 Speaker-Follower/EnvEdit 增强，经 Transformer 预训练、历史状态与地图记忆，演进到 LLM 推理、世界预测和双系统协作。实践分别指向 R2R 基线、DUET 和 InternVLA-N1。

实验复现依赖 Matterport 资源许可、特征、权重和各项目依赖版本，离散图实验不保证连续环境同等指标；时序图仅跟踪实际核对的 DUET 评测入口。

## 对 wiki 的映射

- [第 18 讲独立知识节点](../../wiki/entities/xbotics-handbook-lecture-18.md)
- [课程主节点](../../wiki/entities/xbotics-embodied-ai-handbook.md)
