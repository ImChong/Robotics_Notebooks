# Xbotics 第11讲：VLA 模型导览 —— 从 OpenVLA 到 π0 家族 / Xbotics Lecture 11: VLA Model Tour: From OpenVLA to the π0 Family

- **类型：** 官方教程章节 / course
- **维护者：** Xbotics 具身智能社区
- **官方章节：** https://xbotics-embodied-ai-club.github.io/Xbotics-Embodied-AI-Handbook/part3-end-to-end/11-VLA%E6%A8%A1%E5%9E%8B%E5%AF%BC%E8%A7%88.html
- **固定版本 Markdown：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/docs/part3-end-to-end/11-VLA%E6%A8%A1%E5%9E%8B%E5%AF%BC%E8%A7%88.md
- **核对范围：** 官方章节完整正文及参考资料列表；不把课程所引第三方工作计为独立通读。
- **通读：** 已完成
- **代码入口：** https://github.com/Xbotics-Embodied-AI-club/Xbotics-Embodied-AI-Handbook/blob/342247fb71dd0ee15386b6eab80543c50593d3c1/code/vla/4_vla_inference/4_2_pi0fast_pi05_infer/pi0fast_demo.py
- **验证边界：** 文稿核验与源码静态走读，不代表本次重训、复现实验或真机运行。

## 编译线索

按动作表示把 OpenVLA、π0、π0-FAST、π0.5 和 SmolVLA 放到同一张地图：逐维离散、频域压缩离散和连续生成对应不同的执行开销。

单次成功录像使用指定初始状态和任务，不是总体成功率。离散和连续是类别，不能用一条线性轴给方法排先后；模型大小不单独界定真机延迟。

## 对 wiki 的映射

- [第 11 讲独立知识节点](../../wiki/entities/xbotics-handbook-lecture-11.md)
- [课程主节点](../../wiki/entities/xbotics-embodied-ai-handbook.md)
