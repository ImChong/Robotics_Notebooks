# AnyTouch — 论文来源归档

## 书目信息
- **题名**：AnyTouch: Learning Unified Static-Dynamic Representation across Multiple Visuo-tactile Sensors
- **作者**：Ruoxuan Feng, Jiangyu Hu, Wenke Xia, Tianci Gao, Ao Shen, Yuhao Sun, Bin Fang, Di Hu
- **单位**：中国人民大学、武汉科技大学、北京邮电大学
- **发表**：ICLR 2025；arXiv:2502.12191（v3，2025-04-01）
- **论文**：https://arxiv.org/abs/2502.12191
- **会议页面**：https://proceedings.iclr.cc/paper_files/paper/2025/hash/4d893f766ab60e5337659b9e71883af4-Abstract-Conference.html
- **项目页**：https://gewu-lab.github.io/AnyTouch/
- **代码**：https://github.com/GeWu-Lab/AnyTouch

## 摘要与贡献
AnyTouch 研究跨视觉触觉传感器的统一静态与动态表征。作者构建 TacQuad，并以像素级图像/视频掩码建模、语义多模态对齐和跨传感器匹配学习可迁移表示；实验覆盖静态/动态触觉理解、跨传感器迁移及真实机器人倒珠任务。

TacQuad 的精细时空对齐子集包含 17,524 个接触帧、25 个物体；较粗粒度空间配对部分包含 55,082 帧、99 个物体。论文涉及 GelSight Mini、DIGIT、DuraGel、Tac3D 四类视触觉传感器。倒珠实验以目标质量 60 g、初始 100 g 为设置之一，报告 10 次测试的误差统计。

## 方法拆解
1. **静态 / 动态掩码建模**：分别从触觉图像与视频学习局部及时间变化特征，并包含下一帧预测目标。
2. **语义对齐**：将触觉、视觉与文本描述对齐，以文本语义作为跨模态锚点，支持缺失模态情形。
3. **同物体跨传感器匹配**：利用相同物体/接触位置的配对样本，降低传感器外观差异造成的域间偏移。
4. **下游评估**：冻结表征开展属性感知、跨传感器迁移及真机倒珠实验。

## 证据边界
TacQuad 的两种配对粒度不可混为一个完全时空对齐集合；倒珠是特定平台与物体条件下的机器人验证，不等同于通用操作策略。论文报告的可迁移表示结果也不意味着不同传感器无需校准即可共享原始像素域。

## 对应知识节点
- 论文详情：[paper-anytouch](../../wiki/entities/paper-anytouch.md)
- 项目详情：[project-anytouch](../../wiki/entities/project-anytouch.md)
- 主题：[触觉感知](../../wiki/concepts/tactile-sensing.md)、[视触觉融合](../../wiki/concepts/visuo-tactile-fusion.md)
