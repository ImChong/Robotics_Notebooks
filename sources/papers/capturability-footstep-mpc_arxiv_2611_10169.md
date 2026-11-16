# 可捕获性约束的安全走廊足步规划（2611.10169）

> 论文一手原文归档

- **英文题目：** Capturability-Aware Footstep Planning in Cluttered Environments: A Recursively Feasible Sequential MPC on Axis-Aligned Safe Corridors
- **arXiv：** https://arxiv.org/abs/2611.10169
- **HTML 全文：** https://arxiv.org/html/2611.10169v1
- **PDF：** https://arxiv.org/pdf/2611.10169v1
- **机构：** 雅典国家技术大学、加州大学伯克利分校、特拉华大学、Athena 研究中心
- **入库日期：** 2026-11-16
- **核对范围：** 通读并核对 arXiv v1 II LIP/capturability 公式、III 安全走廊/虚拟停止点/序列 MPC、IV 数值与 Digit 闭门实验、V 结论；与用户提供的 29 篇 PDF 周报核对研究问题、实验口径与局限。
- **通读：** 已完成
- **一句话说明：** 基于 LIP 的 COM 与 DCM 推导多步 capture region，由障碍物生成轴对齐矩形安全走廊，利用相交区域中当前足步、COM 和虚拟停止点作为切换许可，再求解递归可行的序列足步 MPC。虚拟停止点是可行证书，不等于实际每次必须停止。

## 原文摘录（提炼）
1. **问题与动机：** 几何路线能避障，却可能无法让人形机器人停稳；论文要求相邻走廊切换时保留可捕获的未来状态。
2. **方法：** 基于 LIP 的 COM 与 DCM 推导多步 capture region，由障碍物生成轴对齐矩形安全走廊，利用相交区域中当前足步、COM 和虚拟停止点作为切换许可，再求解递归可行的序列足步 MPC。虚拟停止点是可行证书，不等于实际每次必须停止。
3. **实验：** IV-A 比较多种障碍布局的 MPC 可行性和运行开销；IV-B 使用 Digit 穿过带两道动态门的办公室环境。闭门时减速并原地踏步，打开后继续通过。结果限于 LIP 规划模型和文中场景，不代表全身非线性动力学通用安全保证。
4. **边界与局限：** 依赖 LIP 固定质心高度、矩形避障和有限步长假设；缺少复杂真实环境长时成功率。未在论文中发现已公开的官方可运行规划源码。

## 开源状态
未从本次原文核实到作者发布的可运行算法代码入口；有第三方框架引用不表示本论文完整开源。

## 对 wiki 的映射
- [Capturability-Aware MPC](../../wiki/entities/paper-capturability-footstep-mpc.md)
