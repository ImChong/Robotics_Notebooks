# 节庆视频到人形机器人长时舞台表演的 Sim2Real 阶梯（2611.11127）

> 论文一手原文归档

- **英文题目：** From Festival Video to the Stage: A Sim-to-Real Ladder for Long-Horizon Expressive Whole-Body Dance on a Commercial Humanoid
- **arXiv：** https://arxiv.org/abs/2611.11127
- **HTML 全文：** https://arxiv.org/html/2611.11127v1
- **PDF：** https://arxiv.org/pdf/2611.11127v1
- **机构：** Arithmer 株式会社研发部
- **入库日期：** 2026-11-16
- **核对范围：** 通读并核对 arXiv v1 3 视频到机器人六阶段、4 产物与状态估计、5–7 消融/切换、8 真机试验表、10 局限；与用户提供的 29 篇 PDF 周报核对研究问题、实验口径与局限。
- **通读：** 已完成
- **一句话说明：** 单目视频人体姿态恢复、G1 运动重定向与参考动作编辑后，在 Isaac Lab 2048 环境训练 BeyondMimic 风格 RL；导出含关节 PD 增益与默认姿态的 ONNX，由 RoboJuDo 在出厂行走与舞蹈控制器间交接，通过 choreographed settle 段缓解软硬控制器增益突变。

## 原文摘录（提炼）
1. **问题与动机：** 仿真能跟踪动作不等于实机能连续演出；部署的 ONNX 元数据、估计器漂移和控制器增益切换同样关键。
2. **方法：** 单目视频人体姿态恢复、G1 运动重定向与参考动作编辑后，在 Isaac Lab 2048 环境训练 BeyondMimic 风格 RL；导出含关节 PD 增益与默认姿态的 ONNX，由 RoboJuDo 在出厂行走与舞蹈控制器间交接，通过 choreographed settle 段缓解软硬控制器增益突变。
3. **实验：** v5 真机无安全绳 3/3 次完成 43 s；v6 完成含结束与自动交接的 46 s；v10/v11 完成连续 9.7 min 演出。最终仿真 timeout 比例约 99.7% 来自 10 s episode，并非同口径真机成功率；9.7 min 实机有约 1.5 m 的位置漂移。
4. **边界与局限：** 仅 G1 与特定舞蹈路线；泡棉地面使全版本失败，持续位置漂移未消除。公开依赖组件不等于作者发布了整套运行入口。

## 开源状态
未从本次原文核实到作者发布的可运行算法代码入口；有第三方框架引用不表示本论文完整开源。

## 对 wiki 的映射
- [Festival-to-Stage](../../wiki/entities/paper-festival-video-stage-dance.md)
