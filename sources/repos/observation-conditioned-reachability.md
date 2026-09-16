# observation-conditioned-reachability（OCR 官方实现）

- **标题：** Observation-Conditioned Reachability — OCR 四足安全导航
- **类型：** repo
- **仓库：** <https://github.com/albertklin/observation-conditioned-reachability>
- **项目页：** <https://sia-lab-git.github.io/One_Filter_to_Deploy_Them_All/>
- **论文：** One Filter to Deploy Them All, IEEE TRO 2026
- **收录日期：** 2026-09-15
- **开源状态：** **已开源**

## 一句话摘要

离线：`hj_reachability` 生成随机障碍 + 扰动界的 HJ 监督，训练 OCR-VN（LiDAR + 降阶状态 + $\bar{d}$ → 安全价值与梯度）。在线：状态–动作历史估计扰动，Conformal 校准后 QP 滤波 nominal $(v_x,\omega_z)$。

## 对 wiki 的映射

- [paper-one-filter-ocr-quadruped-navigation](../../wiki/entities/paper-one-filter-ocr-quadruped-navigation.md)
