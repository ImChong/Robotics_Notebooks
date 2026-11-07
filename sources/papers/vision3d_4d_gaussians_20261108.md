# 4D Gaussian Splatting — 一手资料归档

> 来源层（原论文或论文出版入口）；视觉与三维重建一手资料专题，2026-11-08。

- **标题：** 4D Gaussian Splatting
- **类型：** paper
- **链接：** https://openaccess.thecvf.com/content/CVPR2024/html/Wu_4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering_CVPR_2024_paper.html
- **入库日期：** 2026-11-08
- **一句话说明：** 动态场景下时空高斯渲染。
- **核对范围：** 已逐节阅读 arXiv:2310.08528v3 HTML 全文：§1–4、六平面/三头变形 Eq.7–9、§5 Table 1–4 与消融、§5.5 局限、§6 结论和 Appendix A.1–A.3；核对相应 PDF 图表（合成场景 Table 1）。
- **通读：** 已完成

## 直接一手来源（同一项目不同入口）

- [4D Gaussian Splatting 源码](https://github.com/hustvl/4DGaussians)
- [4D Gaussian Splatting 原论文](https://openaccess.thecvf.com/content/CVPR2024/html/Wu_4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering_CVPR_2024_paper.html) **（本归档主类型）**

## 原文事实记录

- **架构（§4）**：静态 3D Gaussian 集合 + 时空 HexPlane 六个二维平面 + 三个变形头（位置/旋转/尺度）+ 可微 splatting；3D-GS 静态预热约前 3000 iter。
- **复杂度（§2.2）**：为每时刻存一套高斯的空间复杂度 ~O(tN)，本工作表示为 O(N+F)（F 为共享变形场参数）。
- **数据与指标（§5）**：D-NeRF 合成 800x800 PSNR 34.05、82 FPS、8min、18MB（Table1）；HyperNeRF 25.2、34 FPS（Table2）；Neu3D 31.15、30 FPS（Table3）。均来自作者论文，**没有独立复现**。
- **消融（Table4）**：去 HexPlane PSNR 27.05；去初始化 31.91；去位置变形 26.67；全系统 34.05。
- **局限（§5.5/Appendix）**：大幅运动、缺少背景点、相机位姿误差；单目难分解静态/动态相关运动，城市场景的高斯查询规模仍是瓶颈。
- **版本**：[arXiv v3 2024-07-15](https://arxiv.org/html/2310.08528)；正文/附录链接可追溯。

## 方法与边界

动态场景下时空高斯渲染。已按指定论文版本核对方法与实验数值；源码性能/硬件泛化仍未独立运行验证。

## 对 wiki 的映射与待核验事项

- 来源汇总：[视觉三维专题 90 条来源索引](../sites/vision3d_research_primary_sources_2026_11_08.md)。
- 在升格独立 wiki 实体页前，先按论文 ID / 项目页 / 源码查重，避免与既有节点及不同项目同仓实现冲突。
- [x] 通读 arXiv v3 正文、表格、附录，明确各实验数据集与硬件。
- [x] 核对作者 README、train.py、render.py、GaussianModel、scene/deformation.py 的主要入口，时序图已写入实体页。
- [x] 新建 [4D-GS 实体](../../wiki/entities/paper-4d-gaussian-splatting-wu.md)，含原文流程图、源码时序图、实验、结论和局限；健康度以 CI 为准。
