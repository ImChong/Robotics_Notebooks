# noitom-robotics/hiphi：数据文档与 Motion Viewer

- **类型：** repo / dataset / viewer / motion-capture
- **仓库：** <https://github.com/noitom-robotics/hiphi>
- **项目页：** <https://noitom-robotics.github.io/hiphi/>（[归档](../sites/hiphi.md)）
- **论文：** <https://arxiv.org/abs/2608.16222>（[摘录](../papers/hiphi_arxiv_2608_16222.md)）
- **数据集：** <https://huggingface.co/datasets/noitomrobotics/HiPHI>
- **在线 Viewer：** <https://hiphi-viewer.modalitynet.com/>
- **核查日期：** 2026-10-02
- **一句话说明：** 格式文档和可运行的离线 Viewer；支持按 Frame/LU 检索、55 关节 BVH 与同步物体回放。

## 源码开放边界

**部分开放**：已确认 viewer/run_viewer.py、viewer/hiphi_motion_viewer/、viewer/web/ 和格式文档。未找到论文策略训练/真机部署脚本或 checkpoint；网站 G1 预览不能代替这些产物。

- Viewer：Python ≥3.9，仅标准库；静态 ES modules 与随附 three.js，无需 npm 构建。
- 许可：viewer/LICENSE 为 Apache-2.0；HF 数据受独立的 ModalityNet Open Research License v1.0 与访问申请约束。
- 数据布局：32 个 .tar.zst 档；支持仅解压部分数据，列表扫描已存在的 motion 文件夹。

## 可运行入口与模块对应

| 路径 | 职责 |
|------|------|
| viewer/run_viewer.py | 调用 hiphi_motion_viewer.__main__.main |
| viewer/hiphi_motion_viewer/__main__.py | 解析数据路径、端口、浏览器选项，启动本地服务 |
| viewer/hiphi_motion_viewer/server.py | /api/config、/api/tree 与 /dataset/ 文件服务 |
| viewer/web/main.js | 获取索引、选择动作、加载元数据与播放资源 |
| viewer/web/viewer.js、object-track.js | BVH、OBJ、CSV 解析与共同帧索引播放 |
| docs/data_format.md、docs/mirroring.md | 数据单位、同步与镜像约定 |

仓库根目录运行（将示例数据路径改为已获访问权限并解压的数据根目录）：

```bash
python3 viewer/run_viewer.py /path/to/HiPHI --no-browser
```

打开 http://127.0.0.1:8666。人体 BVH 与 OBJ 网格使用厘米，物体轨迹位置为米；Viewer 将骨架和网格乘以 0.01。物体 trajectory_path 相对 motion 文件夹，mesh_path 相对数据根目录。

## 对 wiki 的映射

- [HiPHI 源码时序与工程实践](../../wiki/entities/paper-hiphi.md)
