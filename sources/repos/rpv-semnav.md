# UTS-RI/RPV-SemNav

> 来源归档（repo）

- **名称：** RPV-SemNav
- **类型：** repo / objectnav / habitat / zero-shot
- **URL：** <https://github.com/UTS-RI/RPV-SemNav>
- **论文：** [arXiv:2607.25448](../papers/rpv_semnav_arxiv_2607_25448.md)
- **项目页：** <https://uts-ri.github.io/RPV-SemNav/>
- **机构：** UTS Robotics Institute
- **许可证：** 见仓库（入库日未逐条审计 LICENSE 文件）
- **入库日期：** 2026-09-23
- **一句话说明：** **Room Probability Vector** 零样本室内语义导航官方实现：Habitat-Sim 0.3.3、VLFM/frontier_exploration 可编辑安装、双终端 vision server + `python -m vlfm.run` 评测 HM3D ObjectNav。

## 运行入口（README）

| 步骤 | 命令 / 模块 |
|------|-------------|
| 环境 | `conda env create -f env_installation_files/environment-rpv.yml` → `rpv` |
| Vision servers | `./scripts/launch_dl_servers.sh` |
| HM3D 评测 | `python -m vlfm.run` |
| 视频输出 | `habitat_baselines.video_dir=...` 等 hydra 覆盖 |

## 开源边界（2026-09-23）

- **已有：** 安装文档、habitat 编译说明、数据集下载步骤、评测入口
- **TODO（仓库自述）：** Add V1 code；checkpoint 下载链接 WIP
- **依赖：** Matterport HM3D token、CUDA habitat-sim、Mask2Former CUDA patch

## 对 wiki 的映射

- [paper-rpv-semnav](../../wiki/entities/paper-rpv-semnav.md) — 源码运行时序图对齐本 README
