# DecomVoxel 官方实现

- **地址：** https://github.com/DecomVoxel/DecomVoxel
- **论文：** https://arxiv.org/abs/2610.01914
- **项目：** https://decomvoxel.github.io/DecomVoxel-Webpage/
- **数据：** https://huggingface.co/datasets/JunfengNi/DecomVoxel
- **核对日期：** 2026-11-24
- **核对范围：** README、run_replica_demo.sh、train_demo.py 入口及并行调用，configs/replica_config.yaml，几何 SDS 的 run_sds_completion_ss / SparseStructureCompleter.complete 与外观 SDS 的 encode_object_slat / slat_sds / decode_slat_to_mesh 相关核心路径；未审计全部辅助代码，未执行 GPU demo。
- **通读：** 未完成

## 复现入口

```bash
source env.sh
bash run_replica_demo.sh
```

脚本将 Replica/scan1 输入传入 train_demo.py --mode full_pipeline，默认 200 views、3 workers，使用 configs/replica_config.yaml，输出 outputs/demo_replica/scan1。README 另要求 Python 3.10 环境、外部模型权重和条件图生成服务密钥（Seedream 默认需要 ARK_API_KEY；其他两种适配器需要 ATLASCLOUD_API_KEY）。不得把密钥写入仓库。项目公开了数据入口，但未实际下载；根目录未见独立 LICENSE 文件，授权范围需要另核实。

## 已核对的入口职责

- train_demo.py：解析 CLI，选择 GPU，run_full_pipeline 串接五阶段。
- run_train_geosvr / run_segm_3d：场景重建及实例分割。
- generate_cond_img_avo_para：按对象生成条件图，排除背景 object_000。
- run_scene_sds_para：接收 semantic_result/object_voxels 与条件图，输出 scene_sds。
- run_scene_appearance_sds_para：按对象执行外观优化。
- 已确认导入 score_distillation_sampling_ss.run_sds_completion_ss 和 score_distillation_sampling_appearance 的 encode_object_slat、slat_sds、decode_slat_to_mesh；核心调用已核对，见下节。

背景补全另有 bg_inpaint_replica.sh，不能假定由 full_pipeline 自动执行。

## 核心实现核对

- `decomvoxel/pipeline/score_distillation_sampling_ss.py`：`run_sds_completion_ss` 调用 `complete`；classic 分支以停止梯度的目标构造 MSE，周期性 redistribute 在 `ss_param.data` 的 logit 空间写入均匀随机值，而非直接清零。
- `decomvoxel/pipeline/score_distillation_sampling_appearance.py`：`encode_object_slat` 聚合 DINO 特征并编码；`slat_sds` 固定稀疏坐标、冻结 flow 模型，仅优化特征，使用 certainty 加权保真项；最终由 `decode_slat_to_mesh` 输出 mesh。
- `configs/replica_config.yaml`：几何 classic/uniform、5000 次；外观 snr、3000 次。配置的 `prune_end_iter: 2110` 与论文附录 B 的 2500 次不同，且实现以 logit -7 为目标；复现应记录配置版本，勿将论文数值视作运行实测。

## 对 wiki 的映射

- [论文归档](../papers/decomvoxel_arxiv_2610_01914.md)
- [项目归档](../sites/decomvoxel.md)
- [DecomVoxel](../../wiki/entities/paper-decomvoxel.md)
