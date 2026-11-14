# WAM 纵深路线健康度验收（Issue #3490）

- 核验日期：2026-11-14（原文与源码核读于 2026-11-13）；基线提交：`0fa8164b4c0f1beaff5c091663ca9609e53e5c5a`。
- 范围：[WAM 纵深路线](../../roadmap/depth-wam.md) 各 Stage 的实际直接 wiki 引用，按节点去重；没有变更路线成员或健康度评分规则。
- 路线本身：3/3 → 3/3；62 个唯一 wiki 节点：**11 → 62 个达到 3/3**；待补齐 **51 → 0**。
- 本批前分布：3/3 11、2/3 30、1/3 21；本批后：3/3 62、2/3 0、1/3 0、0/3 0。
- 页面知识地图还直接引用 6 个路线节点（VLA、BFM、移动操作、模仿学习、导航、接触操作纵深），均为 3/3。因此地图实际显示的 **68 个唯一节点全部满分**；WAM 路线页本身另计为 3/3。
- 全部纵深路线本分支快照：30 条路线、969 个唯一 wiki 节点；3/3 248、2/3 225、1/3 496、0/3 0、缺失 0；剩余 721。详见[全局清单](depth-health-backlog.md)与 [CSV](depth-health-backlog.csv)。

## 补齐方式与核对边界

- 逐项核对主要论文全文或官方原始技术资料及必要附录；各 `original` 档案记录原文版本、核对范围和通读状态。既有满分节点的原有证据继续沿用。
- 有公开可运行实现的项目，核对官方 README 入口和相关调用链，补写对应的 Mermaid 运行时序图；没有实现的项目明确标注不适用。源码审阅范围限定在档案列出的模块，不宣称审阅全部内核。
- 修正方法、基准版本、成功率与进度分数、延迟与吞吐口径，以及当前开源范围；论文、项目页和发布代码存在差异时分别注明。
- 这是文献与源码静态核验及文档检查；未运行模型训练、GPU 推理或机器人实验，文中实验数字归属于其原始资料。
- Kairos 官方平台页面经抓取与浏览器重试仍不可访问，已在[项目页档案](../../sources/sites/kairos-acerobotics.md)记录；主要论文与官方源码可获取，原文完成状态据此认定，不把失败页面标记为通读。

## 共享来源的反向引用复核

完整 Git 历史下，来源更新触发了 40 个反向引用页的复核日期检查（其中 4 个属于本 WAM 路线，36 个为范围外关联页）。已核对它们与本批来源的主题关系和相关引用，同步 `updated`；VLA 页另同步 TrAct/Flex-π 的开源状态及 OpenWAM 的成绩。这些页面的 `original` 与 `status` 保持原有记录，范围外的骨架或原文缺项仍留待后续批次，不将引用复核等同于重新通读整页原文。

预检另发现基线中既有的两处陈旧关联：Ropedia W0 → UnifoLM-WMA 与 Xperience 2 → EgoXperience。已核对归档与对应比较/关联节，更新这两页复核日期；同样保留原文通读状态及原有缺项。

## Stage 与节点前后分数

同一节点若被多个 Stage 引用，会在表中分别出现；唯一节点总数按路径去重。原文证据列链接到各节点 `original` 指定的档案。

### Stage 0 边界与族谱：什么是 / 不是 WAM

| 节点 | 前 → 后 | 本次处理 | 原文证据 |
|---|---:|---|---|
| [wiki/concepts/world-action-models.md](../../wiki/concepts/world-action-models.md) | 3/3 → 3/3 | 保留已完成证据，同步关联实例口径 | [wam_survey_reading_arxiv_2605_12090.md](../../sources/papers/wam_survey_reading_arxiv_2605_12090.md) |
| [wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md](../../wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md) | 3/3 → 3/3 | 沿用已完成证据 | [paligemma_arxiv_2407_07726.md](../../sources/papers/paligemma_arxiv_2407_07726.md)<br>[gr00t_n1_arxiv_2503_14734.md](../../sources/papers/gr00t_n1_arxiv_2503_14734.md)<br>[cosmos_wfm_arxiv_2501_03575.md](../../sources/papers/cosmos_wfm_arxiv_2501_03575.md)<br>[pi07.md](../../sources/papers/pi07.md)<br>[worldlabs_functional_taxonomy_world_models.md](../../sources/blogs/worldlabs_functional_taxonomy_world_models.md) |
| [wiki/queries/embodied-fm-taxonomy-loop.md](../../wiki/queries/embodied-fm-taxonomy-loop.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [paligemma_arxiv_2407_07726.md](../../sources/papers/paligemma_arxiv_2407_07726.md)<br>[gr00t_n1_arxiv_2503_14734.md](../../sources/papers/gr00t_n1_arxiv_2503_14734.md)<br>[cosmos_wfm_arxiv_2501_03575.md](../../sources/papers/cosmos_wfm_arxiv_2501_03575.md)<br>[mint_rss_2026.md](../../sources/papers/mint_rss_2026.md) |
| [wiki/entities/paper-rcl-wam-robot-learning-control-survey.md](../../wiki/entities/paper-rcl-wam-robot-learning-control-survey.md) | 3/3 → 3/3 | 沿用已完成证据 | [rcl_wam_survey_reading_arxiv_2609_16074.md](../../sources/papers/rcl_wam_survey_reading_arxiv_2609_16074.md) |

### Stage 1 世界模型与动力学基座

| 节点 | 前 → 后 | 本次处理 | 原文证据 |
|---|---:|---|---|
| [wiki/methods/generative-world-models.md](../../wiki/methods/generative-world-models.md) | 3/3 → 3/3 | 保留已完成证据，同步关联实例口径 | [gaia1_reading_arxiv_2309_17080.md](../../sources/papers/gaia1_reading_arxiv_2309_17080.md)<br>[unisim_reading_arxiv_2310_06114.md](../../sources/papers/unisim_reading_arxiv_2310_06114.md) |
| [wiki/methods/model-based-rl.md](../../wiki/methods/model-based-rl.md) | 3/3 → 3/3 | 沿用已完成证据 | [mbpo_reading_arxiv_1906_08253_reading_mbrl.md](../../sources/papers/mbpo_reading_arxiv_1906_08253_reading_mbrl.md) |
| [wiki/overview/robot-world-models-training-loop-taxonomy.md](../../wiki/overview/robot-world-models-training-loop-taxonomy.md) | 3/3 → 3/3 | 沿用已完成证据 | [wm_robot_survey_reading_arxiv_2605_00080_tax1.md](../../sources/papers/wm_robot_survey_reading_arxiv_2605_00080_tax1.md) |
| [wiki/entities/tau0-world-model.md](../../wiki/entities/tau0-world-model.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [tau0_wm_tech_report.md](../../sources/papers/tau0_wm_tech_report.md) |
| [wiki/entities/paper-physisforcing.md](../../wiki/entities/paper-physisforcing.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [physisforcing_arxiv_2606_28128.md](../../sources/papers/physisforcing_arxiv_2606_28128.md) |
| [wiki/entities/paper-physmani-dynamic-manipulation-world-model.md](../../wiki/entities/paper-physmani-dynamic-manipulation-world-model.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [physmani_arxiv_2607_01938.md](../../sources/papers/physmani_arxiv_2607_01938.md) |
| [wiki/entities/xiaomi-robotics-u0.md](../../wiki/entities/xiaomi-robotics-u0.md) | 1/3 → 3/3 | 补齐：阅读骨架不完整；缺 original 归档 | [xiaomi_robotics_u0_arxiv_2607_11643.md](../../sources/papers/xiaomi_robotics_u0_arxiv_2607_11643.md) |
| [wiki/entities/paper-odeworld.md](../../wiki/entities/paper-odeworld.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [odeworld_arxiv_2607_27924.md](../../sources/papers/odeworld_arxiv_2607_27924.md) |
| [wiki/entities/paper-hydra-0.md](../../wiki/entities/paper-hydra-0.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [hydra_0_arxiv_2608_18077.md](../../sources/papers/hydra_0_arxiv_2608_18077.md) |
| [wiki/entities/paper-causalwm.md](../../wiki/entities/paper-causalwm.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [causalwm_arxiv_2609_23184.md](../../sources/papers/causalwm_arxiv_2609_23184.md) |
| [wiki/entities/paper-task-centric-world-models.md](../../wiki/entities/paper-task-centric-world-models.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [tc_wm_arxiv_2605_25620.md](../../sources/papers/tc_wm_arxiv_2605_25620.md) |
| [wiki/entities/paper-cd-lam.md](../../wiki/entities/paper-cd-lam.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [cd_lam_arxiv_2607_09185.md](../../sources/papers/cd_lam_arxiv_2607_09185.md) |
| [wiki/entities/paper-scar-continuous-action.md](../../wiki/entities/paper-scar-continuous-action.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [scar_arxiv_2605_16412.md](../../sources/papers/scar_arxiv_2605_16412.md) |
| [wiki/entities/paper-h-jepa-visual-planning.md](../../wiki/entities/paper-h-jepa-visual-planning.md) | 1/3 → 3/3 | 补齐：阅读骨架不完整；缺 original 归档 | [hjepa_arxiv_2610_06805.md](../../sources/papers/hjepa_arxiv_2610_06805.md) |
| [wiki/entities/paper-dreamtrue-action-faithful-world-model.md](../../wiki/entities/paper-dreamtrue-action-faithful-world-model.md) | 1/3 → 3/3 | 补齐：阅读骨架不完整；缺 original 归档 | [dreamtrue_arxiv_2610_12468.md](../../sources/papers/dreamtrue_arxiv_2610_12468.md) |
| [wiki/entities/paper-ad-e2e-jepa.md](../../wiki/entities/paper-ad-e2e-jepa.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [ad_e2e_jepa_arxiv_2609_34085.md](../../sources/papers/ad_e2e_jepa_arxiv_2609_34085.md) |
| [wiki/entities/paper-d-jepa-decision-aligned-latent-world-model.md](../../wiki/entities/paper-d-jepa-decision-aligned-latent-world-model.md) | 1/3 → 3/3 | 补齐：阅读骨架不完整；缺 original 归档 | [d_jepa_decision_aligned_latent_world_model_arxiv_2609_24749.md](../../sources/papers/d_jepa_decision_aligned_latent_world_model_arxiv_2609_24749.md) |
| [wiki/entities/paper-dr-wm.md](../../wiki/entities/paper-dr-wm.md) | 1/3 → 3/3 | 补齐：阅读骨架不完整；缺 original 归档 | [dr_wm_arxiv_2610_17630.md](../../sources/papers/dr_wm_arxiv_2610_17630.md) |
| [wiki/entities/paper-murea-inworld.md](../../wiki/entities/paper-murea-inworld.md) | 1/3 → 3/3 | 补齐：阅读骨架不完整；缺 original 归档；status 限分 | [murea_inworld_technical_report.md](../../sources/papers/murea_inworld_technical_report.md) |
| [wiki/entities/edenlive.md](../../wiki/entities/edenlive.md) | 1/3 → 3/3 | 补齐：阅读骨架不完整；缺 original 归档 | [edenlive_technical_report.md](../../sources/papers/edenlive_technical_report.md) |

### Stage 2 Cascaded WAM：先计划未来，再解码动作

| 节点 | 前 → 后 | 本次处理 | 原文证据 |
|---|---:|---|---|
| [wiki/methods/defi-decoupled-dynamics-vla.md](../../wiki/methods/defi-decoupled-dynamics-vla.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [defi_arxiv_2604_16391.md](../../sources/papers/defi_arxiv_2604_16391.md) |
| [wiki/concepts/world-action-models.md](../../wiki/concepts/world-action-models.md) | 3/3 → 3/3 | 保留已完成证据，同步关联实例口径 | [wam_survey_reading_arxiv_2605_12090.md](../../sources/papers/wam_survey_reading_arxiv_2605_12090.md) |
| [wiki/entities/paper-being-m07-humanoid-latent-wam.md](../../wiki/entities/paper-being-m07-humanoid-latent-wam.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [being_m07.md](../../sources/papers/being_m07.md) |
| [wiki/entities/paper-slim-05b.md](../../wiki/entities/paper-slim-05b.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [slim_05b_arxiv_2608_09771.md](../../sources/papers/slim_05b_arxiv_2608_09771.md) |
| [wiki/entities/paper-tract.md](../../wiki/entities/paper-tract.md) | 1/3 → 3/3 | 补齐：阅读骨架不完整；缺 original 归档 | [tract_arxiv_2608_24101.md](../../sources/papers/tract_arxiv_2608_24101.md) |
| [wiki/overview/robot-world-models-action-consequence-technology-map.md](../../wiki/overview/robot-world-models-action-consequence-technology-map.md) | 1/3 → 3/3 | 补齐：阅读骨架不完整；缺 original 归档 | [dswam_arxiv_2607_04927.md](../../sources/papers/dswam_arxiv_2607_04927.md)<br>[dynawm_arxiv_2607_02604.md](../../sources/papers/dynawm_arxiv_2607_02604.md)<br>[dreamsteer_arxiv_2607_02865.md](../../sources/papers/dreamsteer_arxiv_2607_02865.md)<br>[vt_wam_arxiv_2607_02503.md](../../sources/papers/vt_wam_arxiv_2607_02503.md)<br>[meco_wam_arxiv_2607_05468.md](../../sources/papers/meco_wam_arxiv_2607_05468.md) |

### Stage 3 Joint WAM：共享骨干下联合预测未来与动作

| 节点 | 前 → 后 | 本次处理 | 原文证据 |
|---|---:|---|---|
| [wiki/entities/paper-dit4dit-video-action-model.md](../../wiki/entities/paper-dit4dit-video-action-model.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [dit4dit_arxiv_2603_10448.md](../../sources/papers/dit4dit_arxiv_2603_10448.md) |
| [wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md](../../wiki/entities/paper-motionwam-humanoid-loco-manipulation-wam.md) | 3/3 → 3/3 | 沿用已完成证据 | [motionwam_reading_arxiv_2606_09215_mw1.md](../../sources/papers/motionwam_reading_arxiv_2606_09215_mw1.md) |
| [wiki/entities/paper-abot-m05-mobile-manipulation-wam.md](../../wiki/entities/paper-abot-m05-mobile-manipulation-wam.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [abot_m05_arxiv_2607_00678.md](../../sources/papers/abot_m05_arxiv_2607_00678.md) |
| [wiki/entities/dexmal-dw05.md](../../wiki/entities/dexmal-dw05.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [dexmal_opendw.md](../../sources/repos/dexmal_opendw.md) |
| [wiki/entities/lumo-2.md](../../wiki/entities/lumo-2.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [lumo2_arxiv_2607_11270.md](../../sources/papers/lumo2_arxiv_2607_11270.md) |
| [wiki/methods/pelican-unified-1.md](../../wiki/methods/pelican-unified-1.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [pelican_unified_uei_arxiv_2605_15153.md](../../sources/papers/pelican_unified_uei_arxiv_2605_15153.md) |
| [wiki/entities/paper-kairos-native-world-model-stack.md](../../wiki/entities/paper-kairos-native-world-model-stack.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [kairos_arxiv_2606_16533.md](../../sources/papers/kairos_arxiv_2606_16533.md) |
| [wiki/entities/cosmos-3.md](../../wiki/entities/cosmos-3.md) | 1/3 → 3/3 | 补齐：阅读骨架不完整；缺 original 归档 | [cosmos3_arxiv_2606_02800.md](../../sources/papers/cosmos3_arxiv_2606_02800.md) |
| [wiki/entities/robbyant.md](../../wiki/entities/robbyant.md) | 3/3 → 3/3 | 沿用已完成证据 | [antgroup_robbyant.md](../../sources/sites/antgroup_robbyant.md)<br>[technology-robbyant-com.md](../../sources/sites/technology-robbyant-com.md) |
| [wiki/entities/paper-worldscape-policy-2.md](../../wiki/entities/paper-worldscape-policy-2.md) | 1/3 → 3/3 | 补齐：阅读骨架不完整；缺 original 归档 | [worldscape_policy_2_arxiv_2607_18840.md](../../sources/papers/worldscape_policy_2_arxiv_2607_18840.md) |
| [wiki/entities/paper-omega-0.md](../../wiki/entities/paper-omega-0.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [omega0_arxiv_2608_06375.md](../../sources/papers/omega0_arxiv_2608_06375.md) |
| [wiki/entities/paper-dreamwam.md](../../wiki/entities/paper-dreamwam.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [dreamwam_arxiv_2608_04996.md](../../sources/papers/dreamwam_arxiv_2608_04996.md) |
| [wiki/entities/dyna-2.md](../../wiki/entities/dyna-2.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [dyna_2_million_hour_wam.md](../../sources/blogs/dyna_2_million_hour_wam.md) |
| [wiki/entities/paper-riemann-1.md](../../wiki/entities/paper-riemann-1.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [riemann_1_0.md](../../sources/papers/riemann_1_0.md) |
| [wiki/entities/paper-fact.md](../../wiki/entities/paper-fact.md) | 3/3 → 3/3 | 沿用已完成证据 | [fact_arxiv_2608_10232.md](../../sources/papers/fact_arxiv_2608_10232.md) |
| [wiki/entities/paper-motubrain.md](../../wiki/entities/paper-motubrain.md) | 1/3 → 3/3 | 补齐：阅读骨架不完整；缺 original 归档 | [motubrain_arxiv_2604_27792.md](../../sources/papers/motubrain_arxiv_2604_27792.md) |
| [wiki/entities/paper-flex-pi.md](../../wiki/entities/paper-flex-pi.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [flex_pi_arxiv_2608_10860.md](../../sources/papers/flex_pi_arxiv_2608_10860.md) |
| [wiki/entities/paper-glancewam.md](../../wiki/entities/paper-glancewam.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [glancewam_arxiv_2608_23927.md](../../sources/papers/glancewam_arxiv_2608_23927.md) |
| [wiki/entities/paper-openwam.md](../../wiki/entities/paper-openwam.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [openwam_arxiv_2609_07398.md](../../sources/papers/openwam_arxiv_2609_07398.md) |

### Stage 4 部署职责三分：执行 · 修正 · 筛选（+ 接触 / 几何）

| 节点 | 前 → 后 | 本次处理 | 原文证据 |
|---|---:|---|---|
| [wiki/overview/wm-action-consequence-category-01-wam-action-prediction.md](../../wiki/overview/wm-action-consequence-category-01-wam-action-prediction.md) | 1/3 → 3/3 | 补齐：阅读骨架不完整；缺 original 归档 | [dswam_arxiv_2607_04927.md](../../sources/papers/dswam_arxiv_2607_04927.md)<br>[dynawm_arxiv_2607_02604.md](../../sources/papers/dynawm_arxiv_2607_02604.md)<br>[dreamsteer_arxiv_2607_02865.md](../../sources/papers/dreamsteer_arxiv_2607_02865.md) |
| [wiki/overview/robot-world-models-action-consequence-technology-map.md](../../wiki/overview/robot-world-models-action-consequence-technology-map.md) | 1/3 → 3/3 | 补齐：阅读骨架不完整；缺 original 归档 | [dswam_arxiv_2607_04927.md](../../sources/papers/dswam_arxiv_2607_04927.md)<br>[dynawm_arxiv_2607_02604.md](../../sources/papers/dynawm_arxiv_2607_02604.md)<br>[dreamsteer_arxiv_2607_02865.md](../../sources/papers/dreamsteer_arxiv_2607_02865.md)<br>[vt_wam_arxiv_2607_02503.md](../../sources/papers/vt_wam_arxiv_2607_02503.md)<br>[meco_wam_arxiv_2607_05468.md](../../sources/papers/meco_wam_arxiv_2607_05468.md) |
| [wiki/entities/paper-dswam-dual-system-wam.md](../../wiki/entities/paper-dswam-dual-system-wam.md) | 1/3 → 3/3 | 补齐：阅读骨架不完整；缺 original 归档 | [dswam_arxiv_2607_04927.md](../../sources/papers/dswam_arxiv_2607_04927.md) |
| [wiki/entities/paper-dynawm-vla-online-correction.md](../../wiki/entities/paper-dynawm-vla-online-correction.md) | 1/3 → 3/3 | 补齐：阅读骨架不完整；缺 original 归档 | [dynawm_arxiv_2607_02604.md](../../sources/papers/dynawm_arxiv_2607_02604.md) |
| [wiki/entities/paper-dreamsteer-vla-deployment-steering.md](../../wiki/entities/paper-dreamsteer-vla-deployment-steering.md) | 1/3 → 3/3 | 补齐：阅读骨架不完整；缺 original 归档 | [dreamsteer_arxiv_2607_02865.md](../../sources/papers/dreamsteer_arxiv_2607_02865.md) |
| [wiki/entities/paper-vt-wam-visuotactile-contact-rich.md](../../wiki/entities/paper-vt-wam-visuotactile-contact-rich.md) | 1/3 → 3/3 | 补齐：阅读骨架不完整；缺 original 归档 | [vt_wam_arxiv_2607_02503.md](../../sources/papers/vt_wam_arxiv_2607_02503.md) |
| [wiki/entities/paper-meco-wam-4d-geometry-cotraining.md](../../wiki/entities/paper-meco-wam-4d-geometry-cotraining.md) | 1/3 → 3/3 | 补齐：阅读骨架不完整；缺 original 归档 | [meco_wam_arxiv_2607_05468.md](../../sources/papers/meco_wam_arxiv_2607_05468.md) |

### Stage 5 数据、评测与进阶方向

| 节点 | 前 → 后 | 本次处理 | 原文证据 |
|---|---:|---|---|
| [wiki/entities/paper-egowam-egocentric-human-wam-co-training.md](../../wiki/entities/paper-egowam-egocentric-human-wam-co-training.md) | 3/3 → 3/3 | 沿用已完成证据 | [egowam_arxiv_2607_08436.md](../../sources/papers/egowam_arxiv_2607_08436.md) |
| [wiki/entities/paper-wam-ttt-human-video-test-time-steering.md](../../wiki/entities/paper-wam-ttt-human-video-test-time-steering.md) | 3/3 → 3/3 | 沿用已完成证据 | [wam_ttt_arxiv_2607_06988.md](../../sources/papers/wam_ttt_arxiv_2607_06988.md) |
| [wiki/entities/dyna-2.md](../../wiki/entities/dyna-2.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [dyna_2_million_hour_wam.md](../../sources/blogs/dyna_2_million_hour_wam.md) |
| [wiki/methods/egoscale.md](../../wiki/methods/egoscale.md) | 1/3 → 3/3 | 补齐：阅读骨架不完整；缺 original 归档 | [egoscale_arxiv_2602_16710.md](../../sources/papers/egoscale_arxiv_2602_16710.md) |
| [wiki/entities/paper-navwam-goal-conditioned-visual-navigation-wam.md](../../wiki/entities/paper-navwam-goal-conditioned-visual-navigation-wam.md) | 1/3 → 3/3 | 补齐：阅读骨架不完整；缺 original 归档 | [navwam_arxiv_2606_13494.md](../../sources/papers/navwam_arxiv_2606_13494.md) |
| [wiki/entities/paper-worldvln-aerial-vln-wam.md](../../wiki/entities/paper-worldvln-aerial-vln-wam.md) | 1/3 → 3/3 | 补齐：阅读骨架不完整；缺 original 归档 | [worldvln_arxiv_2605_15964.md](../../sources/papers/worldvln_arxiv_2605_15964.md) |
| [wiki/entities/paper-gigaworld-1-policy-evaluation.md](../../wiki/entities/paper-gigaworld-1-policy-evaluation.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [gigaworld_1_arxiv_2607_02642.md](../../sources/papers/gigaworld_1_arxiv_2607_02642.md) |
| [wiki/entities/paper-sc3-eval.md](../../wiki/entities/paper-sc3-eval.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [sc3_eval_arxiv_2606_18610.md](../../sources/papers/sc3_eval_arxiv_2606_18610.md) |
| [wiki/entities/paper-harnesseval-w.md](../../wiki/entities/paper-harnesseval-w.md) | 2/3 → 3/3 | 补齐：缺 original 归档 | [harnesseval_w_arxiv_2608_16859.md](../../sources/papers/harnesseval_w_arxiv_2608_16859.md) |

## 复核命令

```bash
python3 scripts/audit_depth_health.py --write
python3 scripts/audit_depth_health.py --latest 1
python3 scripts/report_health.py --changed
make ci-preflight
make ci-test
```

`--changed` 按提交相对 `origin/main` 的 wiki 变更出表；完整 62 节点以本页与实际 Stage 清单为准。

## 本地验证结果（2026-11-14）

- `make ci-preflight`：通过；7,571 张 Mermaid 图无语法错误，Wiki lint 无阻塞问题，导出质量检查 15/15。另有 6 条信息型提示，不影响通过。
- `make ci-test`：通过；570 项 Python 测试、843 项子测试、130 项前端测试全部成功，类型检查与依赖审计通过。
- Chromium 验证：地图 68 个唯一节点（146 次显示）均为 3/3；OpenWAM 详情页两张 Mermaid 图实际渲染成功，页面 JavaScript 异常为 0。
- 重新生成的站点数据同样确认地图 68/68 满分，WAM 路线页自身为 3/3。
