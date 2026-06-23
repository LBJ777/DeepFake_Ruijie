# FreqPRISM 实验设计方案

日期：2026-05-26

## 0. 实验原则

FreqPRISM 当前定位是 strict source-only single detector。后续实验可以扩展评估集、补充消融、增加 source-only component-gate learning 或 fusion-weight calibration，但不能用目标集标签做训练、组件门控选择、融合权重选择、epoch 选择或 generator tail 选择。

固定基线使用当前仓库中的三类权重：

- Artifact prior: `checkpoints/artifact_prior/artifact_prior_models.joblib`
- Semantic prior: `checkpoints/semantic_prior/semantic_probe.joblib`
- Residual prior: `checkpoints/residual_prior/checkpoint-1.pth`

当前 17-generator full target baseline 以实际 CSV 为准：

| metric | value |
| --- | ---: |
| mean_acc | 93.68 |
| mean_ap | 99.07 |
| mean_auc | 99.02 |
| mean_r_acc | 94.61 |
| mean_f_acc | 92.76 |

实验输出统一保存在 `results/experiments/<exp_name>/`，每个实验至少包含：

- `overall.csv`
- `per_generator.csv`
- `protocol.json`
- `component_scores/*.npz`，如果该实验涉及消融或组件门控学习

基础数据集固定使用当前仓库内的本地目录：

- 训练集：`dataset/train_100k/progan_train`
- current 17-generator 测试集：`dataset/AIGCDetectBenchmark_test`
- `dataset/` 下其余已迁移数据集用于 Phase 3 外部 benchmark 实验，不再从旧远端路径读取基础训练/测试数据。

## 1. 三类 Prior 必要性消融

### 1.1 目标

证明 artifact、semantic、residual 三类 prior 都不是装饰性模块，而是对不同错误模式有独立贡献：

- Artifact prior 应该提供最强 source-only 基础和 GAN 类稳定性。
- Semantic prior 应该提升 diffusion / commercial / text-to-image 的 fake-side recall。
- Residual prior 应该作为低权重修正项，补充上采样、重采样和 NPR-style 痕迹。

### 1.2 需要先导出的组件分数

对每张图导出以下分数：

| symbol | meaning |
| --- | --- |
| `W` | whole-image artifact score |
| `T` | native tile artifact score |
| `S` | CLIP semantic score |
| `R` | NPR residual score |
| `max_side` | 图像长边，用于 high-resolution guard |

建议新增脚本：

```bash
python scripts/export_component_scores.py \
  --config configs/apfreq_train100k_full.yaml \
  --target_root dataset/AIGCDetectBenchmark_test \
  --output_dir results/experiments/prior_ablation/components \
  --device cuda:0
```

这样所有消融都从同一份 component cache 计算，避免重复跑 CLIP 和 residual prior。

### 1.3 消融矩阵

消融分成两类：component-only 和 drop-one。component-only 说明每类 prior 独立能做到什么；drop-one 更直接证明完整系统中每类 prior 的边际贡献。

| id | variant | score formula | purpose |
| --- | --- | --- | --- |
| A0 | Whole artifact only | `W` | 测 whole-image artifact 底座 |
| A1 | Artifact only | `W + tile_delta(T,W)` | 测 tile artifact 对高分辨率局部伪影的贡献 |
| A2 | Semantic only | `S` | 测 CLIP source-only probe 的独立泛化 |
| A3 | Residual only | `R` | 测 NPR residual 的独立泛化 |
| A4 | No artifact | `S + gamma*R` | 直接验证 artifact 是否不可替代 |
| A5 | No semantic | `artifact + gamma*R` | 验证 semantic 对 diffusion tail 的贡献 |
| A6 | No residual | `artifact + semantic` | 验证 residual 修正项贡献 |
| A7 | No tile | `W + semantic + gamma*R` | 验证 native tile 是否必要 |
| A8 | Full FreqPRISM | 当前固定融合 | 主 baseline |

其中 artifact、semantic、residual 的 logit 融合都沿用当前实现中的 `logit_blend` 和 `combine_whole_tile_aux_signed_delta_guard_scores`。对于缺失分支，直接把对应项置零或使用该分支前的中间分数，不重新训练任何 prior。

### 1.4 报告指标

每个 variant 在每个 generator 上报告：

- `acc`, `ap`, `auc`
- `r_acc`, `f_acc`
- `fpr`, `fnr`

同时报告四个 group mean：

- 全 17 generator mean
- GAN / face / translation mean
- Diffusion / commercial / text-to-image mean
- 当前 tail mean：`wukong`, `stable_diffusion_v_1_4`, `stable_diffusion_v_1_5`, `Midjourney`, `gaugan`, `biggan`

为了避免融合权重设计掩盖排序能力，消融报告分两栏：

- `threshold=0.50`：严格复现当前部署点。
- `source_gamma_anchor`：使用第 2 节锁定的 source-only gamma anchor，最终判别阈值仍固定为 `0.50`，不使用 target labels。
- `pure_source_stress_main`：使用 source-only stress calibration 选出的 promoted compact fusion parameters，最终判别阈值仍固定为 `0.50`，不使用 target labels。Component gate threshold sweep 只作为 diagnostic / appendix。

### 1.5 统计检验

使用 paired bootstrap，对每个 generator 内的图像重采样 1000 次，报告 full model 相对 drop-one variant 的 95% CI：

- `Delta mean_acc`
- `Delta mean_ap`
- `Delta mean_auc`
- `Delta f_acc`，重点看 diffusion tail
- `Delta r_acc`，重点看 GauGAN / BigGAN real-side false positive

如果 full model 相比某个 drop-one variant 在 mean 或关键 tail 上没有稳定优势，该 prior 的必要性结论需要降级为“辅助但非必要”。

### 1.6 Prior 融合权重敏感性与可学习权重消融

三类 prior 的有无消融只能回答“是否需要该 prior”。还需要补充一组融合权重消融，回答当前固定的 `beta/alpha/gamma` 是否合理，以及三类 prior 应该以多大强度进入最终 logit。所有实验只改变组件融合权重，不改变最终测试阈值：

```text
prediction = fake if final_score >= 0.50
```

当前主方法已从 source-gamma anchor 升级为 pure source-only stress-calibrated fusion parameters。先前 source-only anchor 为 `beta/alpha` 原始值加 `gamma=0.12`；promoted 主方法把 source stress calibration 选出的 compact scale 直接折叠进有效参数：

| parameter | promoted value | role |
| --- | --- | --- |
| `beta` | `0.25` | tile artifact 对 whole artifact 的正向增益 |
| `alpha_low_pos` | `0.30` | 低分辨率 semantic fake-side 正证据 |
| `alpha_low_neg` | `0.1875` | 低分辨率 semantic real-side 负证据 |
| `alpha_high_pos` | `0.40` | 高分辨率 semantic fake-side 正证据 |
| `alpha_high_neg` | `0.00` | 高分辨率且 tile 证据强时的 semantic 负证据 |
| `alpha_high_neg_guard` | `0.25` | 高分辨率但 tile 证据不强时的 semantic 负证据 |
| `gamma` | `0.21` | source-gamma anchor `0.12` 乘以 source-stress-selected `residual_scale=1.75` |

对应 compact scale 为 `tile_scale=1.25`、`semantic_pos_scale=2.0`、`semantic_neg_scale=1.25`、`residual_scale=1.75`。这组参数由 `source_gate_stress_only` 选择：在 source BA/AP/AUC、drift、flip-rate 和 real-source logloss 约束下，最小化 fake-side source logloss。current17、UniversalFakeDetect 和 Synthbuster 只用于最终报告。

实验分两层：

| level | experiment | definition | purpose |
| --- | --- | --- | --- |
| W0 | anchor weights | 当前 `beta/alpha/gamma` | 固定融合 baseline |
| W1 | compact 4-scale calibration | `tile_scale`, `semantic_pos_scale`, `semantic_neg_scale`, `residual_scale` | 低成本验证当前权重是否被 source-only protocol 重新选择 |
| W2 | full alpha-split calibration | 独立学习 `beta/gamma/alpha_low_pos/alpha_low_neg/alpha_high_pos/alpha_high_neg/alpha_high_neg_guard` | 主方法候选，直接回应固定融合权重是否人为指定 |
| W3 | one-factor sensitivity | 每次只扫一个权重或 scale | 解释哪个 prior 权重最敏感 |
| W4 | drop-weight sanity | `beta=0`, `gamma=0`, semantic pos/neg 分别置零 | 连接 prior ablation 与融合强度消融 |

针对审稿人可能攻击“`beta/alpha/gamma` 是 fixed 参数”的问题，主报告必须加入 single-factor parameter ablation。该实验不是在 target 上重新挑最优参数，而是验证当前 promoted main 的参数处在合理、稳定的区间内：

- 每次只改变一个参数，其余所有参数固定为 promoted main effective values。
- 三类 prior 权重、component score cache、最终阈值 `0.50`、generator split 和样本集合都不变。
- 每组参数消融单独形成一张结果表；表中必须包含 current value 行、zero/drop 行、低于 current 的若干行和高于 current 的若干行。
- 如果某个参数的最佳 target diagnostic 值不等于 current value，也不能据此回调主方法；只能在讨论中解释为 sensitivity evidence。主方法仍由 source-only protocol 锁定。

统一结果表头：

| column | meaning |
| --- | --- |
| `value` | 当前消融中被改变的 scale 或 direct parameter value |
| `effective_weight` | 代入最终 logit 的实际 `beta/alpha/gamma` 值 |
| `source_worst_ba` | source_gate diagnostic worst-group BA |
| `source_flip_rate` | 相对 promoted main 的 source prediction flip rate |
| `mean_acc / mean_ap / mean_auc` | current 17-generator locked diagnostic |
| `tail_f_acc` | `wukong`, `stable_diffusion_v_1_4`, `stable_diffusion_v_1_5`, `Midjourney`, `gaugan`, `biggan` fake recall mean |
| `gaugan_r_acc / biggan_r_acc` | real-side false-positive 风险监控 |
| `Delta mean_acc / Delta tail_f_acc` | 相对 current value 行的差值 |
| `95% CI` | paired bootstrap 置信区间，主文至少给 `Delta mean_acc` 和 `Delta tail_f_acc` |

#### 1.6.1 `beta` 单因素消融表

`beta` 控制 native tile artifact 对 whole-image artifact 的增益。该表只改变 `beta`，固定全部 `alpha_*` 和 `gamma`：

```text
final_logit =
  w
  + beta * max(0, t - w)
  + semantic_term(current alpha_*)
  + current gamma * r
```

| paper table | varied parameter | fixed parameters | scale grid | effective `beta` grid | interpretation |
| --- | --- | --- | --- | --- | --- |
| Table 4a | `beta_scale` | all `alpha_*`, `gamma` | `0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50` | `0.0000, 0.0625, 0.1250, 0.1875, 0.2500, 0.3125, 0.3750` | 证明 tile artifact 增益不是任意设定；`0.00` 等价于去掉 tile 增益 |

主文希望看到的结论不是“`beta=0.25` 在 target 上绝对最优”，而是 `beta=0.25` 附近的 mean Acc/AP/AUC、tail fake recall 和 GauGAN/BigGAN real accuracy 形成平台区间，且 `beta=0` 有可解释退化。

#### 1.6.2 `alpha` 单因素消融表

当前实现中的 `alpha` 不是单一标量，而是按 resolution 和 semantic logit 正负号拆成多个 effective 参数。为了避免审稿人认为只扫一个 global `alpha` 掩盖问题，报告分两层：

1. 主文放 compact `alpha` 表，分别改变 semantic positive scale 和 semantic negative scale。
2. Appendix 放 alpha-split 表，对每个 `alpha_*` 做单因素消融。

Compact `alpha` 表：

| paper table | varied parameter | changed effective weights | fixed parameters | scale grid | purpose |
| --- | --- | --- | --- | --- | --- |
| Table 4b | `semantic_pos_scale` | `alpha_low_pos`, `alpha_high_pos` | `beta`, negative `alpha_*`, `gamma` | `0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50` | 验证 semantic fake-side 正证据是否必要、是否过强 |
| Table 4c | `semantic_neg_scale` | `alpha_low_neg`, `alpha_high_neg_guard` | `beta`, positive `alpha_*`, `gamma` | `0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50` | 验证 semantic real-side 负证据对 FPR 控制是否必要 |

Alpha-split appendix 表：

| appendix table | varied parameter | value grid | fixed parameters | purpose |
| --- | --- | --- | --- | --- |
| Table S4d | `alpha_low_pos_scale` | `0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50` | other `alpha_*`, `beta`, `gamma` | 低分辨率图像 semantic fake-side 正证据 |
| Table S4e | `alpha_low_neg_scale` | `0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50` | other `alpha_*`, `beta`, `gamma` | 低分辨率图像 semantic real-side 负证据 |
| Table S4f | `alpha_high_pos_scale` | `0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50` | other `alpha_*`, `beta`, `gamma` | 高分辨率图像 semantic fake-side 正证据 |
| Table S4g | `alpha_high_neg_guard_scale` | `0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50` | other `alpha_*`, `beta`, `gamma` | 高分辨率 guard 下 semantic real-side 负证据 |
| Table S4h | `alpha_high_neg` direct | `0.00, 0.02, 0.05, 0.10, 0.15, 0.20` | other `alpha_*`, `beta`, `gamma` | 因 current value 为 `0.00`，不能用 scale sweep，只能 direct sweep |

`alpha` 表重点解释 fake-side recall 与 real-side FPR 的 trade-off。若 `semantic_pos_scale=0` 或 `semantic_neg_scale=0` 出现明显退化，可以直接支撑 semantic branch 不是装饰项；若高于 current value 后 real-side accuracy 下降，则说明当前 fixed alpha 没有过度放大 CLIP semantic evidence。

#### 1.6.3 `gamma` 单因素消融表

`gamma` 控制 residual prior 作为修正项进入最终 logit 的强度。该表只改变 `gamma`，固定 `beta` 和全部 `alpha_*`：

```text
final_logit =
  current artifact_term
  + current semantic_term
  + gamma * r
```

| paper table | varied parameter | fixed parameters | scale grid | effective `gamma` grid | interpretation |
| --- | --- | --- | --- | --- | --- |
| Table 4d | `gamma_scale` | `beta`, all `alpha_*` | `0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00` | `0.0000, 0.0525, 0.1050, 0.1575, 0.2100, 0.2625, 0.3150, 0.3675, 0.4200` | 证明 residual 是低权重修正项；`0.00` 等价于 no residual |

`gamma` 表的重点是证明 residual prior 不能太弱也不能太强：`gamma=0` 应在 diffusion tail fake recall 上退化；过高 `gamma` 若引入 real-side FPR 或 mean Acc 下降，则说明当前 `0.21` 是 source-stress 约束下的稳健折中。

推荐 one-factor sweep 与现有实现的对应关系：

| sweep | varied weight | values | fixed terms | purpose |
| --- | --- | --- | --- | --- |
| B0 | `beta_scale` | `0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50` | other weights fixed at promoted main | 生成 Table 4a |
| A0 | `semantic_pos_scale` | `0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50` | other weights fixed at promoted main | 生成 Table 4b |
| A1 | `semantic_neg_scale` | `0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50` | other weights fixed at promoted main | 生成 Table 4c |
| R0 | `gamma_scale` | `0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00` | other weights fixed at promoted main | 生成 Table 4d |
| AS | alpha split | each `alpha_*` 单独按 scale 扫描；`alpha_high_neg` 用 direct grid | other weights fixed at promoted main | 生成 Table S4d-S4h |

报告方式：

- 在 `source_gate` 上报告 selection-side 指标：worst-group BA、overall BA、score drift、flip rate、anchor distance。
- 在 current 17-generator 上报告 locked diagnostic 指标：mean Acc/AP/AUC、GAN mean、diffusion mean、tail mean。
- current 17-generator 只用于 locked diagnostic / final report，不能反过来选择 `beta/alpha/gamma`。当前 promoted 主方法的参数选择记录为 `target_labels_used_for_selection=false`。
- 主文放 Table 4a-4d：`beta`、compact `alpha_pos`、compact `alpha_neg`、`gamma` 单因素表。Appendix 放 Table S4d-S4h：所有 alpha-split 参数的单因素表。
- 每张表都以 promoted current value 作为 reference row，并报告相对该行的 delta 和 bootstrap CI。

输出文件：

```text
results/experiments/phase2_fusion_weight_sensitivity/
  source_gate_weight_sweep.csv
  current17_weight_sweep_overall.csv
  current17_weight_sweep_per_generator.csv
  current17_weight_sweep_group_slices.csv
  paper_tables/
    table4a_beta_ablation.csv
    table4b_alpha_pos_ablation.csv
    table4c_alpha_neg_ablation.csv
    table4d_gamma_ablation.csv
    tableS4d_alpha_low_pos_ablation.csv
    tableS4e_alpha_low_neg_ablation.csv
    tableS4f_alpha_high_pos_ablation.csv
    tableS4g_alpha_high_neg_guard_ablation.csv
    tableS4h_alpha_high_neg_direct_ablation.csv
  protocol.json
```

复跑命令：

```bash
python scripts/run_fusion_weight_sensitivity.py \
  --source_component_dir results/experiments/phase1w_source_weight_calibration/source_gate_components \
  --current_component_dir results/experiments/phase2_prior_ablation/current17_components \
  --output_dir results/experiments/phase2_fusion_weight_sensitivity \
  --weights_json results/source_weight_selection/selection_protocol.json \
  --compact_sweep_values 0,0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0 \
  --alpha_split_sweep_values 0,0.25,0.5,0.75,1.0,1.25,1.5 \
  --alpha_high_neg_values 0,0.02,0.05,0.10,0.15,0.20
```

如果现有 CLI 一次性产出较宽的 compact grid，出论文表时按 Table 4a-4d 的参数范围过滤；额外行保留在 raw sweep CSV 中。

full alpha-split calibration 复跑命令：

```bash
python scripts/run_full_fusion_weight_calibration.py \
  --source_component_dir results/experiments/phase1w_source_weight_calibration/source_gate_components \
  --current_component_dir results/experiments/phase2_prior_ablation/current17_components \
  --output_dir results/experiments/phase1w_full_alpha_split_calibration \
  --selection_protocol_out results/main/full_fusion_weight_calibration/selection_protocol.json
```

### 1.7 Native Tile / Resolution 消融

FreqPRISM 的 artifact 分支不是简单把图像 resize 后打分，而是保留 native-resolution tile evidence。该消融用于证明 tile 与分辨率处理不是工程细节，而是对高分辨率伪影、局部频域异常和 diffusion tail 有贡献。

实验只改变 artifact tile 的取证方式，不改变最终阈值 `0.50`，不重新训练三类 prior。

| id | variant | definition | purpose |
| --- | --- | --- | --- |
| RZ0 | full native tile | 当前 `W + native T + S + R` | 主方法 |
| RZ1 | whole only | 只用 whole artifact `W`，去掉 `T` | 验证 tile 是否必要 |
| RZ2 | resized tile | 先 resize 到固定短边或 `256/512` 后再 tile | 验证 native resolution 是否必要 |
| RZ3 | center crop tile | 只取中心 crop 或中心 tile | 验证局部全覆盖是否必要 |
| RZ4 | tile mean aggregation | tile score 用 mean 聚合 | 对比当前聚合策略 |
| RZ5 | tile max aggregation | tile score 用 max 聚合 | 验证局部强伪影是否应被放大 |
| RZ6 | downsample before full pipeline | 全图先 downsample 到 `256/512/1024` 再跑完整流程 | 验证高分辨率信息损失 |

报告：

- current 17-generator mean Acc/AP/AUC。
- high-resolution generator 子集：`Midjourney`, `coco_sdxl_nw`, `stable_diffusion_v_1_5`, `wukong`。
- tail fake recall：`wukong`, `stable_diffusion_v_1_4`, `stable_diffusion_v_1_5`, `Midjourney`。
- real-side false positive：`GauGAN`, `BigGAN` 的 `r_acc/fpr`。

输出：

```text
results/experiments/phase2_tile_resolution_ablation/
  overall.csv
  per_generator.csv
  group_slices.csv
  protocol.json
```

主文目标结论：

```text
native tile preserves high-resolution artifact evidence that resize/crop variants lose.
```

### 1.8 Artifact Feature-Family 消融

Artifact prior 内部由多组显式统计特征构成。该消融用于回答：artifact prior 是否只是依赖 JPEG/block shortcut，还是多个频域、色度、纹理 family 都有贡献。

优先采用低成本 feature-family masking，不立刻重训 artifact prior：

```text
for each feature family f:
  replace f by source-real family median
  recompute artifact score W/T
  recompute final score
```

如果 masking 结果显示某些 family 极其关键，再补 appendix 版 retrain/drop-family 实验。

需要报告的 family：

| family | content | question |
| --- | --- | --- |
| `codec_block` | DCT/block/quantization 相关统计 | 是否只靠 codec/block artifact |
| `chroma_luma_coupling` | 色度-亮度耦合与残差 | 是否利用颜色通道异常 |
| `texture_artifact` | 局部纹理、边缘、高频统计 | 是否捕捉 GAN/diffusion texture |
| `recompression_*` | 多质量重压缩响应 | 是否依赖压缩稳定性差异 |
| `residual_spectrum` | residual FFT/radial band | 是否利用频谱异常 |
| `residual_tail_shape` | 残差分布尾部形状 | 是否利用 heavy-tail / over-smooth 现象 |
| `patch_spectrum_heterogeneity` | patch 间频谱异质性 | 是否捕捉局部生成不一致 |

实验变体：

| id | variant | definition |
| --- | --- | --- |
| AF0 | full artifact features | 当前 artifact prior |
| AF1 | no codec/block | mask `codec_block` |
| AF2 | no chroma-luma | mask `chroma_luma_coupling` |
| AF3 | no texture | mask `texture_artifact` |
| AF4 | no recompression | mask `recompression_*` |
| AF5 | no residual spectrum | mask `residual_spectrum` |
| AF6 | no residual tail | mask `residual_tail_shape` |
| AF7 | no patch heterogeneity | mask `patch_spectrum_heterogeneity` |
| AF8 | codec/block only | only `codec_block`, others median-masked |
| AF9 | spectrum only | only spectrum-related families |

报告：

- `Delta artifact_score_logit`
- `Delta final_score_logit`
- `Delta mean_acc/AP/AUC`
- per-generator `f_acc/r_acc`
- family contribution bar chart，用于可视化章节联动。

输出：

```text
results/experiments/phase2_artifact_family_ablation/
  overall.csv
  per_generator.csv
  family_contribution.csv
  protocol.json
```

主文目标结论：

```text
artifact prior is not a single codec shortcut; frequency, chroma, texture, and patch heterogeneity families contribute complementary evidence.
```

### 1.9 Residual / NPR-specific 消融

Residual prior 目前作为低权重修正项进入最终 logit。该消融用于把 residual prior 的作用和 NPR-style residual evidence 讲清楚，尤其说明它对 diffusion / upsampling / resampling 痕迹的补充作用。

实验仍保持最终阈值 `0.50`，不重新选择 target threshold。

| id | variant | definition | purpose |
| --- | --- | --- | --- |
| RP0 | full residual prior | 当前 `gamma * R` | 主方法 |
| RP1 | no residual | `gamma=0` | 与 drop-one 对齐 |
| RP2 | residual only | 只用 `R` | 测 residual 独立泛化 |
| RP3 | artifact + residual | `W/T + gamma*R`，去掉 semantic | 看 residual 与 artifact 互补 |
| RP4 | semantic + residual | `S + gamma*R`，去掉 artifact | 看 residual 与 semantic 互补 |
| RP5 | NPR energy only | 用 NPR residual energy / spectrum 的简单统计替代 residual model | 验证 residual model 是否超过手工能量 |
| RP6 | gamma sweep | `gamma_scale=0,0.5,1.0,1.5,2.0` | 验证 residual 是低权重修正还是强证据 |

重点分析：

- diffusion/text-to-image fake recall。
- `ADM`, `Glide`, `stable_diffusion_v_1_4`, `stable_diffusion_v_1_5`, `coco_sdxl_nw`, `wukong`。
- residual 与 artifact/semantic 冲突样本：`R` 高但 `W/S` 低，或 `R` 低但 `W/S` 高。
- failure case 中 NPR residual map 是否有可解释响应。

输出：

```text
results/experiments/phase2_residual_npr_ablation/
  overall.csv
  per_generator.csv
  conflict_slices.csv
  protocol.json
```

主文目标结论：

```text
residual prior is a low-weight but non-redundant correction signal, especially for upsampling/resampling artifacts in diffusion-style images.
```

## 2. 可学习融合权重：Source-Only Learnable Fusion Weights

### 2.1 问题

当前最终测试阈值 `0.50` 已经是部署协议的一部分，不再修改。需要学习的是三类 prior 之间的融合权重：tile artifact 该给 whole artifact 多大增益，semantic fake/real-side 证据该给多大权重，residual prior 作为修正项应有多强。

目标不是把最终 operating threshold 从 `0.50` 移走，而是在仍然使用：

```text
final prediction = fake if final_score >= 0.50
```

的前提下，把当前人工指定的 `beta/alpha/gamma` 改成低维、锚定、可追溯的 source-only 融合权重，减少“这些权重是否拍脑袋”的方法风险。当前保留两条记录：source-gamma anchor 用于简化对照，promoted main 使用 source_gate_stress_only 选择的 compact fusion scale。

### 2.2 融合公式

设：

```text
w = logit(W)   # whole artifact
t = logit(T)   # tile artifact
s = logit(S)   # semantic
r = logit(R)   # residual
```

当前 anchor 融合为：

```text
tile_delta = max(0, t - w)
semantic_term = alpha_pos * max(0, s) + alpha_neg * min(0, s)
final_logit_0 = w + beta * tile_delta + semantic_term + gamma * r
final_score_0 = sigmoid(final_logit_0)
prediction = fake if final_score_0 >= 0.50
```

学习版保持三类 prior、`tile_delta=max(0,t-w)`、semantic 正/负证据拆分和最终阈值不变，只学习融合权重：

```text
final_logit_w =
  w
  + beta' * max(0, t - w)
  + alpha_pos' * max(0, s)
  + alpha_neg' * min(0, s)
  + gamma' * r

final_score_w = sigmoid(final_logit_w)
prediction = fake if final_score_w >= 0.50
```

`alpha_pos' / alpha_neg'` 仍按 high-resolution guard 选择低分辨率或高分辨率参数。source-only protocol 先确认 target-label-free anchor；promoted main 再用 source stress calibration 从 source-only 候选中选择 compact scale。

### 2.3 参数化

source-only anchor 采用 full alpha-split 参数化，直接学习以下 7 个可解释权重：

| learned weight | anchor | parameterization |
| --- | ---: | --- |
| `beta'` | `0.20` | `beta0 * beta_scale` |
| `alpha_low_pos'` | `0.15` | `alpha_low_pos0 * alpha_low_pos_scale` |
| `alpha_low_neg'` | `0.15` | `alpha_low_neg0 * alpha_low_neg_scale` |
| `alpha_high_pos'` | `0.20` | `alpha_high_pos0 * alpha_high_pos_scale` |
| `alpha_high_neg'` | `0.00` | direct value grid，因为 anchor 为 0，scale 无法离开 0 |
| `alpha_high_neg_guard'` | `0.20` | `alpha_high_neg_guard0 * alpha_high_neg_guard_scale` |
| `gamma'` | `0.12` | source-selected `0.08 * 1.5` |

为控制试错成本，先跑 compact 4-scale diagnostic：

```text
tile_scale
semantic_pos_scale
semantic_neg_scale
residual_scale
```

当前已有 compact diagnostic 结果选择：

```text
tile_scale = 1.0
semantic_pos_scale = 1.0
semantic_neg_scale = 1.0
residual_scale = 1.0
```

这说明窄范围 source-only calibration 重新选择了 anchor 权重。当前 full alpha-split calibration 也已完成，使用 `source_gate` 组件缓存和 coordinate search，仍重新选择非 residual 的 anchor 权重：

```text
beta_scale = 1.0
alpha_low_pos_scale = 1.0
alpha_low_neg_scale = 1.0
alpha_high_pos_scale = 1.0
alpha_high_neg = 0.0
alpha_high_neg_guard_scale = 1.0
gamma_scale = 1.0  # full alpha-split coordinate search
```

随后扩展单因素 source gamma sweep 到 `2.0`。在只使用 source_gate、固定阈值 `0.50`、要求 worst-group BA 不低于 anchor 且 drift/flip 满足约束的规则下，选中 `gamma_scale=1.5`，即 effective `gamma=0.12`。这形成 source-gamma anchor。

promoted main 进一步采用 source-only stress calibration 选择的 compact candidate：

```text
tile_scale = 1.25
semantic_pos_scale = 2.0
semantic_neg_scale = 1.25
residual_scale = 1.75
```

因此折叠进 YAML 的有效参数为 `beta=0.25`、`alpha_low_pos=0.30`、`alpha_low_neg=0.1875`、`alpha_high_pos=0.40`、`alpha_high_neg_guard=0.25`、`gamma=0.21`。这组参数的 evidence chain 记录在 `results/main/pure_source_stress_calibration/selection_protocol.json`，并明确 `target_labels_used_for_selection=false`。

### 2.4 Source-only 数据划分

为了避免学到 ProGAN-only shortcut，不能直接在已经用于训练三类 prior 的同一批 source 图像上学习融合权重。采用 8/2 source split：

```text
dataset/train_100k/progan_train
  -> source_fit: 80% = 80000 total = 40000 real + 40000 fake
  -> source_gate: 20% = 20000 total = 10000 real + 10000 fake
```

当前低成本 protocol 使用现有 full-source 权重，只在 `source_gate` 的 component cache 上学习融合权重；该 diagnostic 可用于方法设计和主线决策。严谨版如果需要，则在 `source_fit` 上重新训练三类 prior，再在 `source_gate` 上重新学习权重。

### 2.5 学习目标与 Shortcut 抑制

使用低维 grid / coordinate / beam search，而不是训练高容量网络。候选空间限制在 anchor 附近：

```text
scale grid for nonzero anchors:
  {0.50, 0.75, 1.00, 1.25, 1.50}

alpha_high_neg direct grid:
  {0.00, 0.02, 0.05, 0.10, 0.15, 0.20}
```

推荐搜索流程：

1. 先跑 compact 4-scale grid，确认大方向是否需要增强或削弱 tile / semantic / residual。
2. 以 compact 结果为中心，跑 full alpha-split coordinate search。
3. 保留 top-K source candidates，再用 source leave-class-out / group-DRO 约束排序。
4. 锁定一个 source-only candidate 后才评估 current 17-generator 和外部 benchmark。
5. 当前 promoted main 采用 source stress calibration：source BA/AP/AUC、drift、flip-rate、real-source logloss 作为硬约束，在满足约束的候选中最小化 fake-side source logloss。

选择目标为：

```text
maximize worst_group_BA(source_gate)
  - lambda_drift * mean(abs(final_score_w - final_score_0))
  - lambda_flip * flip_rate(pred_w, pred_0)
  - lambda_anchor * distance(weights, anchor)
```

硬约束：

- final threshold 始终固定为 `0.50`。
- `source_gate` overall BA 相比 anchor 下降不超过 `0.2` 个百分点。
- clean source flip rate 不超过 `1%`。
- mean score drift 不超过 `0.01`。
- augmented source worst-group BA 不低于 anchor。
- current 17-generator 和外部 benchmark 只用于锁定后的最终报告，不能反过来选择任何权重。

group `g` 由以下因素交叉构成：

- source class / object category
- label: real / fake
- corruption view: clean, jpeg35, jpeg50, webp70, resize50, blur1
- resolution bin: short, medium, long side
- baseline confidence bin: low / medium / high
- component disagreement type: tile-only high, semantic-only high, residual-only high, all-agree

附加约束：

- 学习输入只能是 `W,T,S,R,max_side` 及其 logit/delta，不能输入 path、source class、generator id、图像原始像素。
- 超参只能通过 source leave-class-out validation 选择。
- 不允许看 target per-generator 表后再调 `beta/alpha/gamma`。

### 2.6 接受标准

可学习融合权重不是强行替换最终阈值，而是在固定 `0.50` 判别阈值下校准三类 prior 的相对强度。

报告方式：

| model | fusion policy | final threshold | selection data |
| --- | --- | --- | --- |
| FreqPRISM-anchor | 当前 `beta/alpha/gamma` | `0.50` | none |
| FreqPRISM-LFW compact | 4-scale source-calibrated weights | `0.50` | source_gate only |
| FreqPRISM-LFW full | full alpha-split source-calibrated weights | `0.50` | source_gate only |
| FreqPRISM-SSCF main | source-stress compact weights folded into YAML | `0.50` | source_gate_stress_only |

可接受目标：

- 在 source_gate 的 worst-group balanced accuracy 不低于 anchor。
- 在 current 17-generator locked evaluation 中，mean_acc 不应低于 anchor 超过 bootstrap CI 的 `0.5` 个百分点。
- diffusion tail 的 `f_acc` 应提升或持平。
- GauGAN / BigGAN 的 `r_acc` 不应出现明显塌陷。
- 当前 promoted 主方法写为 **pure source-only stress-calibrated fusion**，source-gamma anchor 作为简化 baseline。

Component trigger-threshold learning 不再进入代码主线或后续实验。当前保留 fixed 内部触发逻辑，后续只讨论 `beta/alpha/gamma` 这类融合权重的 source-only anchor 与 source stress-calibrated 主方法。

## 3. 外部 Benchmark 补充

### 3.1 选择标准

补充数据集需要覆盖三类缺口：

- 与当前 17-generator 重叠但真实图分布不同，用于验证是否只适配当前 benchmark。
- 新商业模型或更强 diffusion 模型，用于验证时间外泛化。
- real/fake 风格匹配更严格的数据集，用于减少“真实图和假图风格不一致”导致的伪提升。

### 3.2 P0 最终主表数据集

最终主表选择遵循“学术认可度 + 方法优势匹配 + 结果风险可控”的原则。FreqPRISM 的强项是 frequency/artifact、CLIP semantic、native-resolution tile 和 residual prior 的组合，因此主表优先选择能体现这些优势的数据集。

| dataset | venue/source | generators / content | why it matters |
| --- | --- | --- | --- |
| Synthbuster | OJSP dataset; used by recent CVPR work | DALL-E 2, DALL-E 3, Adobe Firefly, Midjourney v5, SD 1.3/1.4/2/XL, Glide; 1000 fake/model, RAISE-based prompts | 最推荐主表数据集。无 JPEG/resampling degradation，prompt 来自真实 RAISE 图像描述，商业/扩散模型覆盖好；artifact + CLIP semantic 都有发挥空间，结果风险相对可控 |
| UniversalFakeDetect official benchmark | CVPR 2023 | 19 model family benchmark; includes GAN and diffusion domains; official CLIP ViT-L/14 comparison protocol | 与 FreqPRISM 的 CLIP semantic prior 强相关，同时 FreqPRISM 还有 artifact/residual 分支，适合做“超过纯 CLIP detector”的主表 |
| GenImage | NeurIPS 2023 Datasets and Benchmarks | ImageNet classes; Midjourney, SD v1.4/v1.5, ADM, GLIDE, Wukong, VQDM, BigGAN | 百万级、顶会 D&B benchmark，市面上使用很多，必须进入主表。主指标同时报 AP/AUC 和 fixed `0.50` Acc；若 Wukong/SD tail 影响 Acc，放 per-generator 解释 |

### 3.3 P1 压力测试 / 附录数据集

| dataset | venue/source | use |
| --- | --- | --- |
| SPAI / Any-Resolution benchmark | CVPR 2025 | 与 FreqPRISM 的 native tile + frequency prior 定位高度匹配，但数据较新、实验成本和不确定性更高。建议作为 P1 扩展表，展示任意分辨率和频域线索优势 |
| Fake-Inversion benchmark | CVPR 2024 | 使用 reverse image search 匹配 real controls，专门减少 real/fake style mismatch。可信度强但难度和不确定性更高，建议放附录或 robustness table，不作为冲 SOTA 主表 |
| DIRE / DiffusionForensics | ICCV 2023 | 八类 diffusion model benchmark，适合专门验证 residual 与 frequency artifact 对 diffusion fake 的贡献 |
| ForenSynths / CNNDetection | CVPR 2020 lineage | 老牌 GAN benchmark，结果大概率好，但时代较早；可作为 appendix sanity check，不建议占主表篇幅 |

### 3.4 转换格式

Phase 3 直接使用当前 `dataset/` 下已经迁移好的外部数据目录；若后续新增 benchmark，再转换为同一结构：

```text
dataset/<benchmark>/<generator>/
  0_real/
  1_fake/
  manifest.csv
  protocol.json
```

当前本地可用的 Phase 3 roots：

- `dataset/synthbuster`
- `dataset/UniversalFakeDetect official benchmark`
- `dataset/CNNSpot_trainingdata`

`protocol.json` 至少记录：

- 原始下载 URL / DOI
- license
- 原始 generator 名称
- real 来源
- fake 来源
- 是否经过 resize、jpeg、webp、metadata strip
- 是否与当前训练源有重叠风险

评估命令应复用当前 `scripts/evaluate_target.py`，只切换 `--target_root` 和 `--output_dir`。

## 4. 可视化与可解释性实验

### 4.1 目标

可视化实验用于回答两个问题：

1. FreqPRISM 在真假图上到底看到了哪些 evidence。
2. artifact、semantic、residual 三类 prior 的作用是否互补，而不是重复看同一种 shortcut。

可视化不参与训练、权重校准或模型选择，只用于解释 Phase 1-W 锁定的 source-calibrated fusion detector 行为。输出统一保存在：

```text
results/experiments/visualization/
```

### 4.2 样本选择

每个可视化 panel 选成对样本：

- `true negative`: real 图且预测 real。
- `true positive`: fake 图且预测 fake。
- `false positive`: real 图但预测 fake。
- `false negative`: fake 图但预测 real。

每类至少选：

- ProGAN / StyleGAN2：展示 GAN 类强项。
- BigGAN / GauGAN：展示 real-side false positive 风险。
- SD v1.5 / Wukong / Midjourney：展示 diffusion/text-to-image tail。
- 最新模型 style probe：展示 2026 模型压力测试中的成功与失败案例。

每个 generator 至少输出 `4 x 4` panel：4 种结果类型，每种 4 张图。

### 4.3 Artifact Prior 可视化

Artifact prior 不是 CNN heatmap，而是显式统计特征，所以可视化重点放在“特征来源图”和“family contribution”。

对每张图输出：

| visualization | content | purpose |
| --- | --- | --- |
| RGB image | 原图 | 视觉参考 |
| residual map | `image - local_mean(image)` | 显示局部残差与纹理异常 |
| high-pass / edge map | Sobel / Laplacian 响应 | 显示边缘和高频伪影 |
| DCT block energy map | 8x8 block DCT low/mid/high energy | 显示 codec/block artifact |
| FFT spectrum | log magnitude spectrum + radial band profile | 显示全局频谱偏差 |
| chroma-luma map | chroma residual / luma residual | 显示色度-亮度耦合异常 |
| tile score grid | native 3x3 tile score heatmap | 显示局部高分辨率 artifact 触发区域 |

同时输出 artifact feature-family contribution bar chart。实现上不直接解释 HGB 内部树，而采用稳定的 counterfactual family masking：

```text
for each feature family f:
  replace f by source-real family median
  recompute artifact score
  contribution_f = logit(score_original) - logit(score_masked)
```

需要报告的 family 至少包括：

- `codec_block`
- `chroma_luma_coupling`
- `texture_artifact`
- `recompression_*`
- `residual_spectrum`
- `residual_tail_shape`
- `patch_spectrum_heterogeneity`

### 4.4 Semantic Prior 可视化

Semantic prior 使用 CLIP ViT-L/14 linear probe。可视化不做“伪 Grad-CAM”作为主证据，而采用更稳定的 embedding 和 patch occlusion：

| visualization | content | purpose |
| --- | --- | --- |
| CLIP score | semantic fake probability | 显示语义 prior 是否支持 fake |
| nearest neighbors | source real/fake CLIP nearest neighbors | 展示当前图在 source semantic space 靠近哪类 |
| PCA / UMAP | source real/fake + target samples embedding | 显示 fake 与 real 的语义分布位置 |
| semantic occlusion map | 遮挡 16x16 / 32x32 patch 后的 CLIP score delta | 显示哪些语义区域影响 fake score |

为了避免把 semantic prior 解释成低层 artifact，semantic occlusion 图只作为辅助，主图以 nearest neighbors 和 embedding projection 为主。

### 4.5 Residual Prior 可视化

Residual prior 输出 NPR-style 残差信号：

```text
NPR(x) = x - upsample_nearest(downsample_nearest(x))
```

对每张图输出：

| visualization | content | purpose |
| --- | --- | --- |
| NPR residual map | RGB residual 或 luma residual | 显示上采样/重采样痕迹 |
| residual spectrum | NPR residual 的 FFT spectrum | 显示残差频谱模式 |
| residual activation proxy | patch-level residual energy + model score delta | 显示 residual prior 主要响应区域 |

如果后续实现 Grad-CAM，只作为 supplement；主文优先使用 NPR residual map 和 patch score delta，因为它们更直接、可复现。

### 4.6 三类 Prior 互补性可视化

每张样本输出一个 unified evidence panel：

```text
original image
artifact maps: residual / DCT / FFT / chroma / tile grid
semantic maps: CLIP score / nearest neighbors / occlusion
residual maps: NPR residual / residual spectrum
component logits: W, T, S, R
fusion contribution: beta*tile_delta, semantic_term, gamma*residual
final score and prediction
```

核心图表：

- Component logit waterfall：展示 `W -> +tile -> +semantic -> +residual -> final`。
- Real vs fake component distribution：对每个 generator 画 `W,T,S,R,final` 的 violin/box plot。
- Failure taxonomy：把 FP/FN 按主导组件归因，例如 artifact-FP、semantic-FN、residual-conflict。
- Calibrated-weight effect：展示 anchor weights 和 source-calibrated weights 对同一张图的 component contribution 变化，但最终阈值仍固定 `0.50`。

### 4.7 论文图建议

主文放 3 张图：

1. **Figure: Prior Evidence Map**  
   展示同一 real/fake pair 的 artifact、semantic、residual evidence。

2. **Figure: Component Waterfall**  
   展示成功样本和失败样本中三个 prior 如何推动 final logit。

3. **Figure: Generator-Level Component Distributions**  
   展示 GAN strong set、diffusion tail、latest model style probe 的 `W,T,S,R` 分布差异。

附录放：

- 每个 generator 的 4x4 qualitative panel。
- FP/FN failure case atlas。
- Feature-family contribution bar charts。
- JPEG / WebP / metadata-stripped robustness visualizations。

### 4.8 接受标准

可视化实验应满足：

- 同一张图能同时看到 `W,T,S,R` 组件分数和对应 evidence 图。
- 至少能解释三个典型现象：GAN 类高准确、diffusion tail fake recall 较低、GauGAN/BigGAN real-side FP。
- calibrated weights 的作用必须表现为组件贡献变化，而不是最终阈值变化。
- 所有图都从 frozen detector 和 frozen component scores 生成，不参与任何模型选择。

## 5. 最新生成模型压力测试

### 5.1 目标

验证当前 detector 对 2026 年最新强生成模型是否仍有检测能力。目标是：

```text
latest-model style-probe mean_acc >= 75%
```

主测模型：

- OpenAI `gpt-image-2`，官方文档称其为当前 state-of-the-art image generation model，支持 text 和 image input。
- Google Nano Banana 2，即 `Gemini 3.1 Flash Image`，Google 官方称其为最新 state-of-the-art image model，并在 Gemini / Search / Ads / AI Studio / Vertex AI 中推出。

如果预算允许，可追加：

- Midjourney current production model
- Microsoft MAI-Image-2
- Seedream 5 系列或其他 2026 图像模型，但只有在找到官方 API / model card 且记录版本号后才纳入主表

追加模型只作为扩展，不影响 `gpt-image-2` 和 Nano Banana 2 的主结果。

### 5.2 数据构造

建立新数据根：

```text
dataset/latest_gen_style_probe/
  gpt_image_2_stylegan2_face/
    0_real/
    1_fake/
  gpt_image_2_biggan_object/
    0_real/
    1_fake/
  nano_banana_2_stylegan2_face/
    0_real/
    1_fake/
  ...
```

每个最新模型选 8 个代表风格，每个风格先做 pilot `n=25`，正式实验 `n=100`：

| style bucket | corresponding current generator family | prompt direction |
| --- | --- | --- |
| `stylegan2_face` | StyleGAN / StyleGAN2 / WhichFaceIsReal | photorealistic human portrait, studio or candid face |
| `biggan_object` | BigGAN | ImageNet-like object photo, centered object, natural background |
| `gaugan_scene` | GauGAN | landscape / outdoor scene, semantic-layout-like composition |
| `cyclegan_translation` | CycleGAN / StarGAN | translated domain look, season/weather/style transfer |
| `sd15_photoreal` | SD v1.4 / SD v1.5 | generic photoreal text-to-image |
| `midjourney_cinematic` | Midjourney | cinematic, high-detail, stylized commercial aesthetic |
| `dalle2_compositional` | DALL-E2 | compositional object interaction and clean prompt following |
| `wukong_chinese_visual` | Wukong | Chinese cultural visual elements, ink/fantasy/architecture |

真实图来源：

- 从当前 17-generator target 的对应 `0_real` 目录抽样。
- 每张真实图配一条 prompt。prompt 可以由 VLM 自动 caption 后人工清洗，或者人工模板化生成。
- 生成 fake 时使用该真实图作为参考图，prompt 明确要求“生成一张新的图，不复制原图像素，只保持主体、构图或语义内容”。

这比纯文本生成更严格，因为 real/fake 在内容上更接近，减少内容分布差异。

### 5.3 Prompt 模板

每条 prompt 包含三层：

```text
Content: describe the reference image content.
Style bucket: describe the target generator-family style without naming a living artist.
Constraint: create a new image, do not copy exact pixels, no watermark, no text unless requested.
```

示例：

```text
Use the reference image only as a high-level content and layout guide.
Create a new photorealistic image of the same kind of subject, with a highly polished cinematic text-to-image aesthetic, detailed lighting, shallow depth of field, and natural camera perspective.
Do not reproduce exact pixels, logos, watermarks, or visible text from the reference image.
```

对于 `wukong_chinese_visual`：

```text
Use the reference image as a semantic guide.
Create a new image with Chinese fantasy visual language, ink-painting inspired atmosphere, ornate architecture or landscape details when appropriate, and high-quality text-to-image rendering.
Do not copy exact pixels or add text.
```

### 5.4 评估设置

每个 latest model / style bucket 单独作为一个 generator group：

```bash
python scripts/evaluate_target.py \
  --target_root dataset/latest_gen_style_probe \
  --output_dir results/experiments/latest_gen_style_probe/fixed \
  --config_name apfreq_train100k_full.yaml \
  --device cuda:0 \
  --per_label 0
```

默认跑 Phase 1-W 锁定的 source-calibrated fusion weights 主方法，最终阈值仍为 `0.50`。

报告：

- overall mean acc / ap / auc
- per latest model mean
- per style bucket mean
- fake recall，也就是 `f_acc`
- real false positive，也就是 `fpr`

通过标准：

- 主目标：latest-model style-probe mean_acc >= 75%。
- 更严格目标：每个最新模型 mean_acc >= 75%，每个 style bucket mean_acc >= 65%。
- 如果 AP/AUC 高但 acc 低，说明分数分布集中在最终 `0.50` 附近；优先分析融合权重是否能把有效证据推离决策边界。
- 如果 AP/AUC 也低，说明 detector 的排序信号对最新模型失效，需要考虑新 prior 或新 source training。

### 5.5 文件元数据说明

FreqPRISM 当前只通过 PIL/torch 读取图像像素，不读取 EXIF、C2PA、文件来源字段或其他 metadata。因此最新模型实验第一版不单独做 `metadata_preserved` / `metadata_stripped` 分支，避免增加不必要的实验成本。

论文中只需在 protocol 里明确：

- detector 输入是 RGB pixel tensor。
- scoring pipeline 不读取文件 metadata。
- 所有结果不依赖 EXIF/C2PA/provenance 字段。

如果审稿或复现实验要求进一步排除平台 provenance 影响，再追加 metadata stripping 和 JPEG/WebP recompression 作为 appendix robustness，而不作为主实验前置步骤。

## 6. 推荐执行顺序

执行顺序已经从“source-only learnable fusion weights 是否足够”推进到“source stress calibration 是否能选择更强 compact fusion”。当前主方法为 **FreqPRISM pure source-only stress-calibrated fusion**：source_gate_stress_only 负责选择 compact fusion scale，current17 与外部 benchmark 只做最终报告，最终阈值仍固定为 `0.50`。

早期 source-only compact 4-scale calibration 重新选择 anchor：

```text
tile_scale = 1.0
semantic_pos_scale = 1.0
semantic_neg_scale = 1.0
residual_scale = 1.0
```

随后扩展 source-only gamma sweep 到 `2.0` 后，按 source selection 规则选中 `gamma_scale=1.5`，effective `gamma=0.12`。这形成 target-label-free source-gamma anchor。新的 promoted 主方法进一步采用 source stress calibration 选择的 compact scales：

```text
tile_scale = 1.25
semantic_pos_scale = 2.0
semantic_neg_scale = 1.25
residual_scale = 1.75
```

折叠后的有效参数为 `beta=0.25`、`alpha_low_pos=0.30`、`alpha_low_neg=0.1875`、`alpha_high_pos=0.40`、`alpha_high_neg_guard=0.25`、`gamma=0.21`。这一路线可以在论文中明确写成 pure source-only stress-calibrated，因为 protocol 记录 `target_labels_used_for_selection=false`。

### Phase 0: 组件分数缓存与低成本诊断

1. 实现 component score cache，至少导出 `W,T,S,R,max_side,final_fixed`，先覆盖 current 17-generator 和 source split。
2. 建立 8/2 source split：`source_fit=80k`，`source_gate=20k`。先不重训，用当前 full-source 权重做 diagnostic。
3. 在 `source_gate` 上导出 component scores，并先跑 compact 4-scale fusion calibration。
4. 用锁定的 compact weights 在 current 17-generator 上只做一次 sanity evaluation。这里不能反复调参，只判断 learnable fusion weights 是否有进入 full alpha-split 的价值。

Phase 0 的通过标准：

- source_gate worst-group BA 提升或持平。
- current 17-generator mean_acc 不低于 anchor 超过 `0.5` 个百分点。
- diffusion tail 的 `f_acc` 有提升或至少不下降。
- GauGAN / BigGAN 的 `r_acc` 不明显下降。

Phase 0 已完成 compact 4-scale calibration，并选择 anchor weights。由于 source_gate worst-group BA 持平、drift/flip 均为 0，fusion-weight 主线继续进入 full alpha-split 校准。

### Phase 1-W: Source-only anchor 与 source-stress 主方法

5. 在 `source_gate=20k` 的 component cache 上运行 full alpha-split calibration，学习 `beta/gamma/alpha_low_pos/alpha_low_neg/alpha_high_pos/alpha_high_neg/alpha_high_neg_guard`。
6. 锁定 `results/main/source_weight_calibration/selection_protocol.json` 或新目录 `results/main/full_fusion_weight_calibration/selection_protocol.json`，记录 search grid、objective、constraints、selected weights 和 `target_labels_used_for_selection=false`。
7. 在 current 17-generator 上只做一次 locked report：`anchor_weights` vs `source_calibrated_weights`。该结果只用于最终报告和 sanity audit，不参与 weight 选择。
8. Source-only calibration 结果作为 target-label-free anchor 保留。当前 promoted 主方法另行记录在 `results/main/pure_source_stress_calibration/selection_protocol.json`，其中 `target_labels_used_for_selection=false`。current17、UniversalFakeDetect 和 Synthbuster 只作为外部报告。

严谨版扩展：如果后续要证明非 anchor fusion weights 是否有额外价值，再在 `source_fit=80k` 上重训三类 prior 并重新学习 full alpha-split weights。这不改变当前主方法锁定协议。trigger-threshold 支线已停止，不再安排执行。

### Phase 2: Prior 消融与主结果表

12. 基于 Phase 1-W 锁定的主方法，跑 prior ablation：whole artifact only、artifact only、semantic only、residual only、no artifact、no semantic、no residual、no tile、full。
13. 先在 current 17-generator 上出主表和 drop-one 表，确认三类 prior 必要性。
14. 跑 fusion weight sensitivity：按 Table 4a-4d 分别生成 `beta`、compact `alpha_pos`、compact `alpha_neg`、`gamma` 单因素表；按 Table S4d-S4h 生成 alpha-split appendix 表；补充 drop-weight sanity variants。
15. 跑 native tile / resolution ablation，确认 native tile 和高分辨率处理的贡献。
16. 跑 artifact feature-family masking ablation，确认 artifact prior 不是单一 codec/block shortcut。
17. 跑 residual / NPR-specific ablation，确认 residual prior 是低权重但非冗余的修正信号。
18. 对 drop-one、weight sensitivity、tile/resolution 和 residual ablation 都做 bootstrap CI，避免只看均值导致结论不稳。

### Phase 3: 外部 Benchmark 主表

当前 promoted 主方法的 final-report 路径固定为：

```text
results/apfreq_full_target/
results/experiments/phase3_external_benchmarks/universalfakedetect_learned_gates/
results/experiments/phase3_external_benchmarks/synthbuster_learned_gates/
```

19. 优先使用已经迁移到 `dataset/` 下的数据集：Synthbuster、UniversalFakeDetect official benchmark、CNNSpot_trainingdata。
20. 按风险从低到高评估：Synthbuster -> UniversalFakeDetect official benchmark -> CNNSpot_trainingdata；GenImage 若后续迁入 `dataset/` 再加入主表。
21. 每个 benchmark 都跑 Phase 1-W 锁定的 source-calibrated fusion weights 主方法。
22. Fake-Inversion、DIRE、ForenSynths 放附录或 robustness，不影响主方法选择。

### Phase 4: 可视化与失败分析

23. 在 current 17-generator 和外部 benchmark 已有 component cache 上生成可视化，不额外跑模型。
24. 先做三类最有论文价值的图：prior evidence map、component waterfall、generator-level component distribution。
25. 再做 failure atlas：diffusion tail FN、GauGAN/BigGAN FP、latest model style-probe failure。

### Phase 5: 最新生成模型压力测试

26. 在前面主方法和可视化工具都稳定后，再构造 `latest_gen_style_probe` pilot，每模型每风格 25 张。
27. 如果 pilot mean_acc >= 75%，扩展到每模型每风格 100 张。
28. 如果 pilot 低于 75%，先用 Phase 4 的 component visualization 定位失败来源，再决定是否只作为压力测试报告，而不是继续扩大样本量。

## 7. 预期论文表格

### Table 1: Current 17-generator main result

固定阈值 `0.50`，与当前 baseline 对齐。

### Table 2: Prior ablation on current 17-generator

行是 A0-A8，列是 mean Acc/AP/AUC、GAN mean、diffusion mean、tail mean。

### Table 3: Drop-one prior analysis

只放 A4-A8，突出每类 prior 的边际贡献和 bootstrap CI。

### Table 4: Single-factor fusion parameter ablation

主文拆成四张单因素表，而不是一张混合 sweep 表：

- Table 4a: `beta_scale` sweep，固定全部 `alpha_*` 和 `gamma`。
- Table 4b: `semantic_pos_scale` sweep，固定 `beta`、negative `alpha_*` 和 `gamma`。
- Table 4c: `semantic_neg_scale` sweep，固定 `beta`、positive `alpha_*` 和 `gamma`。
- Table 4d: `gamma_scale` sweep，固定 `beta` 和全部 `alpha_*`。

每张表都报告 source_gate diagnostic、current 17-generator locked diagnostic、tail fake recall、GauGAN/BigGAN real accuracy、相对 current row 的 delta 和 bootstrap CI。完整 alpha-split 单参数表放 Table S4d-S4h。

### Table 5: Source-only calibrated fusion weights

比较 anchor weights 与 source-calibrated weights：

- source_gate worst-group BA
- selected `beta/alpha/gamma` 或 compact scales
- current 17-generator mean
- diffusion tail fake recall
- GauGAN / BigGAN real accuracy
- target labels used for selection: always `false`

### Table 6: Native tile / resolution ablation

比较 full native tile、whole-only、resized tile、center crop、不同 tile aggregation 和 downsample-before-pipeline。

### Table 7: Artifact feature-family ablation

报告 feature-family masking 后的 artifact logit drop、final logit drop 和 current 17-generator 指标变化。

### Table 8: Residual / NPR-specific ablation

报告 no residual、residual only、artifact+residual、semantic+residual、NPR energy only 和 gamma sweep。

### Table 9: External benchmark generalization

Synthbuster、UniversalFakeDetect、GenImage 分别报告 per benchmark mean 与 per generator mean。SPAI、Fake-Inversion 和 DIRE 放 robustness / appendix。

### Table 10: Latest generator style probe

`gpt-image-2` 与 Nano Banana 2 分模型、分 style bucket 报告 Phase 1-W source-calibrated fusion weights 主方法结果。

### Figure 1: Prior evidence visualization

真假图 pair 上展示 artifact / semantic / residual evidence maps。

### Figure 2: Component contribution waterfall

展示 `W -> +tile -> +semantic -> +residual -> final` 的 logit 贡献。

### Figure 3: Generator-level component distributions

展示 GAN strong set、diffusion tail、latest model style probe 的 `W,T,S,R` 分布差异。

## 8. 风险与备选方案

| risk | impact | mitigation |
| --- | --- | --- |
| calibrated weights 在 source 上好但 target 上差 | 说明 source-only fusion calibration 不稳定 | 回退到 anchor weights，并将 calibration 作为 negative experiment |
| full alpha-split 搜索空间过大 | source 上容易出现偶然最优 | 先跑 compact 4-scale，再用 coordinate / beam search；加入 anchor penalty、drift 和 flip-rate 约束 |
| external benchmark real/fake 风格差异太大 | 可能虚高 | 主表优先使用 Synthbuster 的 RAISE-based prompt 设计；Fake-Inversion 放附录验证 matched-control robustness |
| latest model 生成图过于干净，acc < 75% | 当前 detector 对 2026 模型失效 | 通过 component scores 定位失败 prior；考虑加入新 source-only synthetic proxy，但不能用 latest model 标签训练后再测同集 |
| API 生成成本高 | 样本量不足 | 先 pilot 25，再扩 100；优先两大模型和 8 个风格 |
| 平台水印或 metadata 被质疑 | 论文可信度降低 | protocol 明确 detector 只读 RGB pixel tensor；如被要求再补充 metadata stripping / recompression appendix |
