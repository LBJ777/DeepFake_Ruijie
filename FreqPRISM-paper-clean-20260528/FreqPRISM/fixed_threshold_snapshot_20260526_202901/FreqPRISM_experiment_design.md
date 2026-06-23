# FreqPRISM 实验设计方案

日期：2026-05-26

## 0. 实验原则

FreqPRISM 当前定位是 strict source-only single detector。后续实验可以扩展评估集、补充消融和增加 source-only component-gate learning，但不能用目标集标签做训练、组件门控选择、融合权重选择、epoch 选择或 generator tail 选择。

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
  --target_root /data/lizihao/AIGC/AIGCDetectBenchmark-main/dataset/test/test \
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

为了避免组件门控设计掩盖排序能力，消融报告分两栏：

- `threshold=0.50`：严格复现当前部署点。
- `learned_component_gates`：使用第 2 节的 source-only 组件门控学习算法，但最终判别阈值仍固定为 `0.50`，不能用 target labels。

### 1.5 统计检验

使用 paired bootstrap，对每个 generator 内的图像重采样 1000 次，报告 full model 相对 drop-one variant 的 95% CI：

- `Delta mean_acc`
- `Delta mean_ap`
- `Delta mean_auc`
- `Delta f_acc`，重点看 diffusion tail
- `Delta r_acc`，重点看 GauGAN / BigGAN real-side false positive

如果 full model 相比某个 drop-one variant 在 mean 或关键 tail 上没有稳定优势，该 prior 的必要性结论需要降级为“辅助但非必要”。

## 2. 可学习组件间阈值：Group-Robust Source-Only Gate Learning

### 2.1 问题

当前最终测试阈值 `0.50` 已经是部署协议的一部分，不再修改。需要学习的是组件之间的门控阈值：tile evidence 什么时候允许增强 whole-image artifact，semantic prior 什么时候作为正/负证据介入，residual prior 什么时候作为修正项介入。

目标不是把最终 operating threshold 从 `0.50` 移走，而是在仍然使用：

```text
final prediction = fake if final_score >= 0.50
```

的前提下，让组件融合内部的触发条件更稳健，减少 ProGAN-only shortcut，并保持当前 17-generator baseline 的良好结果。

### 2.2 总体思路

冻结三类 prior 和已有融合权重，只学习少量 score-space gate thresholds。设：

```text
w = logit(W)   # whole artifact
t = logit(T)   # tile artifact
s = logit(S)   # semantic
r = logit(R)   # residual
```

当前固定融合可以写成：

```text
tile_delta_0 = max(0, t - w)
semantic_term_0 = alpha_pos * max(0, s) + alpha_neg * min(0, s)
final_logit_0 = w + beta * tile_delta_0 + semantic_term_0 + gamma * r
final_score_0 = sigmoid(final_logit_0)
```

学习版不改变最终阈值，也不新增 final bias，而是把内部触发点变成可学习门控：

```text
tile_delta = max(0, t - w - tau_tile)

semantic_pos = max(0, s - tau_sem_pos)
semantic_neg = min(0, s + tau_sem_neg)

residual_gate = max(g_min, sigmoid((abs(r) - tau_res_conf) / eta))

final_logit_gate =
  w
  + beta * tile_delta
  + alpha_pos * semantic_pos
  + alpha_neg * semantic_neg
  + gamma * residual_gate * r

final_score_gate = sigmoid(final_logit_gate)
prediction = fake if final_score_gate >= 0.50
```

其中学习参数只有：

- `tau_tile`：tile 比 whole 强多少时才提供增益。
- `tau_sem_pos`：semantic fake-side 正证据触发阈值。
- `tau_sem_neg`：semantic real-side 负证据触发阈值。
- `tau_res_conf`：residual 置信度达到多少时才完整介入。

固定参数仍保持当前配置：

- `beta`, `alpha_*`, `gamma` 不学习。
- `alpha_pos` / `alpha_neg` 仍按当前 high-resolution guard 规则选择，只改变 semantic logit 的触发阈值。
- `high_res_threshold=960` 不学习，避免把分辨率分布学成 shortcut。
- final decision threshold 固定 `0.50`。

为保证当前良好结果，所有门控从当前行为初始化：

- `tau_tile=0`
- `tau_sem_pos=0`
- `tau_sem_neg=0`
- `tau_res_conf=-inf`，即 residual 默认总是开启

如果 source-only 学习没有稳定收益，就回退到这组默认门控。

### 2.3 Source-only 数据划分

为了避免学到 ProGAN-only shortcut，不能直接在已经用于训练三类 prior 的同一批 source 图像上学习门控。采用 8/2 source split，而不是 9/1：

- 8/2 给 `source_gate` 留出 `20000` 张图，足够做 corruption view、resolution bin、confidence bin、component disagreement group 的 worst-group 估计。
- 9/1 只留 `10000` 张图，训练 prior 多出的 `10000` 张对当前全量 100k protocol 的边际收益小，但会让门控学习的 group 统计更不稳定。
- 门控参数只有 4 个低维 `tau_*`，更需要稳定 validation/gate 集，而不是尽可能大的 fit 集。

```text
dataset/train_100k/progan_train
  -> source_fit: 80% = 80000 total = 40000 real + 40000 fake
  -> source_gate: 20% = 20000 total = 10000 real + 10000 fake
```

最终严谨版 protocol：

1. 在 `source_fit` 上重新训练 artifact、semantic、residual prior。
2. 在 `source_gate` 上冻结模型并导出 `W,T,S,R,max_side` component scores。
3. 只用 `source_gate` 学习 `tau_tile,tau_sem_pos,tau_sem_neg,tau_res_conf`。
4. 只在锁定后评估 current 17 target、外部 benchmark、最新生成模型测试集。

如果训练成本太高，可以先跑 diagnostic 版：使用当前 full-source 权重，只在 source split 上学习门控。但 diagnostic 版不能作为论文主结论。

### 2.4 学习目标与 Shortcut 抑制

门控阈值使用低维 grid search 或 derivative-free search，而不是训练一个高容量网络。候选空间限制在当前门控附近：

```text
tau_tile      in {0.00, 0.25, 0.50, 0.75, 1.00}
tau_sem_pos   in {0.00, 0.25, 0.50, 0.75, 1.00}
tau_sem_neg   in {0.00, 0.25, 0.50, 0.75, 1.00}
tau_res_conf  in {-inf, 0.50, 1.00, 1.50, 2.00}
```

选择目标采用 conservative Group-DRO：

```text
maximize min_g BA_g(final_score_gate >= 0.50, y)
  - lambda_drift * mean(abs(final_score_gate - final_score_0))
  - lambda_flip * flip_rate(pred_gate, pred_0)
```

同时必须满足硬约束：

- final threshold 始终为 `0.50`。
- clean source overall BA 相比默认门控下降不超过 `0.2` 个百分点。
- clean source flip rate 不超过 `3%`。
- mean score drift 不超过 `0.03`。
- augmented source worst-group BA 不低于默认门控。

group `g` 由以下因素交叉构成：

- source class / object category
- label: real / fake
- corruption view: clean, jpeg35, jpeg50, webp70, resize50, blur1
- resolution bin: short, medium, long side
- baseline confidence bin: low / medium / high
- component disagreement type: tile-only high, semantic-only high, residual-only high, all-agree

如果多个候选满足硬约束，选择与默认门控最接近的一组：

```text
argmin mean(abs(final_score_gate - final_score_0))
```

这保证默认行为是强 fallback：学习门控只有在 source-only worst-group 稳健性提高且不明显改变原模型预测时才会被采用。

附加约束：

- 门控学习输入只能是 `W,T,S,R,max_side` 及其 logit/delta，不能输入 path、source class、generator id、图像原始像素。
- 超参只能通过 source leave-class-out validation 选择。
- 不允许看 target per-generator 表后再调任何 `tau_*`。

### 2.5 接受标准

学习门控不是强行替换最终阈值，而是在固定 `0.50` 判别阈值下改进组件间触发逻辑。

主报告方式：

| model | component-gate policy | final threshold | selection data |
| --- | --- | --- | --- |
| FreqPRISM-fixed-gates | 当前固定组件门控 | `0.50` | none |
| FreqPRISM-learned-gates | learned `tau_*` | `0.50` | source_gate only |

可接受目标：

- 在 source_gate 的 worst-group balanced accuracy 不低于固定门控。
- 在 current 17-generator locked evaluation 中，mean_acc 不应低于 fixed-gates 超过 bootstrap CI 的 `0.5` 个百分点。
- diffusion tail 的 `f_acc` 应提升或持平。
- GauGAN / BigGAN 的 `r_acc` 不应出现明显塌陷。

注意：current 17-generator 结果只能用于最终报告和 sanity audit，不能反过来选择 `tau_*`。如果 learned-gates 在 locked target 上不如 fixed-gates，论文主模型仍报告 fixed-gates，learned-gates 作为 negative/diagnostic experiment。

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

所有外部数据统一转换为当前项目格式：

```text
dataset_external/<benchmark>/<generator>/
  0_real/
  1_fake/
  manifest.csv
  protocol.json
```

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

可视化不参与训练、门控学习或模型选择，只用于解释固定 detector 和 learned-gates detector 的行为。输出统一保存在：

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
- Learned-gates effect：展示 fixed-gates 和 learned-gates 对同一张图的 component contribution 变化，但最终阈值仍固定 `0.50`。

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
- learned-gates 的作用必须表现为组件触发逻辑变化，而不是最终阈值变化。
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

同时跑两套组件门控：

- Fixed-gates: 当前固定组件门控，最终阈值 `0.50`
- Learned-gates: 第 2 节锁定的 source-only component gates，最终阈值仍为 `0.50`

报告：

- overall mean acc / ap / auc
- per latest model mean
- per style bucket mean
- fake recall，也就是 `f_acc`
- real false positive，也就是 `fpr`

通过标准：

- 主目标：latest-model style-probe mean_acc >= 75%。
- 更严格目标：每个最新模型 mean_acc >= 75%，每个 style bucket mean_acc >= 65%。
- 如果 AP/AUC 高但 acc 低，说明分数分布集中在最终 `0.50` 附近；优先分析 component gates 是否能把有效证据推离决策边界。
- 如果 AP/AUC 也低，说明 detector 的排序信号对最新模型失效，需要考虑新 prior 或新 source training。

### 5.5 文件元数据说明

FreqPRISM 当前只通过 PIL/torch 读取图像像素，不读取 EXIF、C2PA、文件来源字段或其他 metadata。因此最新模型实验第一版不单独做 `metadata_preserved` / `metadata_stripped` 分支，避免增加不必要的实验成本。

论文中只需在 protocol 里明确：

- detector 输入是 RGB pixel tensor。
- scoring pipeline 不读取文件 metadata。
- 所有结果不依赖 EXIF/C2PA/provenance 字段。

如果审稿或复现实验要求进一步排除平台 provenance 影响，再追加 metadata stripping 和 JPEG/WebP recompression 作为 appendix robustness，而不作为主实验前置步骤。

## 6. 推荐执行顺序

执行顺序要先回答“主方法到底是 fixed-gates 还是 learned-gates”。如果 learned-gates 成功，论文贡献可以写成 source-only learnable component-gated FreqPRISM；如果不成功，主方法仍保持 fixed-gates，learned-gates 作为 negative/diagnostic experiment。后续所有外部 benchmark、可视化和最新模型测试都以这个决策为前提。

### Phase 0: 组件分数缓存与低成本诊断

1. 实现 component score cache，至少导出 `W,T,S,R,max_side,final_fixed`，先覆盖 current 17-generator 和 source split。
2. 建立 8/2 source split：`source_fit=80k`，`source_gate=20k`。先不重训，用当前 full-source 权重做 diagnostic。
3. 在 `source_gate` 上学习 `tau_tile,tau_sem_pos,tau_sem_neg,tau_res_conf`，只做低维 grid search，并执行 drift / flip-rate / worst-group 约束。
4. 用锁定的 diagnostic gates 在 current 17-generator 上只做一次 sanity evaluation。这里不能反复调参，只判断 learned-gates 是否有进入严谨版的价值。

Phase 0 的通过标准：

- source_gate worst-group BA 提升或持平。
- current 17-generator mean_acc 不低于 fixed-gates 超过 `0.5` 个百分点。
- diffusion tail 的 `f_acc` 有提升或至少不下降。
- GauGAN / BigGAN 的 `r_acc` 不明显下降。

如果 Phase 0 不通过：停止 learned-gates 路线，后续主方法使用 fixed-gates；仍保留 diagnostic 结果说明我们验证过可学习门控但选择保守固定门控。

### Phase 1: 严谨版 Learned-Gates 主方法确认

5. 如果 Phase 0 通过，再在 `source_fit=80k` 上重新训练 artifact、semantic、residual prior。
6. 在 `source_gate=20k` 上重新导出 component scores，并重新学习 `tau_*`。
7. 锁定 learned-gates，写入 protocol，不再根据 target 结果修改。
8. 在 current 17-generator 上跑 fixed-gates vs learned-gates 对比，决定主方法命名：
   - learned-gates 优于或持平：主方法写为 **FreqPRISM-LG**，fixed-gates 作为 ablation。
   - learned-gates 不稳定：主方法写为 **FreqPRISM fixed source-only fusion**，learned-gates 进入附录。

### Phase 2: Prior 消融与主结果表

9. 基于 Phase 1 锁定的主方法，跑 prior ablation：whole artifact only、artifact only、semantic only、residual only、no artifact、no semantic、no residual、no tile、full。
10. 先在 current 17-generator 上出主表和 drop-one 表，确认三类 prior 必要性。
11. 做 bootstrap CI，避免只看均值导致结论不稳。

### Phase 3: 外部 Benchmark 主表

12. 优先下载和转换 Synthbuster、UniversalFakeDetect、GenImage。
13. 按风险从低到高评估：Synthbuster -> UniversalFakeDetect -> GenImage。
14. 每个 benchmark 都跑 fixed-gates 和 locked learned-gates，但主表只采用 Phase 1 决定的主方法。
15. Fake-Inversion、DIRE、ForenSynths 放附录或 robustness，不影响主方法选择。

### Phase 4: 可视化与失败分析

16. 在 current 17-generator 和外部 benchmark 已有 component cache 上生成可视化，不额外跑模型。
17. 先做三类最有论文价值的图：prior evidence map、component waterfall、generator-level component distribution。
18. 再做 failure atlas：diffusion tail FN、GauGAN/BigGAN FP、latest model style-probe failure。

### Phase 5: 最新生成模型压力测试

19. 在前面主方法和可视化工具都稳定后，再构造 `latest_gen_style_probe` pilot，每模型每风格 25 张。
20. 如果 pilot mean_acc >= 75%，扩展到每模型每风格 100 张。
21. 如果 pilot 低于 75%，先用 Phase 4 的 component visualization 定位失败来源，再决定是否只作为压力测试报告，而不是继续扩大样本量。

## 7. 预期论文表格

### Table 1: Current 17-generator main result

固定阈值 `0.50`，与当前 baseline 对齐。

### Table 2: Prior ablation on current 17-generator

行是 A0-A8，列是 mean Acc/AP/AUC、GAN mean、diffusion mean、tail mean。

### Table 3: Drop-one prior analysis

只放 A4-A8，突出每类 prior 的边际贡献和 bootstrap CI。

### Table 4: Source-only learned component gates

比较 fixed-gates 与 learned-gates：

- source_gate worst-group BA
- current 17-generator mean
- diffusion tail fake recall
- GauGAN / BigGAN real accuracy

### Table 5: External benchmark generalization

Synthbuster、UniversalFakeDetect、GenImage 分别报告 per benchmark mean 与 per generator mean。SPAI、Fake-Inversion 和 DIRE 放 robustness / appendix。

### Table 6: Latest generator style probe

`gpt-image-2` 与 Nano Banana 2 分模型、分 style bucket 报告 fixed-gates / learned-gates 结果。

### Figure 1: Prior evidence visualization

真假图 pair 上展示 artifact / semantic / residual evidence maps。

### Figure 2: Component contribution waterfall

展示 `W -> +tile -> +semantic -> +residual -> final` 的 logit 贡献。

### Figure 3: Generator-level component distributions

展示 GAN strong set、diffusion tail、latest model style probe 的 `W,T,S,R` 分布差异。

## 8. 风险与备选方案

| risk | impact | mitigation |
| --- | --- | --- |
| learned gates 在 source 上好但 target 上差 | 说明 source-only component gates 不稳定 | 保留 fixed-gates 为主模型，learned-gates 作为附加实验；收紧 `tau_*` 搜索范围 |
| external benchmark real/fake 风格差异太大 | 可能虚高 | 主表优先使用 Synthbuster 的 RAISE-based prompt 设计；Fake-Inversion 放附录验证 matched-control robustness |
| latest model 生成图过于干净，acc < 75% | 当前 detector 对 2026 模型失效 | 通过 component scores 定位失败 prior；考虑加入新 source-only synthetic proxy，但不能用 latest model 标签训练后再测同集 |
| API 生成成本高 | 样本量不足 | 先 pilot 25，再扩 100；优先两大模型和 8 个风格 |
| 平台水印或 metadata 被质疑 | 论文可信度降低 | protocol 明确 detector 只读 RGB pixel tensor；如被要求再补充 metadata stripping / recompression appendix |
