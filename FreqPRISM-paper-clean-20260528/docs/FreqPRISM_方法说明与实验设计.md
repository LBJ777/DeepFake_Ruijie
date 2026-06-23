# FreqPRISM 方法说明与实验设计

> 文档版本：v1.0 · 生成日期：2026-05-21  
> 对应代码：`networks/detector.py`, `scripts/train_detector.py`, `configs/apfreq_train100k_full.yaml`  
> 全称：**FreqPRISM — Frequency-domain Prior Integration with Semantic and Residual Modeling**

---

## 一、核心思想

`FreqPRISM` 是一个严格 source-only 的 AI 生成图像检测方法。它的目标不是重新训练一个更大的端到端 backbone，而是在 ProGAN-only 训练协议下，把三类互补证据组织成一个单一检测器：

- **Artifact prior**：从压缩、重采样、色度耦合、局部纹理和频谱统计中提取稳定的生成伪影线索；
- **Semantic prior**：用冻结 CLIP ViT-L/14 的语义表示训练 source-only 线性探针，补充仅靠低层伪影不足以覆盖的语义分布线索；
- **Residual prior**：用 NPR-style 残差信号建模上采样/重采样带来的生成痕迹；
- **Pure source-only stress-calibrated fusion**：用固定阈值和 source-only stress calibration 选择融合权重，不使用目标集标签做训练、调参、阈值选择或 epoch 选择。

核心假设是：

> **跨生成器泛化不能只依赖单一频域统计。更稳健的检测器应同时吸收图像级伪影、局部高分辨率 tile 伪影、语义先验和残差先验，但所有选择必须保持 source-only。**

因此，FreqPRISM 的重点不是“更复杂”，而是 **把多种 prior 以可审计、可复现、无目标标签泄漏的方式组合起来**。

---

## 二、整体架构

FreqPRISM 的推理流程可以概括为：

```text
输入图像
  │
  ├─ Whole-image artifact prior
  │    图像变体: clean / jpeg35 / jpeg50 / resize50 / blur1
  │    APSD feature extractor → codec expert + chroma expert
  │
  ├─ Native-tile artifact prior
  │    高分辨率图像切 256×256 native tiles
  │    tile score 聚合: top1
  │
  ├─ Semantic prior
  │    OpenAI CLIP ViT-L/14 frozen encoder
  │    source-only LogisticRegression linear probe
  │
  └─ Residual prior
       NPR-style residual ResNet
       x - nearest_down_up(x)

  └─ Signed logit fusion with high-resolution guard
       → final fake probability
       → threshold = 0.50
```

最终检测器输出一个 fake probability，标签约定为：

```text
real = 0
fake = 1
```

这仍然是一个 **single detector**：虽然内部有三个 prior 分支，但对外只暴露一个最终分数和一个固定阈值。

---

## 三、Artifact Prior

Artifact prior 是 FreqPRISM 的频域与伪影核心。它不直接训练深度分类器，而是先用 `ArtifactPriorFeatureExtractor` 提取显式统计特征，再训练轻量 source-only experts。

### 3.1 特征来源

默认输入尺寸为 `256 × 256`。特征包含多类显式统计：

| 特征族 | 作用 |
|--------|------|
| RGB / residual / high-pass views | 捕获基础亮度、边缘和残差信号 |
| DCT-local features | 捕获 JPEG block、局部频域能量分布 |
| FFT-global features | 捕获全局频谱能量与方向性 |
| recompression stability | 判断图像经过再压缩/重采样后的统计稳定性 |
| chroma-luma coupling | 捕获色度与亮度耦合异常 |
| texture / co-occurrence / tail-shape | 捕获局部纹理、残差分布尾部和共现模式 |
| codec block features | 捕获压缩块边界和 codec-like artifacts |

当前 artifact 模型训练协议记录的特征维度为：

```text
feature_dim = 849
```

### 3.2 两个 source-only expert

Artifact prior 训练两个专家：

1. **codec expert**  
   使用 `HistGradientBoostingClassifier`，主要关注 `codec_block` 特征族。

2. **chroma expert**  
   使用 `StandardScaler + LogisticRegression`，主要关注精细色度耦合特征。

两个 expert 的输出不是简单平均，而是先转成 logit 后组合：

```text
artifact_logit = logit(codec_score) + alpha * logit(chroma_score)
alpha = -0.40
```

这里 `alpha` 为负，表示 chroma expert 在当前固定组合中主要作为修正项，而不是无条件增强项。

### 3.3 训练与评估变体

artifact prior 的训练变体为：

```text
clean, jpeg50, jpeg50, resize50, blur1
```

评估变体为：

```text
clean, jpeg35, jpeg50, resize50, blur1
```

多个变体的分数使用 `mean_logit` 聚合。这样做的目的不是数据增强式投票，而是让 artifact prior 对常见压缩和重采样扰动保持稳定。

---

## 四、Semantic Prior

Semantic prior 使用冻结的 OpenAI CLIP ViT-L/14 提取图像语义特征，再训练一个 source-only 线性探针。

### 4.1 设计动机

仅依赖低层伪影时，模型容易在部分 diffusion / commercial generator 上出现 fake-side recall 不足。CLIP 语义表示提供的是另一种互补信号：

- 它不直接看 JPEG block 或局部残差；
- 它对图像内容、构图、对象组合和生成图常见语义模式更敏感；
- 通过线性探针限制复杂度，避免引入新的大规模目标域调参空间。

### 4.2 训练方式

当前配置为：

| 参数 | 值 |
|------|----|
| CLIP 模型 | `ViT-L/14` |
| 训练集 | `dataset/train_100k/progan_train` |
| train_per_label | `0`，即不截断 |
| linear C | `1.0` |
| 训练变体 | `clean` |
| target report during training | disabled |

训练得到：

```text
checkpoints/semantic_prior/semantic_probe.joblib
```

需要特别说明：当前 `training_protocol.json` 中的 `source_holdout` 字段只是训练流程中的诊断输出。由于 `train_per_label=0` 和 `holdout_per_label=0` 都表示“不截断、使用全部样本”，该字段不是独立 source holdout 指标，不能用于选择 CLIP 模型、`linear_c`、融合权重或阈值。

FreqPRISM 当前并不依赖这个字段做选择。semantic prior 的训练超参固定，目标标签也没有参与训练或选择。

---

## 五、Residual Prior

Residual prior 来自 NPR-style 残差检测思路。它关注图像经过 nearest down-up 后留下的残差信号：

```text
NPR(x) = x - upsample_nearest(downsample_nearest(x))
```

这个分支的作用是补充 artifact prior 对某些生成器不敏感的上采样/重采样痕迹。

### 5.1 模型结构

当前实现使用一个轻量化的 ResNet-style 模型：

```text
RGB image
  → NPR residual
  → Conv + BN + ReLU + MaxPool
  → ResNet layer1
  → ResNet layer2
  → AdaptiveAvgPool
  → Linear → fake logit
```

训练时使用 `BCEWithLogitsLoss`，推理时输出 `sigmoid(logit)` 作为 residual fake probability。

### 5.2 当前训练协议

| 参数 | 值 |
|------|----|
| 训练集 | `dataset/train_100k/progan_train` |
| epochs | `2` |
| selected checkpoint | `checkpoint-1.pth` |
| train image size | `256` |
| inference image size | `null`，即尽量使用 native resolution |
| max_samples_per_label | `0`，即不截断 |
| random_state | `100` |
| target_labels_used | `false` |

训练日志显示：

| epoch | train_loss |
|-------|------------|
| 0 | `0.1067` |
| 1 | `0.0174` |

`checkpoint-1.pth` 是配置中固定选择的 residual prior，不由目标集表现决定。

---

## 六、分数融合

FreqPRISM 的融合在 logit 空间进行。设：

```text
W = whole-image artifact score
T = tile artifact score
S = semantic score
R = residual score
```

首先计算 tile 对 whole score 的正向增益：

```text
delta_tile = max(0, logit(T) - logit(W))
```

这样 tile prior 只在局部证据比 whole-image 证据更强时提供增益，避免高分辨率图像中少数低质量 tile 无条件拉低整体判断。

随后引入 semantic prior 的 signed contribution：

```text
semantic_term =
  alpha_pos * max(0, logit(S)) +
  alpha_neg * min(0, logit(S))
```

其中 `alpha_pos / alpha_neg` 会根据图像是否高分辨率以及 tile 增益状态切换。当前主方法采用 pure source-only stress-calibrated fusion parameters：

| 参数 | 值 |
|------|----|
| tile_mode | `top1` |
| tile_size | `256` |
| tile_grid_size | `3` |
| beta | `0.25` |
| alpha_low_pos | `0.30` |
| alpha_low_neg | `0.1875` |
| alpha_high_pos | `0.40` |
| alpha_high_neg | `0.00` |
| alpha_high_neg_guard | `0.25` |
| high_res_threshold | `960.0` |
| tile_delta_threshold | `0.00` |
| gamma | `0.21` |
| threshold | `0.50` |

基础分数为：

```text
base = sigmoid(
  logit(W) + beta * delta_tile + semantic_term
)
```

最终再融合 residual prior：

```text
final = sigmoid(logit(base) + gamma * logit(R))
```

这里 `gamma=0.21` 来自 promoted compact scale：先由 source-gate gamma sweep
把 residual anchor 锁定到 `0.12`，再乘以 source-only stress calibration 选出的
`residual_scale=1.75`。这组参数只使用 source_gate component scores 和 source labels 选择；current17、UniversalFakeDetect 和 Synthbuster 只作为最终报告。

---

## 七、训练设计

FreqPRISM 的训练分为三个 stage：

```bash
python train.py --config configs/apfreq_train100k_full.yaml --stage all --device cuda:0
```

### 7.1 数据协议

训练根目录：

```text
dataset/train_100k/progan_train
```

训练集结构：

```text
root/<class>/0_real
root/<class>/1_fake
```

当前 full protocol 使用：

```text
50000 real
50000 fake
100000 total
```

默认配置中不设置 `max_sample` 或 per-label cap：

```text
artifact_prior.train_per_label = 0
semantic_prior.train_per_label = 0
residual_prior.max_samples_per_label = 0
```

### 7.2 三个训练阶段

| stage | 输出 | 是否使用目标标签 |
|-------|------|------------------|
| `artifact` | `checkpoints/artifact_prior/artifact_prior_models.joblib` | 否 |
| `semantic` | `checkpoints/semantic_prior/semantic_probe.joblib` | 否 |
| `residual` | `checkpoints/residual_prior/checkpoint-*.pth` | 否 |

semantic stage 虽然接收 `target_root` 参数，但默认通过 `--skip_target_report` 禁止训练时生成目标报告。因此目标集不参与模型训练，也不参与模型选择。

---

## 八、评估设计

正式评估命令：

```bash
python test.py --config configs/apfreq_train100k_full.yaml --device cuda:0
```

评估根目录：

```text
dataset/AIGCDetectBenchmark_test
```

当前配置：

```text
evaluation.per_label = 0
evaluation.full_target = true
```

这表示对每个目标生成器下的全部 real/fake 图像进行评估，不做每类截断。

### 8.1 目标生成器

当前 full target report 包含 17 个 generator：

```text
ADM
DALLE2
Glide
Midjourney
VQDM
biggan
coco_sdxl_nw
cyclegan
gaugan
progan
stable_diffusion_v_1_4
stable_diffusion_v_1_5
stargan
stylegan
stylegan2
whichfaceisreal
wukong
```

按大类可粗略分为：

| 组别 | 生成器 |
|------|--------|
| GAN / face / translation | ProGAN, StyleGAN, StyleGAN2, BigGAN, CycleGAN, StarGAN, GauGAN, WhichFaceIsReal |
| Diffusion / commercial / text-to-image | ADM, Glide, Midjourney, SD v1.4, SD v1.5, VQDM, Wukong, DALLE2, COCO-SDXL-NW |

### 8.2 指标

FreqPRISM 输出：

| 指标 | 含义 |
|------|------|
| `acc` | 固定阈值 `0.50` 下的整体准确率 |
| `ap` | Average Precision |
| `auc` | ROC-AUC |
| `r_acc` | real 类准确率 |
| `f_acc` | fake 类准确率 |
| `fpr` | real 被误判为 fake 的比例 |
| `fnr` | fake 被误判为 real 的比例 |

注意：`acc / r_acc / f_acc` 使用固定阈值 `0.50`，不是在目标集上搜索最佳阈值。

---

## 九、当前实验结果

主结果文件：

```text
results/apfreq_full_target/overall.csv
results/apfreq_full_target/per_generator.csv
results/apfreq_full_target/protocol.json
```

整体结果：

| 指标 | 值 |
|------|----|
| Mean Acc | `94.82` |
| Mean AP | `99.34` |
| Mean AUC | `99.30` |
| Mean r_acc | `94.69` |
| Mean f_acc | `94.95` |
| Mean FPR | `5.31` |
| Mean FNR | `5.05` |

分组结果：

| 组别 | Mean Acc | Mean AP | Mean AUC |
|------|----------|---------|----------|
| GAN / face / translation | `94.63` | `99.51` | `99.46` |
| Diffusion / commercial / text-to-image | `94.99` | `99.20` | `99.16` |

### 9.1 代表性强项

FreqPRISM 在多数 GAN 类和部分商业/扩散模型上表现稳定：

| generator | Acc | AP | r_acc | f_acc |
|-----------|-----|----|-------|-------|
| ProGAN | `99.99` | `100.00` | `99.98` | `100.00` |
| StarGAN | `99.95` | `100.00` | `100.00` | `99.90` |
| CycleGAN | `96.63` | `99.83` | `93.26` | `100.00` |
| DALLE2 | `98.15` | `99.86` | `97.50` | `98.80` |
| Glide | `98.38` | `99.85` | `98.23` | `98.53` |

这些结果说明 artifact + semantic + residual 的组合并没有只对 ProGAN 过拟合，在多个未见 generator 上仍保持很高排序能力。

### 9.2 当前尾部集合

当前主要薄弱点不是 AP 大幅崩塌，而是固定阈值下不同 generator 的错误方向不同：部分集合 real 侧误报较多，部分集合 fake 侧仍偏保守。

| generator | Acc | AP | r_acc | f_acc | 主要问题 |
|-----------|-----|----|-------|-------|----------|
| GauGAN | `82.83` | `99.51` | `65.66` | `100.00` | real 侧误报较多 |
| Wukong | `87.62` | `97.50` | `98.10` | `77.13` | fake 侧漏检较多 |
| BigGAN | `89.33` | `99.27` | `79.35` | `99.30` | real 侧误报较多 |
| WhichFaceIsReal | `90.20` | `97.45` | `90.10` | `90.30` | 两侧相对均衡但低于均值 |

这组现象很关键：这些集合的 AP 仍然较高，说明排序能力并不差；固定 `0.50` 阈值没有用 target labels 调整，因此尾部主要体现为 calibration / domain mismatch，而不是排序信号完全失效。

---

## 十、为什么 FreqPRISM 有效

FreqPRISM 的有效性来自三点组合。

### 10.1 Artifact prior 提供强 source-only 基础

APSD / codec / chroma / texture 统计直接针对图像生成与后处理痕迹，不依赖目标生成器标签。它提供了一个稳定的、可解释的底座。

### 10.2 Semantic prior 补足纯伪影统计的盲区

部分扩散模型的局部伪影弱，或者经过压缩后伪影不明显。CLIP 语义 prior 提供了更高层的分布线索，使检测器不完全受限于低层 artifact。

### 10.3 Residual prior 作为受控修正项

NPR residual 对上采样/重采样痕迹敏感。当前主方法先用 source-only sweep
得到 `gamma=0.12` anchor，再通过 source-only stress-calibrated scale 得到
`gamma=0.21`。它补充 artifact/semantic 分支，但最终判断仍由三类 prior 的 logit
融合共同决定。

### 10.4 Source-only 参数选择避免目标集泄漏

很多 detector 的风险不在模型本身，而在 target 上调阈值、挑权重、挑 epoch 或只报告有利 generator。FreqPRISM 当前把融合参数选择显式写入 source-only stress calibration protocol：

```text
target_labels_used_for_selection = false
selection_data = source_gate_stress_only
selection_protocol = results/main/pure_source_stress_calibration/selection_protocol.json
```

因此，论文中可以把当前主方法表述为 pure source-only stress-calibrated fusion。current17、UniversalFakeDetect 和 Synthbuster 只用于最终报告和外部诊断。

---

## 十一、与常见频域 detector 的定位差异

| 方法类型 | 主要信号 | 风险 | FreqPRISM 的处理 |
|----------|----------|------|------------------|
| 单一频域 CNN | 局部频谱 / residual | 可能对特定 generator 过拟合 | 加入显式 artifact expert 与 residual prior |
| 纯手工统计 + 传统分类器 | codec / texture statistics | 语义与高层模式不足 | 加入 CLIP semantic prior |
| 纯 CLIP/语义 detector | 高层语义分布 | 可能忽视真实成像伪影 | 保留 artifact prior 作为主线索 |
| target-tuned ensemble | 多模型融合 | 易发生目标标签选择泄漏 | 固定 source-only 融合和阈值 |

FreqPRISM 更像是一个 **可审计的 prior integration detector**，而不是黑盒 ensemble。它的三个 prior 都有明确角色，融合参数也写入配置和 protocol。

---

## 十二、当前局限与下一步方向

FreqPRISM 当前最主要的局限有三类。

### 12.1 固定阈值下的 fake-side recall 问题

Wukong、SD v1.4、SD v1.5、Midjourney 的 AP 仍高，但 `f_acc` 明显低于 `r_acc`。这说明排序空间里有可用信号，只是固定阈值更偏向保护 real-side。

后续如果要调整阈值，必须使用 source-only validation，不能用目标集标签直接选阈值。

### 12.2 部分 GAN real-side domain mismatch

GauGAN、BigGAN 的 real-side false positive 偏高，说明这些集合的真实图分布与 ProGAN source real 存在差异。这里更适合从 source real 分布覆盖、artifact calibration 或 high-resolution guard 角度改进。

### 12.3 当前 semantic 诊断字段命名不够严谨

`semantic_prior/training_protocol.json` 中的 `source_holdout` 当前不是独立 holdout。它不影响当前 demo 合规性，因为没有用于选择权重或阈值，但后续文档和代码最好改名为 `source_train_diagnostic` 或禁用该输出，避免误读。

---

## 十三、复现命令

训练全部 source-only components：

```bash
python train.py --config configs/apfreq_train100k_full.yaml --stage all --device cuda:0
```

完整目标评估：

```bash
python test.py --config configs/apfreq_train100k_full.yaml --device cuda:0
```

记录 source-only 权重选择协议：

```bash
python validate.py --config configs/apfreq_train100k_full.yaml
```

测试项目协议：

```bash
/data/lizihao/.conda/envs/aigc/bin/python -m pytest -q
```

---

## 十四、一句话总结

`FreqPRISM` 的核心价值在于：

> **在严格 source-only 协议下，把频域 artifact、CLIP semantic 和 NPR residual 三类互补 prior 组合成一个可审计的 single detector，以固定融合和完整目标评估获得稳定跨生成器泛化。**

它不是通过目标集调参取巧，也不是简单堆模型，而是把已有有效线索整理成一条合规、清晰、可复现的检测路线。
