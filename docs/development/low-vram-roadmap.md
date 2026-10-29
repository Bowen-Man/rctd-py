# rctd-py 低显存 Streaming 与 GPU 优化路线图

> 版本：v0.4 整合版 · 更新：2026-10-28（Asia/Shanghai）  
> 开发分支：`perf/triton-optimization` · 硬件目标：96 GB RAM、RTX 5070 12 GB VRAM  
> 来源：用户提供的 v0.3 开发文档、当前源码与已完成的真实数据回测。  
> 当前决策：**收尾现有验证 → sigma 驻留审计与优化 → Doublet → Multi → 异步 → 自适应内存规划 → Triton。**  
> 本轮仅整理文档；下面的开发步骤留待下一轮执行，不代表已经实现、通过验收或已安排自动运行。

## 1. 交接结论：现在做到哪里

| 模块 | 已有结果 | 尚未完成的验收 |
|---|---|---|
| P0.1 Full 输入 streaming | 已实现按 batch 上传；已测快照在 pilot 和全量真实 IRWLS 中与原版逐元素完全一致；全量阶段显存显著下降 | 评测之后新增的 `torch.as_tensor` 改动未测；固定 G/K/B 的 N 扩展实验未完成 |
| P0.2 Normalize 数据放置 | CPU longdouble reduction、缩减大临时数组、CPU counts 入口已实现；全量阶段 RAM/VRAM 下降，时间缩短 | `bulk_weights`、`norm_profiles` 有微小差异，当前零容差比较失败；不能标记为严格数值验收完成 |
| P0.3 独立回测框架 | 固定真实数据、源码快照、分阶段与完整流程、RAM/VRAM、数值和标注指标、重复运行已落地 | 受控显存扩展实验、部分阶段内部观测、精度政策与差异归因仍待补齐 |
| P0.4 sigma | 已定位全量流程的显存瓶颈；源码确认全部 Q/SQ 表上 GPU，采样 spots 一次进入 IRWLS | 仅有初步定位，完整生命周期审计与优化尚未执行；用户确认这是下一块优先工作 |
| P1/P2 Doublet/Multi | 已发现完整 counts 驻留和 GPU 结果列表累积 | 尚未实现 spot-major streaming，也没有对应真实全量验收 |

**先区分“实现完成”“已测快照通过”“当前工作区通过”。三者不能互相替代。**

### 1.1 当前工作区比评测快照多了什么

最近一轮完整评测冻结的 `_full.py` 使用 `torch.tensor`；当前工作区将 profiles、Q/SQ、x、完整或分批 counts/nUMI 的构造改为 `torch.as_tensor`。这组新改动没有包含在下文回测结果中。

| 文件 | 已评测 SHA256 | 当前工作区 SHA256（本次整理时） |
|---|---|---|
| `_full.py` | `95c2fdeaa654ff432867485270260efc5d0eedc5e59f7d5aca77747bd18c6804` | `6f739f8ced72b5ab0757bd0041a0046ae3405cf64b49204c85486355f4f6ac78` |
| `_normalize.py` | `ca5791f8b23d8328ef6fcc8c0c6113396c2eb0dec4f0fab24ef9d17367f0384b` | 同左 |
| `_rctd.py` | `35ebca60c18b9d4ded392310322dc54e40f56963c66ab40ea4f8dc0c6885b5d0` | 同左 |

`as_tensor` 在满足条件的 CPU NumPy 输入上可共享内存；把 CPU 数据交给 CUDA 仍需要传输，不意味着 GPU 零拷贝，也不自动形成异步流水线。下一轮需验证共享/别名、dtype、device 和输入生命周期。另有一处注释 `if stream_gpu: # 不用流式就直接全上` 与分支方向不符，列为小范围清理项，本轮未改源码。

## 2. 目标、范围与不变量

目标是在 CPU RAM 足够的前提下，让大于 VRAM 容量的 RCTD 输入分块运行，同时保留统计语义，并用实测证明内存和吞吐的变化。

- CPU RAM 保存全量数据与最终结果；GPU 保存当前工作集和必要的固定状态。
- Full、Doublet、Multi 最终都应支持显存有界的执行路径。当前 `stream_gpu` 默认关闭，公开入口仅支持在 Full 中开启。
- 本阶段解决 RAM → GPU streaming；不把它描述为已经支持 SSD/H5AD 按需读取的 out-of-core 系统。
- gene selection、nUMI、平台归一化和 sigma 是全局准备过程；完成之后再切 spot chunk。不能对每块分别执行整条 RCTD 并重新校准。
- 保留数学公式、筛选阈值、采样规则、候选顺序、分类语义和精度逻辑；不以减少 `N_fit`、`N_epoch` 或固定 sigma 替代内存优化。
- 保留 resident 对照路径，单次只改变一个可以独立解释和回测的因素。

### 2.1 符号

| 符号 | 含义 |
|---|---|
| `N` | 当前模式的全部有效 spatial spots 数 |
| `G` | 当前阶段的基因数；区分 `G_raw → G_common → G_bulk → G_reg` |
| `K` | cell types 数 |
| `C` | Doublet/Multi 的 spot chunk size，控制输入驻留 |
| `B` | Full 或 task batch size，控制一次求解工作量；当前 Full 也用它控制输入上传粒度 |
| `S` | sigma 采样数，`min(N_fit, eligible_spots)` |
| `B_sigma` | 未来 sigma 求解分批参数，当前尚未实现；与 `N_fit` 分开 |
| `T` / `W` | 全部 sigma 表对数 / 同时驻留的工作窗口表对数 |

### 2.2 数据放置原则

1. **先 reduction，再传输。** 对立即归约的 `N×G` 输入，在 CPU 得到 `G` 向量或标量后送 GPU。
2. **batch compute 不等于输入 streaming。** 完整 counts 已在 GPU 时，缩小计算 batch 不会消除 `N×G` 驻留。
3. **固定状态也必须算大小。** 单个 sigma 的 Q/SQ 表对约 6.72 MiB，但全部 126 对约 846.56 MiB，不能统一称为“小表，全部常驻”。
4. **GPU 结果及时落 RAM。** 分批求解后仍把所有结果留在 GPU 列表，最终 `cat().cpu()`，同样会随总任务数增加显存。
5. **同步正确之后再异步。** 第一版先证明数据流、数值和生命周期正确，再引入 pinned buffers、streams、events。

## 3. 已完成回测：证据与边界

### 3.1 实验身份

基线提交：`233db3fbcb22287a0cae6b7b0e630a28981a05f1`。候选是保存 Normalize CPU 入口之后、后续 Full `as_tensor` 改动之前冻结的未提交源码。

```text
结果根目录：
/home/administrator/rctd_bench/evaluations/saved-normalize-20261027-v2

summary.md / summary.json                 汇总
pilot-baseline / pilot-current            小样本原始测量
pilot-comparison                         小样本逐元素比较
full-case                                固定全量数据、选择和基线校准
full-baseline / full-current              全量原始测量
full-comparison                          全量逐元素比较
frozen_current / frozen_current.json      候选源码与来源
harness / working.patch                  评估器快照与代码差异

候选源码集合 SHA256：
93cc4d9cbbffc99e98fc15d7f1a99c9a2f0d97e9e963d1deec4e3cf981744896
全量 case ID：
650866a98c402e36e46c5456a651b7515dca638376d416e2e16bbc1c0bddeb2a
```

条件：RTX 5070、FP64、`compile=false`、Full `batch_size=128`、4 CPU threads；每阶段独立进程，1 次预热 + 5 次测量；候选 Full 开 streaming，独立 Normalize 使用 CPU 输入。输入完整性检查和各版本自身重复性均通过，Full/完整流程报告的收敛比例均为 100%。

前一份不带 `-v2` 的实验因评估器 CPU 输入多复制一份 counts 而中止。修正为与实际调用一致的 `as_tensor` 后，基线与候选全部重跑。中止实验不参与本报告，不能混合不同评估器版本的性能结果。

### 3.2 固定真实数据

```text
/home/administrator/rctd_data/visiumhd_human_colon_cancer_03/
  st.h5ad
  sc.h5ad
  annotation.csv
```

- reference：260,506 cells，18,082 genes，`obs['Level2']` 提供 38 类标注。
- spatial：11,385 spots、18,085 genes；全量运行保留 11,326 spots，`G_bulk=7,827`、`G_reg=5,007`、`K=38`、sigma 为 0.41（内部整数 41）。
- pilot：固定 256 spots，`G_bulk=7,777`、`G_reg=4,983`、sigma 为 0.40。
- 两个 H5AD 的原始 gene names 有重复，固定案例使用唯一的 `gene_ids`。必须记录这一选择及其对按基因符号过滤规则的影响，后续比较不能悄悄更换 ID 或预处理。
- 原始文件、spot 选择、基因/细胞类型顺序、配置与基线校准均由 case/manifest 约束。单独 Full 使用冻结的全局校准输入，完整流程则从原始输入重新校准。

### 3.3 全量性能

时间为 5 次测量的中位数；RAM/VRAM 为评测报告中的峰值口径。单位 MiB/GiB 按 1024 进制。

| 阶段 | 时间 s：原版 → 候选 | RAM RSS GiB | GPU allocated MiB | GPU reserved MiB | 当前结论 |
|---|---:|---:|---:|---:|---|
| Normalize | 4.257 → 3.596 | 3.540 → 1.886 | 379.63 → 41.47 | 394 → 54 | 时间约降 15.5%，RAM/VRAM 明显下降；严格数值门槛未过 |
| Full | 118.369 → 120.729 | 2.299 → 2.268 | 679.17 → 251.76 | 886 → 474 | GPU allocated 约降 63%；没有得到加速证据；已测快照数值完全一致 |
| 完整流程 | 158.728 → 162.443 | 8.642 → 8.646 | 2111.14 → 2112.09 | 3588 → 3650 | 整体峰值未降；观察到约 2.34% 变慢；框架时间判定为 inconclusive |

Full 完整 FP64 counts 约 432.66 MiB，单个 128-spot batch 约 4.89 MiB，输入部分理论减少约 427.77 MiB，与实测阶段显存变化相符。剩余显存还包括 profiles、Q/SQ、IRWLS 的 `G×K²` 相关状态及 batch 工作区。

时间范围：Normalize 原版 4.06–4.49 s、候选 3.57–3.67 s；Full 118.02–120.58 / 119.11–122.90 s；完整流程 154.61–160.57 / 161.73–163.10 s。框架要求重复范围与超过 5% 的中位变化共同支持方向判断；这是描述性规则，不是统计显著性检验，也不代表小于 5% 的变化不存在。

pilot 的 Full 显存仅从 255.50 降到 250.60 MiB：256 spots 本来很小，省下的输入仅约一个 batch。其完整流程约 28.5 s、GPU allocated 约 1,276 MiB，两版没有明确性能差异。**pilot 与全量的 G、sigma 不同，不能替代固定 G/K/B 的 N 扩展实验。**

### 3.4 数值验收结果

本次比较采用 `atol=0, rtol=0`，不是宽松容差验收。

| 对象 | pilot 最大绝对差 | 全量最大绝对差 | 零容差结果 |
|---|---:|---:|---|
| 单独 Full weights | 0 | 0 | 通过 |
| Normalize bulk_weights | 1.6893e-8 | 4.5091e-8 | 未通过 |
| Normalize norm_profiles | 1.5477e-8 | 1.0578e-8 | 未通过 |
| 完整流程 weights | 9.5686e-9 | 2.6582e-7 | 未通过 |

spot IDs、cell type 顺序、pixel mask、基因列表、sigma 和已记录的收敛标志一致。连续值差异很小，但不能因此写成“精度无损失已证明”。

Normalize 差异应优先拆分核查：longdouble reduction 的实现、reduction 返回 dtype、与 FP64 profiles 组合运算时的 dtype 提升顺序、CPU/GPU 放置。此前同 FP64 输入的对照与混合 dtype 的真实路径表现不同，因此不能直接把误差归因于 longdouble 被取消。下一轮保留严格基线，把每个因素单独控制后再决定是否修复或制定容差。

### 3.5 与 annotation 的一致程度

全量 11,326 个输出中，10,427 个标注总量非零，899 个零标注行排除；无无效预测行。报告保留主导类型并列计数及逐类型指标。

| 完整流程指标 | 原版 | 候选 |
|---|---:|---:|
| RMSE | 0.0630591445 | 0.0630591450 |
| MAE | 0.0170396697 | 0.0170396698 |
| macro Pearson | 0.5578331033 | 0.5578331029 |
| JSD（base 2） | 0.1895997234 | 0.1895997247 |
| 主导类型一致率 | 76.349861% | 76.349861% |

这些值说明目前标注一致性指标几乎不变。`annotation.csv` 是细胞计数比例，而 RCTD 输出表示 RNA 贡献，二者不是同一种真值；上述一致率也不是独立验证的生物学准确率。

官方方法与评测依据参见 [RCTD 原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8606190/) 和 [官方 Figure 3 分析代码](https://github.com/dmcable/spacexr/blob/master/AnalysisPaper/MainFigures/figure3.Rmd)。框架已纳入有适用条件的 doublet 分类/类型识别指标，但本轮模式是 Full，没有 doublet 置信分类，这些指标应显示不适用。论文的模拟混合、真值、类型映射与筛选协议没有在当前真实数据上完整复刻；R 实现 parity 也尚未运行。

## 4. 为什么阶段变好，整条流程还没有变好

**时间近似相加，峰值内存取各阶段活跃内存的最大值。** 阶段优化是否改变全流程结果，取决于它是否命中当前瓶颈。

当前完整流程内部，Full 求解约 120–123 s，sigma 约 26 s，Normalize 约 4 s。Normalize 节省不到 1 s，Full streaming 的主要收益是容量；相同 batch 下分批传输还可能带来调度开销。因此不能从 Normalize 加速推出整条流程加速。

全流程约 2,112 MiB 的 GPU allocated 峰值出现在 sigma 路径。Full 的阶段峰值从约 679 降到 252 MiB，仍低于 sigma，整体峰值自然不随之下降。CPU 参考构建、数据准备等仍影响全流程 RAM，Normalize 的阶段 RSS 降低也不保证整体 RSS 峰值降低。

这些内部阶段时间与单独阶段回测的输入准备、调用边界不同；`platform_total` 还包含 Normalize 和 sigma，不能重复相加。各项中位数之和也不要求等于完整流程中位数。

## 5. P0.3：先把现有结果立住

### 5.1 已完成

- 独立框架位于 [benchmarks/regression](../../benchmarks/regression/README.md)，支持 prepare/run/compare、源码冻结、独立子进程、原始输出归档、数据完整性和重复性检查。
- 当前已测快照完成 pilot 和真实全量的 Normalize / Full / pipeline 对照。
- RAM 记录包含运行期每 10 ms 采样的进程 RSS，以及独立记录的 Linux 生命周期高水位 `ru_maxrss`；两者统计边界不同。
- CUDA 记录 allocated 与 reserved，使用同步后的时间和重置的峰值统计。它们不是 `nvidia-smi` 整卡占用。
- Full 的输入显存收益已有真实案例证据。

### 5.2 下一轮仍要完成

1. 检查当前未提交 diff，验证第 1.1 节新增的 `as_tensor` 改动，重新记录源码身份；不把旧报告自动继承给新源码。
2. 对 Normalize 的微小差异做最小归因；记录“严格等价”是否为必须条件。若使用容差，需在下一次验收前按输出/dtype 制定并说明理由，不能看到失败后自动放宽。
3. 固定 G/K/B、profiles、Q/SQ、sigma 与输入生成方式，增加 N；分别记录 allocated、reserved、RSS。预期 Full 输入驻留由 `O(NG)` 变为 `O(BG)`，不承诺总显存等于输入大小。
4. 保存可重现的阶段 checkpoint 和剩余问题。已有源码快照可以用作诊断基准，但不能标注为所有数值门槛均通过的发布版本。

现有 [test_full_gpu_streaming.py](../../tests/test_full_gpu_streaming.py) 的 3 个用例使用 mock IRWLS：验证 CPU/CUDA 的传输、batch 行为和输入分配，**不直接证明真实求解器数值一致**。真实 IRWLS 的证据来自本次独立回测。后续继续同时保留结构测试和真实求解对照。

## 6. P0.4：sigma 驻留审计与优化（下一块优先工作）

### 6.1 sigma 阶段在做什么

sigma 是 RCTD 似然模型中的额外变异参数。当前流程从满足 UMI 条件的 spots 中采样，在当前 sigma 下拟合权重，再评估附近 sigma 候选及缩放因子的似然，迭代选择 sigma。它在全局校准阶段运行，先于最终 Full/Doublet/Multi 求解。

当前配置 `N_fit=1000`、`N_epoch=8`；实际 `S=min(N_fit, eligible_spots)`。Full 的 `batch_size=128` 和 `stream_gpu` **没有传入这个采样 IRWLS 调用**。因此 Full 输入已经 streaming，不代表 sigma 已经分批。

### 6.2 已从源码确认的驻留问题

源码入口：[sigma](../../src/rctd/_sigma.py)、[全局调用](../../src/rctd/_rctd.py)、[IRWLS](../../src/rctd/_irwls.py)。以下是初步定位，尚不是完整的动态生命周期测量。

| 对象 | 当前形状/数量与放置 | 审计重点 |
|---|---|---|
| 全部 Q/SQ | `_sigma.py` 将两个完整字典转为 GPU tensors；126 表对 | 每对 Q/SQ 均为 `(1003,439)` FP64，合计约 846.56 MiB；生命周期跨整个 sigma 选择 |
| 采样 counts / nUMI | `S×G` / `S` 一次上 GPU | IRWLS 当前一次处理 S；与 Full batch 无关 |
| profiles / 求解固定状态 | `G×K`，另有 `G×K²` 级相关工作量 | 哪些跨 epoch 可复用，哪些每次重新构造，需实测 |
| prediction | `S×G` | 生成、clamp、flatten、子采样及下一 epoch 之间的存活范围 |
| 似然抽样及工作区 | 最多 1,000,000 个展平坐标；逐候选、逐缩放计算 | 子采样索引、派生数组与 `_calc_q_all_eager` 的临时张量 |
| 候选表窗口 | 当前 sigma 左右各 8 个索引，最多 17 个候选 | 是否只保留当前窗口或当前表对即可满足顺序计算 |

单独 Full 通常只需要最终 sigma 的一对 Q/SQ，约 6.72 MiB；不能用这个大小推断 sigma 阶段的表占用。

### 6.3 分成四个可独立验收的小步骤

**P0.4a — 只做观测和生命周期图。**

记录每个大张量的 shape、dtype、CPU/GPU、分配点、最后使用点、跨 epoch 复用、与其他张量重叠存活的情况。分别测表加载、当前 sigma 求解、预测/抽样、候选打分、释放之后的 allocated/reserved。明确全量 Q/SQ 的 CPU 构建成本；暂不把它与 GPU 缓存修改合并。

输出物：sigma 数据流表、分段峰值与时间、首个最小补丁的边界。观测代码归于独立评估/调试路径，防止计时方式或临时引用本身改变被测驻留。

**P0.4b — 先尝试 Q/SQ 有界 GPU 缓存。**

保留 CPU 全表作为来源，GPU 只加载当前所需表对或有限候选窗口。先通过审计选择最简单的同步策略，明确缓存容量与淘汰点。

- 保留原始表值、dtype、候选顺序与查表语义。
- 检查局部变量、字典及求解器引用，确保旧 GPU 表确实不再存活；仅删除某个字典键未必释放所有引用。
- 若同时保留最多 17 表对，表本体理论约 114.22 MiB；这是相对全部 126 对的容量估算，不是总峰值预测或已实现收益。
- 控制窗口移动与新旧表交接时的瞬时双份驻留。报告 H2D 次数、命中率/复用和时间代价，不能只看表数量。
- 先不同时改变 sigma 求解 batch、打分 reduction 或 CPU spline 预计算，便于归因。

**P0.4c — 再独立引入 sigma IRWLS 分批。**

保持相同的采样 spots、顺序、nUMI、profiles、初值和停止条件，用新的 `B_sigma` 控制当前求解工作集。每块结果按原顺序恢复，再执行与原版一致的 prediction 和候选评分。

需要特别验证 batch 形状改变带来的浮点差异、独立 spot 的收敛状态、输入是否仍全量驻留，以及块间是否隐藏重复的固定状态。第一次只解决求解工作区；若仍保存 `S×G` prediction，应如实记录剩余峰值，不将它称为整个 sigma 已完全 streaming。

**P0.4d — 最后根据测量处理 prediction / likelihood 工作区。**

保留 RNG 调用顺序、展平索引规则、最多一百万坐标的同一抽样、候选窗口、`0.8/0.9/1.0/1.1/1.2` 缩放及 argmin/tie-breaking。若分块累计分数会改变 reduction 顺序，单独评估其数值影响；不与表缓存补丁一起提交为“纯放置变化”。

### 6.4 sigma 验收门槛

- 最终 sigma 一致，同时检查逐 epoch 的候选、抽样身份和候选分数；只看最终 sigma 会漏掉中间偏移。
- IRWLS 权重/收敛、归一化输入和完整流程输出按预先设定的数值规则比较。
- 同时报告 sigma 阶段和完整流程的时间、RAM、allocated/reserved；确认峰值是否转移到其他阶段。
- 固定 G/K/S 测 GPU 表工作集/`B_sigma` 的影响，再按需要测试 S 的扩展；完整 N 与 S 的含义不可混淆。
- 不通过调低 `N_fit`/`N_epoch`、跳过自动 sigma 或更换采样数据获得优化结果。

**这块首先解决整体显存瓶颈。Full IRWLS 仍占大部分运行时间，sigma 省显存不意味着端到端必然加速。**

## 7. P1：Doublet spot-major streaming

sigma 优先之后，保留 v0.3 的 Doublet 主设计。当前源码初查确认：`_doublet.py` 全量创建 `Y_full_gpu/nUMI_full_gpu`，多个阶段把 GPU weights/scores 放进列表，最后拼接后才转 CPU。这里既有输入随 N 增长，也有结果随总 task 数增长的问题。

### 7.1 目标数据流

全局校准完成后，以真实 spot chunk 为最外层单位：

```text
全局 profiles / sigma / genes / nUMI 准备完毕
                    ↓
每个 C-spot chunk：counts/nUMI 上传一次
                    ↓
Full fit → candidate selection → pair scoring → singlet scoring
                    ↓
CPU classification → final two-type fit → 小结果写入 RAM
                    ↓
释放/复用 chunk，继续下一块
```

一个 spot 有 m 个候选类型时，可产生 `m(m-1)/2` 个 pair tasks 和 m 个 singlet tasks；m=8 时是 28+8 个候选任务。若逐 task 从 RAM 上传，同一基因向量会被反复搬运。spot-major 的目的就是保留 GPU 内复用，同时把驻留上限从 N 限制为 C。

### 7.2 P1.0 — 完整审计

对 Full、候选生成、pair、singlet、分类、最终分解六步，列出输入/输出、shape、dtype、device、索引、生命周期、重复传输和累积结果；形成可核对的 Doublet 数据流记录。当前只是源码定位，不把它标为审计完成。

### 7.3 P1.1 — 区分 C 与 B，复用 resident chunk

- `C=spot_chunk_size` 决定一次常驻多少真实 spots；`B=task_batch_size` 决定一次求解多少派生任务。
- Full solver 需要能消费已经在 GPU 上的 chunk。函数名可后定，关键是不在 Full 完成后丢掉 counts，再给 pair 阶段重新上传。
- global spot ID、chunk-local index 与最终输出位置必须有明确映射，保证 mask 恢复和排序不变。

### 7.4 P1.2 — chunk 内任务与结果生命周期

| 阶段 | 第一版执行策略 |
|---|---|
| Full / 候选 | chunk 的 Full 权重 `C×K` 转 CPU，沿用现有候选逻辑；counts 保留 GPU |
| Pair / singlet | 只生成当前 chunk 的 tasks，按 B 求解；使用 chunk-local 索引从 GPU counts gather |
| 分类 | 将当前 chunk 的小 scores 送 CPU，保持现有分类逻辑 |
| 最终拟合 | 复用尚未释放的 chunk counts，完成最终两类型分解 |
| 结果回收 | 每个 batch/chunk 的结果及时写入预分配 CPU 数组，GPU 不累计全量 task 输出 |

`Y_chunk_gpu[local_pix_idx]` 是 gather，会产生 `B×G` 工作张量，不能当作无额外内存的 view。每个 pair 的 profiles、如 `B×G×2` 的输入和 IRWLS/Hessian 工作区也应纳入预算。任务数量上界还受候选数影响，不能只计算 `C×G`。

### 7.5 P1.3/P1.4 — 正确性与扩展性

比较原版与 streaming 的 Full weights、候选集合和顺序、pair/singlet scores、分类标签、最终类型、最终权重和可观察的收敛状态。对临界阈值、并列候选、空任务、单任务、最后不足 C/B 的块以及不同 C/B 组合做针对性验证。

当前框架没有完整保存上述所有中间状态；先补独立 observer/调试输出，再宣称覆盖完成。不要仅用最终分类一致代替数值检查。

固定 G/K/C/B 和全局校准，增加 N，测量：

```text
M_GPU ≈ M_fixed + M_chunk(C,G) + M_tasks(B,G,K) + M_live_results(B)
```

目标是输入和结果 GPU 驻留都不再随全局 N 增长。CPU 输出数组随 N 增长是预期行为。边界重叠、临时副本和 allocator reserved 仍需实测。

## 8. P2—P5：后续阶段与进入条件

### P2 — Multi spot-major streaming

Doublet 稳定之后处理 Multi。当前 Multi 也有全量 GPU counts 和 GPU 结果列表。保留同一个 C/B 与全局校准原则，同时处理动态候选扩张、不同 spot 路径长度和不同任务量。不能把 Doublet 验收直接视为 Multi 已通过。

### P3 — 有限 pinned buffers 与异步流水线

同步 Full/Doublet/Multi 数据流稳定后，引入有限 CPU pinned staging buffers 和 GPU 双缓冲，不 pin 整份数据。用 events 明确 buffer 所有权、H2D 完成、计算完成和 D2H 完成，防止提前覆盖或释放。

目标是重叠 `H2D(i+1) / compute(i) / D2H(i-1)`。单独设置 `non_blocking=True` 不构成验收；需要 profiler 显示实际重叠，并证明额外缓冲未破坏显存上限。

### P4 — streaming-aware 自适应内存规划

基础容量估算应在选择 C/B 和开启双缓冲前就做；完整的自适应 planner 后置，不推迟必要的内存预算。

```text
M_peak 预算 = 固定状态（含 sigma 缓存）
            + spot chunk
            + task gather/profiles
            + IRWLS 临时工作区
            + 输出和双缓冲
            + 碎片与安全余量
```

根据可用显存选择 C、B、`B_sigma` 和缓冲数量，以实测校准估算。明确 OOM 回退是否可能影响批次形状、顺序或重复计算，保持结果可复现。

### P5 — Triton IRWLS 热点融合

先 profile 再决定 kernel：prediction、likelihood derivatives、gradient/Hessian 组装、mask/active-set 是候选。目标是减少 `B×G` 中间张量、HBM 流量和 launch，必要时做 gene tiling 与 partial reductions。

不预设替换成熟 GEMM、eigh 或线性求解实现就会更快。每个 kernel 分别检查 FP32/FP64、极端计数/预测值、收敛和完整流程结果。

## 9. 统一验收标准：现有能力与待补项

| 维度 | 现有证据/能力 | 后续补齐 |
|---|---|---|
| 可重现性 | 数据/源码指纹、固定选择、配置、环境、原始结果和同版本重复性 | 新改动重新记录身份；评估器变化时重跑两侧 |
| 结构一致 | spot/type/gene 顺序、mask、sigma、已保存的收敛标志 | sigma 轨迹、Doublet/Multi 候选及中间分类状态 |
| 连续值 | raw weights、bulk_weights、norm_profiles 的最大差、RMSE、超容差数量 | 各 dtype 的事先容差政策、分数/中间结果、临界输入 |
| 标注一致性 | RMSE/MAE/Pearson/JSD/主导类型及逐类型结果 | 官方模拟协议、适用的 doublet 真值、独立 R 对照 |
| 资源 | 阶段及 pipeline 时间、RSS、生命周期高水位、CUDA allocated/reserved | 固定 G/K 的 N/C/B/S 扩展，OOM 容量边界 |
| 吞吐与调度 | 总体时间及部分内部阶段计时 | tasks/s、传输字节/带宽、GPU 利用率、kernel 时间和实际 overlap |

数值比较保留原始未归一化权重，不通过 clipping 或行归一化隐藏差异；用于标注指标的比例转换单独定义，记录零行和无效行。结构/离散结果原则上精确一致，连续值规则必须事先明确。出现 NaN/Inf 不能用平均误差掩盖。

内存采样有边界：10 ms RSS 采样可能漏过瞬时峰值；生命周期高水位又可能来自加载阶段。allocated 是活跃 tensor 分配，reserved 还包括分配器缓存。不同口径不能相减后直接宣称“函数净占用”。

当前约 1.1 万 spots、约 2 GiB 流程 allocated 的结果，只证明这个案例上的变化；尚未证明在 12 GB 卡上跑通远大于 VRAM 的全部模式。需要受控压力测试或更大真实数据，以及无 OOM 的容量记录。

## 10. 下一轮的具体工作顺序

1. 保存 VS Code 文件，检查 `git status/diff` 和源码 hash，确认是否又有新增改动；保留用户已有暂存区。
2. 先验证尚未测过的 Full `as_tensor` 改动：现有传输/分配测试 + 冻结真实输入的 Full 对照。结果通过后单独标记新快照。
3. 收尾 Normalize 精度归因与 Full 受控扩展验证。把已验证能力和未通过门槛明确登记；纯只读 sigma 审计可以同步推进，不能将未过精度门槛的版本写成全部通过。
4. 完成 P0.4a sigma 生命周期审计，确定峰值组合，再只做 P0.4b 表缓存这一项最小改动。
5. 针对性测试与 pilot 对照通过后，运行 sigma/完整流程资源和数值对照；需要时再进入 P0.4c/d。不要在同一补丁里一起改 Doublet、异步和 Triton。
6. sigma 阶段的证据和限制记录完整后，进入 P1.0 Doublet 审计。

一次完整双版本、两种数据规模、1+5 重复的本轮评测约花费 66 分钟。开发期间先运行与改动相关的测试和 pilot；到有效 checkpoint 再跑完整套件。减少无关重复，不省略正式验收所需的两侧对照与重复测量。

### 10.1 回测命令模板（WSL，留待下一轮执行）

以下从仓库根目录执行，复用已经冻结的全量 case。使用同一评估器重跑两侧，以免跨评估器版本比较。输出目录必须是未使用的新目录；示例前缀仅供下一次命名，不表示已经创建或运行。

```bash
cd /home/administrator/Github/rctd-py

.venv/bin/python benchmarks/regression/bench.py run \
  --case /home/administrator/rctd_bench/evaluations/saved-normalize-20261027-v2/full-case \
  --source baseline \
  --out /home/administrator/rctd_bench/runs/next-checkpoint-baseline

.venv/bin/python benchmarks/regression/bench.py run \
  --case /home/administrator/rctd_bench/evaluations/saved-normalize-20261027-v2/full-case \
  --source working --stream-gpu --bulk-input-cpu \
  --out /home/administrator/rctd_bench/runs/next-checkpoint-current

.venv/bin/python benchmarks/regression/bench.py compare \
  --baseline /home/administrator/rctd_bench/runs/next-checkpoint-baseline \
  --candidate /home/administrator/rctd_bench/runs/next-checkpoint-current \
  --out /home/administrator/rctd_bench/comparisons/next-checkpoint
```

只验证 Full 时，两次 run 都使用 `--stages full`；pilot 可使用 `/home/administrator/rctd_bench/colon-pilot-v2`。`--bulk-input-cpu` 是独立 Normalize 输入准备选项，不能仅因为 pipeline 调用已在 CPU 就省略它。

未来新增 sigma 观测或执行参数时，先补充评估器记录与兼容性检查，再给出实际命令；这里不虚构尚不存在的 `--sigma-batch-size` 等开关。

## 11. 后续开发约定

以下保留原文的开发约束，供后续获得实现授权时使用，本轮没有执行这些优化或提交操作。

- 先读相关源码和测试，画清张量 shape、驻留和生命周期，再写最小补丁。
- 不改数学模型、阈值、候选/分类语义，不删 longdouble 等精度逻辑换性能。
- 每次显存修改有针对性的分配/扩展验证，每次数值路径修改有真实求解 parity；mock 测试与真实算法证据分别标注。
- 保留 baseline、原始结果与失败记录；不覆盖旧报告或在失败后静默更改阈值。
- patch、测试、评测和文档对应同一个源码身份；现有用户暂存内容不重新整理或覆盖。
- 工作顺序为阅读 → 数据流 → 最小修改 → 针对性检查 → parity → 内存 → benchmark → 更新记录；提交 Git 由后续任务明确处理，本轮不执行。

## 12. v0.4 对 v0.3 的整合记录

- 保留 reduce-before-transfer、spot-major、C/B 解耦、GPU 结果及时回 RAM、同步优先和 Triton 后置的主设计。
- 将“P0.1/P0.2 已完成”拆成实现状态、已测源码状态和验收状态，纠正对 mock 测试覆盖范围的描述。
- 纳入保存 CPU 入口后的真实全量报告，解释阶段改善与流程峰值不变的原因，明确零容差未通过。
- 根据用户选择，在 Doublet 前新增 P0.4 sigma；区分 Q/SQ 全表驻留、采样 IRWLS 工作区与 likelihood 临时张量。
- 标记评测后新增的 Full `as_tensor` 改动未验证，避免误把旧报告套到当前工作区。
- 将官方评测协议、当前标注代理指标、R parity 和未来 profiler 指标分开，明确尚未覆盖的范围。
- 补齐下一轮入口、证据路径、命令与验收门槛。原始 v0.3 文件保留，仓库本文件作为后续维护主文档；下载目录的 v0.4 是本次交接副本。
