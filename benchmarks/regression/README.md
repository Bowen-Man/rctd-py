# 固定数据回归评估

本目录是独立工具，不修改 `src/rctd/`。在 **WSL/Linux 的仓库虚拟环境**运行。
当前配置使用 `Benchmark_new/01` 下载的结肠癌真实数据。输出默认建议放在
`~/rctd_bench`，避免数据和代码快照进入 Git。

## 日常使用

在 `/home/administrator/Github/rctd-py` 中：

```bash
# 只做一次：固定数据、设置、原始提交和该提交的校准结果。
.venv/bin/python benchmarks/regression/bench.py prepare \
  --config benchmarks/regression/colon_pilot.json \
  --case ~/rctd_bench/colon-pilot --source HEAD

# 保留这个基线，以后的改动都可以与它比较。
.venv/bin/python benchmarks/regression/bench.py run \
  --case ~/rctd_bench/colon-pilot --source baseline \
  --out ~/rctd_bench/baseline

# 每次改完代码后，换一个输出目录；未提交的源码也会被快照保存。
.venv/bin/python benchmarks/regression/bench.py run \
  --case ~/rctd_bench/colon-pilot --source working --stream-gpu \
  --out ~/rctd_bench/change-001

.venv/bin/python benchmarks/regression/bench.py compare \
  --baseline ~/rctd_bench/baseline --candidate ~/rctd_bench/change-001 \
  --out ~/rctd_bench/comparison-001
```

`report.md` 是可读报告，`comparison.json` 是完整比较，`run.json` 包含每次测量、
环境、逐类型指标和重复运行一致性。各阶段的 `output_*.npz` 保存原始输出，
`worker.log` 保存运行日志。输出目录不允许覆盖；没有自动“接受新基线”。
退出码：0 成功/数值通过，2 运行阶段失败或数值不通过，1 输入/环境/配置错误。
性能变慢不混入数值 PASS/FAIL；报告单独给出 faster / slower / inconclusive。

`--source` 也可指定任意本地 Git 提交。`--stages full` 可以只测 Full。
`--bulk-input-cpu` 只影响独立 Normalize 测试的输入设备，不改变完整流程；
仅在所选实现支持 CPU 输入时使用。旧版本没有 `stream_gpu` 时，开启该选项明确报错。
CPU 输入通过 `as_tensor` 共享冻结 NumPy 数组的内存，与当前实际入口一致；
评估器不会为这些输入额外复制一份完整 CPU 矩阵。

`colon_pilot.json` 固定随机抽取 256 个真实 spot，**不是全量结论**。
`colon_full.json` 使用所有 11,385 个 spot；另建 case 后按上述命令运行。
需要容量曲线时，用不同 `max_spots` 建独立 case，保持 batch/dtype/其他设置固定；
每个规模先比较相同输入，再比较各规模的显存峰值。OOM/超时不能算作数值通过。

## 三个测量范围

| 阶段 | 输入和测量范围 | 用途 |
|---|---|---|
| normalize | 基线实际传给 fit_bulk 的原始输入；包括输入张量创建/传输和返回 CPU 结果 | 聚合、Normalize 的数值及内存回归 |
| full | 冻结的基线基因、profile、sigma、Q/SQ、counts；包括输入传输和 CPU 输出 | 隔离分批传输及 solver 改动 |
| pipeline | 同一原始 AnnData，重新构造 Reference、RCTD，使用真实 run_rctd 入口 | 捕获基因筛选、Normalize 调用、sigma 及 Full 的联动变化 |

H5AD 读取、检查、抽样在计时外，单独记录 `input_load_validate_s`；pipeline 中的
Reference 构建在计时内。每个阶段是独立进程，同一阶段先完整预热一次、再运行五次。
每轮 pipeline 都重新构造对象，避免 `is_normalized` 跳过工作。
计时前后 CUDA 同步；阶段观察器只计时/读取结果，不替换 solver。
`platform_total_s` **包含** `fit_bulk_s`、`choose_sigma_s`，不能重复相加。
各轮均重置随机种子，保证 sigma 的随机采样可比。

首轮 `first_call_s` 包括进程内首次调用开销，不能等同于机器完全冷启动。
每个 worker 有独立编译缓存；默认 compile=false，测 compile=true 必须另建 case。
时间报告中位数，原始列表提供范围/波动。至少三次重复、范围不交叠且中位数变化超过
5% 才给出 faster/slower；这是保守的描述规则，不是置信区间或显著性检验。
运行时请关闭其他 GPU 计算任务，不要同时运行两组基准。

CUDA allocated 是 PyTorch 张量分配峰值，reserved 是缓存分配器保留峰值，单位 bytes。
reserved 包含预热留下的缓存；不在各轮之间 empty_cache。二者都不是整卡 nvidia-smi。
RSS 每 10 ms 采样，可能漏掉短峰值；另存 Linux 进程生命周期的 ru_maxrss（包括读取数据）。
不会将前后 ru_maxrss 差误称为某阶段峰值，也不宣称记录了完整系统 RAM/VRAM。

## 数值一致性的硬检查

- 数据文件、Q 文件、固定中间输入均记录 SHA256；每次运行前后核验。
- 保存实际源码文件内容、Git SHA、工作区 diff、评估器代码及环境版本。
- 比较时要求 case、配置、评估器和软硬件环境相同；算法源码和显式优化开关允许不同。
- spot ID 顺序、基因集合/顺序、细胞类型顺序、pixel_mask、sigma、converged 必须完全相同。
- 比较**未归一化原始 weights**、bulk weights、normalized profiles；NaN/Inf 即使相同也失败。
- 浮点条件为 `abs(candidate-baseline) <= atol + rtol*abs(baseline)`。
  默认 atol=rtol=0，先看是否逐元素一致；不能看到失败后自动放宽。
- 每个计时输出都对照基线首个输出，同时检查两边各自的重复一致性。
  报告最大绝对误差、RMSE、超差元素数量、最差位置及对应值。

如果以后允许有误差，应在看结果前确定容差、另建 case。FP32/FP64 分别建基线。
与旧 Python 代码一致不代表与 R 一致；R parity 需要另外提供对应输入和 R 输出，
当前报告不会捏造 R 运行结果。仓库既有 R concordance/parity 测试可独立运行。

## RCTD 官方评测口径和扩展指标

依据 [RCTD 原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC8606190/) 及
[官方 Figure 3 分析代码](https://github.com/dmcable/spacexr/blob/master/AnalysisPaper/MainFigures/figure3.Rmd)
核对口径。官方代码 blob SHA：`ab1f9d883e1d420dd2e2da4335f3522d1e4b2be9`。
论文的某次实验分数不是通用验收阈值，也不是本数据集应达到的分数。

| 类别 | 指标 | 本工具中的处理 |
|---|---|---|
| 官方比例恢复 | 已知混合比例与预测的 RMSE | 提供整体及每种类型 RMSE；当前全模式/真实数据不能冒充原论文按双细胞类型对的模拟实验复现 |
| 官方分类 | singlet/doublet 分类率 | pipeline 配置 mode=doublet 时计算；同时列出排除 reject 和包含 reject 的分母 |
| 官方置信调用 | confident cell-class identification | doublet 时第一类型排除 reject，第二类型仅 doublet_certain；在真双类型且少数比例≥0.25 的 spot 上按类型对汇总均值/标准差，支持 config.class_df |
| 官方平台效应 | 已知平台效应的恢复 R² | 当前无平台效应真值，标记 N/A |
| 官方生物学验证 | marker/空间分布合理性 | 需要独立验证方案，不用当前标注代替；标记 N/A |
| 补充比例指标 | MAE、逐类型 Pearson、macro Pearson、JSD | 明确公式/分母，见下文 |
| 补充分类指标 | 主导类型 argmax 准确率 | 不是官方 doublet 分类；另报真值并列最大数，按固定列顺序打破平局 |
| 工程指标 | 时间、吞吐、CUDA allocated/reserved、RAM、OOM、超时、重复性 | 与生物学指标分开判定 |

`mode` 默认 full，另建 `mode=doublet` 或 `mode=multi` 的 case 可评估相应完整流程；
独立 full 阶段始终评估 Full solver。stream_gpu 当前仅支持 full。
Full 不产生 spot_class，官方置信分类指标明确 N/A；不从 argmax 猜造 confidence。
Doublet 的比例指标用选定类型的 weights_doublet，reject 行计入无效预测；multi 使用 sub_weights。

比例指标：预测 w 与标注 t 分别按行归一化 p=w/sum(w)、q=t/sum(t)。
整体 RMSE = sqrt(mean((p-q)^2))，MAE = mean(abs(p-q))；逐类型沿 spot 计算。
Pearson 在逐类型向量常量时为 null，macro 只取有定义类型并报告数量。
JSD = 0.5 KL2(p||(p+q)/2) + 0.5 KL2(q||(p+q)/2)，这里是 divergence，**没有开平方**。
零标注行、负权重/非有限/零总量预测行会计数并排除比例指标，不静默裁剪或伪造均匀分布。
标签必须精确对齐；不静默丢弃缺失的细胞类型。

本数据 annotation.csv 看起来是**细胞计数**，而 RCTD 权重代表 RNA 贡献比例；
这些数值应解释为“与所提供标注的一致程度”，不能直接声称真实准确率或论文复现。
标注产生方式及其与参考集的独立性还需确认。RMSE/Pearson 改善也不能掩盖原始权重回归失败。

## 数据读取约定

使用 raw counts 的 X；若原始计数在 layer 中，可设置 spatial_layer/reference_layer。
当前两份 H5AD 基因名称有重复，所以显式采用唯一 gene_ids 对齐，不修改原文件。
注意 RCTD 内部的 `mt-` 名称过滤在使用 gene_ids 时不再等同于用符号过滤；如要显式去掉
线粒体基因，应先定义独立、固定的数据预处理协议，不能在不同版本间偷偷改变。
Reference 使用 Level2 标签与既有默认过滤/每类型最多 10000 细胞的固定抽样。
抽取 spot 在全部原始基因上进行，nUMI 的计算仍由原算法完成。

工具不下载数据、不安装依赖、不切换 Git 分支、不提交代码、不修改源码或原始 H5AD。
失败目录保留日志；重新尝试请选择新目录。完整 artifact 可能较大，可手动归档旧实验。

## 验证

```bash
.venv/bin/python -m pytest -q tests/test_regression_metrics.py tests/test_regression_harness.py
.venv/bin/ruff check benchmarks/regression tests/test_regression_metrics.py tests/test_regression_harness.py
```
