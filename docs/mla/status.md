# MLA 当前状态

当前分支 `feat/mla-cann-migration`，本轮起点 `c4df562e9f2e6b075d5c9b04cdf4685c732136f4`。**M10b1 独立 BSND split-KV partial attention 已闭环。** 本轮只完成独立 partial 数值组件，不接通 FD 合并或公开 split。

## 当前能力与本步边界

生产能力仍为 M09e 的五布局无 mask/causal/window、共享 512+64、FP16/BF16；公开 `num_splits` 仍仅接受 1。M10a CPU 分片/workspace 未改变算法。新增独立 Kernel 的 partial 特化复用任务环/Cube/Vector，按 M10a 端点与 workspace 生成真实 FP32 部分 O、八路广播 sum/max；一核一个 M/S2 片，超过物理核数的显式片由 probe 分波。没有新的 metadata 或占位 accumulator，CANN/CATLASS 未修改。

固定源在 FD 写回前已执行 aMLA RowDivs，部分 O 已归一化，仍为 FP32；本步纠正 M10a 的相反文字描述。sum/max 不是 LSE，有 tile 的全遮罩行保留源原始缓冲语义，无 tile 空片由设备初始化为 O=0、sum=0、max=-inf。FA3 accumulator 重排、空行适配、LSE 转换及 FD 归并尚未接通，协议和对照边界见 [design 6.2.9](design.md#629-独立-bsnd-split-kv-partial-attention)。

模块单元归 `tests/mla/unit/split_kv_attention/`，完整入口 `--unit split_kv_attention`。独立组件未接通公开 split，因此不注册伪 split E2E；后续公开接入时必须新增自己的日常清单。旧无 mask E2E 不替代本模块的 partial/mask 验收。

## 本次验收证据

证据根目录 `/tmp/mla-m10b1/`；构建/reference 沿用 `/tmp/flash-attention-npu-mla-46928a793876/mla-validation-artifacts/`。`build/` 只保留 `mla-codex-loop/`。M10a 累计 E2E 119 项原始证据 13 文件哈希完整，核对见 `takeover.json`；本步共享 Kernel/Vector 和构建驱动改变，累计 E2E 实跑。**继承 E2E 为 0，历史单元/性能继承为 0。**

| 检查 | 本次结果与耗时 | 证据（相对根目录） |
| --- | --- | --- |
| 双 dtype 代表与最长 KV | 8/8，94.8 秒，含独立 probe 编译；reference 命中 8、首次计算 0 | `check/summary.json` |
| 其余边界与 host 失败原子性 | 36/36，125.5 秒，含独立 probe 编译；新输入 reference 计算并缓存 34 | `acceptance/summary.json` 的 `split-kv-attention` |
| 组件性能 | 2/2，117.0 秒，含双 dtype probe 编译；6 个固定输入配置 | 同 attempt 的 `split-kv-attention-profile/probes/{fp16,bf16}/profile/` |
| 验收工具 | 11/11，5.2 秒 | `tools-final/summary.json` |
| 命令与收集 | 完整模块 46（正确性 44、性能 2）、工具 11、累计 E2E 119；只计集合核对 | `final-dry-run.json`、`final-collect/`、`tools-collect/`、`device-collect/` |
| 累计小输入 E2E | 119/119，26.1 秒；reference 命中 266、首次计算 0 | `acceptance/summary.json` 的 `e2e` |
| 构建与实际加载 | 构建 485.1 秒、runtime 21.4 秒；固定源探针和生产扩展通过 | 最终 attempt 的 `build-manifest.json`、`runtime.json`、`e2e/loaded.json` |
| 原始证据与源码/加载身份 | 五组证据完整、源码未变；6333 项构建依赖和 12 项产物完整，独立 probe 与生产实际加载匹配 | `final-evidence-check.json` |
| 文档与仓库质量 | 涉及的 14 个本地链接/锚点、差异和 staged 质量检查全部通过 | `doc-links-final.json`、`quality-final.log` |

正确性共 42 个 shape/dtype 配置、1310 个真实分片，另有双 dtype host 失败原子性检查。覆盖无 mask/causal/window、非零 S2 起点、M=16/17/128/129/255/256 和尾行、N=32/128/512 与尾列、空片/全遮罩/零 KV、跨 M/核边界、1/2/8/256、零/负 scale、内容/RoPE 扰动、aMLA 跨块最大值变化、三次重复及输入/输出/workspace guard。有效行 O/LSE/max/sum 的 1344 次双数学比较全部通过，源精度限制为 0；部分 O 和原始八路 sum/max 与固定源逐元素一致。

全遮罩行原始值单独保存，不将其当成 FA3 LSE。例如 `causal-tail` 的源片内空行实际为有限 max≈-1e12、sum=7，而非规范空片的 max=-inf/sum=0；后续归并必须做空行适配。全部输入、双数学结果和双方原始部分缓冲在各 dtype 的 `accuracy/*.pt`，摘要及哈希在同名 JSON。数值用例执行合计 37.5 秒（不含 probe 构建），最大进程 RSS 2151536 KiB；成本边界 Sk=16384。开发首次计算 8 个代表/reference 后，最终代表全部命中，最长 KV 的检查从约 27 秒降至不足 1 秒；同输入未重复计算 reference，设备仍实际执行。

组件计时使用预热 5、五轮各 20 个 wall/10 个 profiler 样本及固定 CPU 亲和性。6 配置的迁移/源耗时比算术平均：Kernel **0.997403**，预分配 probe 调用 **1.000964**；分别有 1/2 个单例大于 1，全部单例、wall、Kernel CSV 与原始 profiler 数据保留。它们比较相同片的 partial 阶段，源初始化保留，不包括 host 分片/分配/H2D，不称公开 E2E；未用公开平均阈值包装组件结果。

### 复现与恢复

```bash
python3 -m tests.mla --suite split-kv-attention-tools --output /tmp/mla-m10b1/tools-final
python3 -m tests.mla --suite split-kv-attention-check --device 6 --output /tmp/mla-m10b1/check
python3 -m tests.mla --suite split-kv-attention split-kv-attention-profile e2e --device 6 --output /tmp/mla-m10b1/acceptance
```

各组先 dry-run/collect-only；模块全集也可用 `--unit split_kv_attention`。最终代表 attempt 为 `check/attempt-1791980038413488270`，边界/性能/E2E 为 `acceptance/attempt-1791980180732089500`。恢复先核对进程、summary、原始证据与依赖，不能只看已有 `.so`。独立代表检查用本次新编译 probe，未调用当时加载的旧生产扩展；最终 E2E 用本次生产构建。

## 环境差异与失败记录

接手 HEAD/分支与 M10a 一致、工作树干净；无遗留 MLA 测试/构建，第 6 张 910B3 空闲，仅有本轮无头控制器。Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN 9.0.0-beta.2；CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`。初始实际加载扩展位于仓库 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA-256 `f221a81a97b2524c6aba78620e710e7855ce4ee3718cc356ed1f0b12cd1b15de` 与 M10a 一致，见 `environment.json`。最终加载同一路径新构建，SHA-256 **`275c8663ed275861be4319077c9df77973a3aab9eb8af76734bca08d3f950ec2`**，与 manifest、runtime、E2E loaded 记录一致。

CANN 源仓 HEAD 为 `2060a3db064c3352fec6262faed8a2decaf3a233`；oracle 从固定 `ca18aee610729ca5270a44e2d84c751bc687fbf0` Git 对象提取，不使用当前源树替代。开发与最终检查未发生数值/编译失败；首次格式检查改写 Python/C++ 格式后复查通过。加入源独立参数构造和 host 失败原子性用例后，重新固定实现并复跑受影响代表，数学缓存命中；此前开发结果仅作开发记录。最终实现固定后只更新文档，无未解决验收失败。

普通 FA3、历史模块单元/性能、公开性能、full-matrix 和 910C 实机未运行，不属于本独立 910B 组件的必要范围，不写为继承或通过。仓库质量使用 `PYTHONPATH=/tmp/mla-m09e/quality-deps ci/code_quality.sh --staged`；该变量不用于设备验收。

## 唯一下一步

[M10b2：独立 FD 归并与 FA3 accumulator 适配](plan.md#唯一下一步-m10b2独立-fd-归并与-fa3-accumulator-适配)。本轮只完成 M10b1，提交成功后立即停止，不开始 M10b2。当前闭环提交由 `git log -1 -- docs/mla/status.md` 定位，无需追加自身提交号。
