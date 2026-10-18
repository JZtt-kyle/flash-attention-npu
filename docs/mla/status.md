# MLA 当前状态

**M15g2 已完成两种表示、五布局的调度选项完整公开接入，必要正确性、回归和性能验收均通过。唯一下一步是 [M16 图执行](plan.md#唯一下一步-m16-图执行)；本轮提交后停止。** 接手基线为 `feat/mla-cann-migration` / `eebbdd4`，工作树原本干净；这是最近专门文档轮后的第四个成功开发轮。

## 当前能力与边界

`pack_gqa=None/True/False` 与合法 `sm_margin` 已接通 host、AICPU、Kernel、metadata 和公开 real/fake 入口，覆盖 compressed 512+64 / prefill 192/128、BSND、TND、TND Q + BSND cache、BSND Q + PA、TND Q + PA。v47 记录复用 M15g1 的坐标、分核和资源预算；unpacked 按独立 Q head 调度，margin 预留 AIC 与两倍 AIV。默认有效选项保留既有路径，三 mask、全部 split、softcap、真实 accumulator 与布局能力同时接入。契约见 [design 5.21](design.md#521-gqa-调度选项记录)。

完整公开性能输入暴露了共享 softcap 的比较尾部错误，已修复并补齐双 dtype 的组件与公开 PA48 尾行回归；日常 E2E 累计 783 项。CANN 与 CATLASS 未修改，生产路径不转调 CANN。graph replay、910C 生产平台仍未开放。

## 最近一次闭环证据

证据根为 `/tmp/mla-m15g2/`。下列均为本轮实际执行并通过的最终集合；源码/构建变化后旧 E2E 不继承，继承数为 **0**。`closing-evidence.log` 使用统一驱动已有校验复核当前依赖、用例清单、结果身份和全部原始证据哈希，并检查生产及数值源探针的构建/加载身份；不以历史中间结果替代本表。

| summary 目录 | 必跑组 | 通过项 / 耗时 |
| --- | --- | --- |
| `tools-fixed-2/` | `gqa-options-tools` | 21 / 13.1 s |
| `cpu-fixed/` | `unit-gqa_options`；`gqa-options-sanitize` | 109 / 37.3 s；109 / 116.5 s |
| `component-fixed/` | `softcap-mask-tail-check`；`softcap-profile-check` | 6 / 318.4 s；8 / 40.6 s |
| `device-fixed-check/` | `unit-softcap`；`gqa-options-check`；`gqa-options-source`；`gqa-options-profile-check` | 100 / 245.0 s；80 / 51.1 s；24 / 28.0 s；1 / 110.1 s |
| `device-fixed-final/` | `detail-gqa_options`；`gqa-options-boundaries` | 364 / 412.9 s；100 / 20.9 s |
| `device-fixed-final/` | 累计 `e2e`；普通 FA3 `ordinary` | 783 / 164.3 s；504 / 36.0 s |
| `device-fixed-final/` | `softcap-profile`；`gqa-options-profile` | 8 / 113.2 s；1 / 1068.0 s |

对应 attempt 依次为 `1792312294537429420`、`1792312317715479660`、`1792312741648106790`、`1792313159349934560`、`1792313676509108600`。专项含真实/fake/compile、长度/stride/空值、复用/生命周期、24 项固定源和 20 项隔离 release Trap；native 夹具实际覆盖五布局、两种表示和双 dtype。固定源 O/LSE 逐元素一致，partial accumulator O 一致、LSE 满足 1e-6，双数学通过；本步没有新增源精度例外。

最终两次设备执行各用 1.3 s 校验构建，加载身份分别耗时 21.3 / 21.1 s；实际扩展为本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `18078f0cb681b026257dda07358bc0f6c9a1f5674c016831e07a5b389fea5fb9`。`fix-acceptance-collect/` 与 `fix-selection-check.log` 核对最终集合：原日常 781 节点、原专项 362 节点完整保留，仅各增双 dtype 尾部用例；普通组仍为 504 项。

数学缓存仍按内容复用：最终专项命中 455 次，E2E 命中 923 / 新增 94，普通组命中 1 / 新增 30，公开性能命中 76 / 新增 24。修复后的完整 softcap 单元命中 6 / 新增 98，独立尾部组命中 1 / 新增 5；GQA 代表命中 79 / 新增 1，固定源命中 24。待测实现均实际执行。

## 性能结果与适用范围

最终公开性能证据为 `device-fixed-final/attempt-1792313676509108600/gqa-options-profile/public-profile/`，保留四组 `accuracy.json`、逐项 `profile.json`、原始 wall/kernel 数据和汇总 `average.json`。固定 **80** 个配置全部完成正确性/native 预检后才开始计时；依据既有 design，40 个 single/explicit 原生 FD 配置和 20 个正 softcap 配置单列，适用平均的 20 个 cap=0 auto 配置平均耗时比为 **0.7392201795800999 ≤ 1**。配置均值退化 0 个，入口/R 分段退化 14 个；`implicit_1` 分段均值为 1.0242560589294685。全部结果保留，未按耗时筛选输入。

组件性能 8 项也已实测：cap=0 partial kernel 比值范围 0.9377–1.0473，FD 为 1.7719–1.9792；正 cap 对无 cap 源仅作参考。组件单例结果不替代公开平均门槛。公开采样主要开销为每组约两分钟的 profiler 整理，未更改预热、重复次数、精度阈值或计时边界。

## 本步失败、修正与未运行项

`profile-final/summary.json`（attempt `1792310556058094930`）在 prefill FP16、BSND Q + PA48 KV、Q=17、KV=(1057,513)、Hq/Hkv=8/2、unpacked、margin=1、auto、causal、softcap=15 的双数学预检失败，O 最大误差 0.00390625，partial O/LSE 同时异常。原始张量为该 attempt 下 `gqa-options-profile/gqa-42e5f270a4c8c01b27f8.pt`；`profile-failure-tensors.log`、`tail-repro/`、`mask-tail-before/` 保留分析与独立复现。

根因是 910B `CompareScalar` 的 count 重载只处理完整 64 个 FP32 元素，32/480 元素调用漏写尾部 mask 并残留 Tanh scratch 位。比较范围现已在既有 scratch 内向上对齐到 64，Select 仍只使用有效元素；未更换算法或放宽阈值。新增 compressed 32、prefill 32/480 的双 dtype 组件用例和双 dtype 公开尾行用例；共享 Tile 改动后的必要组全部重跑，结果见上表。

中间构建/夹具失败及设备锁拒绝目录保留，不作通过证据。特别是 `tail-component-check/` 在生产构建完成后因工具测试修改使准备身份失效，attempt 仍记为失败；驱动随后重新核对构建 manifest 并完成上述最终执行。`tools-fixed/` 的失败模拟探针 stub 已修正，`tools-fixed-2/` 全通过；`mask-tail-fixed/` 被设备锁拒绝、未运行。旧窗口/接口测试中已开放行为的拒绝断言、TND fake 长度及空 KV batch 夹具均已修正并由最终集合覆盖。

未运行图重放、910C 实机与未选中的历史组件穷举；这些不在 M15g2 的必要集合。不能由本轮结果宣称全部历史单元或全量矩阵通过。

## 复现与环境

以下使用统一驱动；首次执行先加 `--dry-run`、`--collect-only` 核对命令和集合，再固定实现运行。设备收集同样需要 NPU 可见，sanitizer 需要允许 ptrace。输出目录改用新的名称可保留本次原始证据。

```bash
export MLA_NATIVE_ROOT=/tmp/mla-m13b/native
python3 -m tests.mla --device 7 --suite gqa-options-tools unit-gqa_options gqa-options-sanitize --output /tmp/mla-m15g2/reproduce-cpu
python3 -m tests.mla --device 7 --suite softcap-mask-tail-check softcap-profile-check --output /tmp/mla-m15g2/reproduce-component
python3 -m tests.mla --device 7 --suite unit-softcap gqa-options-check gqa-options-source gqa-options-profile-check --output /tmp/mla-m15g2/reproduce-check
python3 -m tests.mla --device 7 --suite detail-gqa_options gqa-options-boundaries e2e ordinary softcap-profile gqa-options-profile --output /tmp/mla-m15g2/reproduce-final
PYTHONPATH=/tmp/mla-m10c8a/quality ci/code_quality.sh --staged
```

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；卡 7 Ascend910B3，20 AIC/40 AIV。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，只读 CANN `/home/zhangqijia/repo/ops-transformer` 固定为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`；FA3 参考位于 `/tmp/mla-fa3-reference/d15f1531a460ba456f41b01a774f33ab2db8febf/`。

构建、源探针、数学缓存位于 `/tmp/flash-attention-npu-mla-46928a793876/`，native 包位于 `/tmp/mla-m13b/native`，验收结果位于 `/tmp/mla-m15g2/`；`build/` 只保留 `mla-codex-loop/`。本次影响的 24 处文档链接/锚点、`git diff --check` 和 46 个暂存文件的仓库质量检查均通过，日志为 `changed-doc-links.log`、`final-quality.log`；未绕过提交钩子。只提交本轮文件一次，成功后立即停止。
