# MLA 当前交接

## 状态与下一步

- 当前步骤：M04b2b1 已完成，优化在线 FP32 输出重缩放与归一化；连续 BSND 支持范围仍为 KV `1..1024`。
- 当前分支：`feat/mla-cann-migration`；接手基点 `478f757`。方案已确认，本轮一个提交后停止。
- 本步产物：[在线累积 Block](../../csrc/ascend910/flash_attn_npu_3/mla/mla_online_epilogue.hpp)、[每核输出行数检查](../../tests/test_flash_attn_npu_v3_mla_online.py)。
- 唯一下一步：M04b2b2，继续 KV `>1024` 的流水与调度优化，验收含 `8193` 的单/多 query、FP16/BF16 O/LSE、重复执行及同轮 CANN 性能；保持显式 `num_splits=1`。

M04b2b 在实现前拆为 M04b2b1（逐 tile 输出累积优化）与 M04b2b2（更长 KV 调度）。本步没有接通 >1024，也没有完成 8193 验收；没有生产 CANN 回退。

## 当前入口与实现

- `flash_attn_func(q, k, v, qv=...)` 支持 FP16/BF16、`Dq=64/Dv=512/Hkv=1`、连续 BSND、正 batch/query 长度、KV `1..1024`、无 mask、显式 `num_splits=1`。O 为 `[B,Sq,Hq,512]`，自然对数 FP32 LSE 为 `[B,Hq,Sq]`。
- KV `1..128` 保留单 tile 路径；`129..1024` 使用 128 列 tile 的在线 softmax。最大值、分母和输出分子保留 FP32；默认 scale 为 `1/sqrt(576)`，显式有限 scale 保持 FP32，支持零/负值。
- 复用固定 CATLASS QK+RoPE / PV Block，Q 留在 L1，PV 复用内容 KV；score/probability/PV partial 双缓冲和跨 task 信号排空保持。KV >512 且空闲核充足时按 8 个 head 分工，否则 16 个；每任务遍历完整 KV，没有 split/合并。
- 本步将每个向量核少于 8 行时的输出重缩放/归一化改为逐行指令、沿 512 列重复，同一 FP32 行系数覆盖整行；指令数由 8 降至实际行数。8 行及以上保留逐列遍历，兼容原 C1V1 的 16 行情形。乘法、加法和除法的数学顺序保持；仅实测 910B C1V2。
- 紧凑 dense custom op、Python/fake/C++ 校验、公共入口和 host 调度均未修改。连续非零 storage offset、极值、尾块、跨任务复用和异长调用连续入队仍正确。
- MLA prefill、varlen/kvcache 公共入口、KV >1024、空 KV/零 query、外层非连续 stride、mask/window、自动/多 split、softcap、pack_gqa/sm_margin、metadata、追加 KV、预分配 out、实际长度/页表/rotary/descale 和梯度请求仍明确拒绝。MLA metadata/图执行尚未验收。

## 最近闭环证据

| 检查 | 结果 |
| --- | --- |
| 接手核对 | PASS：工作区干净；上轮 320 项 checksum、扩展/源码/对象哈希全部一致；CANN 源仓既有状态保持 |
| 构建溯源 | PASS：仅 v3/910；4 个 MLA 单元按 header 依赖重编，只有 FP16/BF16 online 对象哈希变化；2 个 short、flash_api 和 9 个普通 FA3/AICPU 对象哈希保持；最终扩展重新链接 |
| 910B MLA | PASS：完整 `586 passed`，无 skip；契约 233、短路径 141、online 212；新增 22 项覆盖每个向量核 1..8 行及奇数 head，含双 reference、O/LSE、输入保持与重复执行 |
| 固定源码 CANN | PASS：47 个 shape/scale × FP16/BF16，共 94 组；CANN 与新实现分别通过原双 reference/comparator，阈值未改；94 次 FIA launch 全匹配配套 vendor binary |
| 普通 FA3 | PASS：`11 passed`，无 skip；dense/varlen/kvcache/FlashDecode、metadata、编译图及 dense/varlen backward |
| CANN 性能 | PASS：32 例同轮交替测量，kernel 比值 `0.603..0.975`、端到端比值 `0.833..0.882`；逐例低于 1.0，kernel 配对 bootstrap 95% 区间上界最高 `0.987` |
| 上版对照 | 同进程加载保留的上版扩展，与新版使用同一 C++ 参数逐例交替；32 例 O/LSE 逐位一致；逐例 kernel 与调用时间及置信区间见 `versions-summary.json` |
| 既有 CPU | 本步未改 reference/profile/provenance，继承上轮 88 项通过记录和已核对的证据，不记为本轮重跑 |
| 质量与完整性 | 提交前执行 `ci/code_quality.sh --staged`、文档检查、`git diff --check` 和 `final_integrity.py`；结果与文件哈希保存在本轮证据目录 |

CANN 性能每实现预热 10 次，同步调用 7 轮 × 50 对 AB/BA 交替，profiler 独立 3 轮 × 10 对；1920 次调用均对应一个 kernel，无 Slice 转换。相同输入/dtype/scale、物理 7 号卡；CANN 原生自动策略、MLA 单 split，无 metadata。门槛保持逐例比值 `<=1.0`。

部分结果如下，kernel 单位微秒；全部逐例数据在 `compare-cann-final.json`。

| 用例 | CANN kernel | MLA kernel | kernel 比值 | 调用比值 |
| --- | --- | --- | --- | --- |
| kv1023-q1-float16 | 21.7205 | 21.1805 | 0.9751 | 0.8699 |
| kv1023-q1-bfloat16 | 21.8505 | 21.1910 | 0.9698 | 0.8671 |
| kv1023-q7-float16 | 35.8810 | 27.7600 | 0.7737 | 0.8709 |
| kv1023-q7-bfloat16 | 36.2900 | 27.8305 | 0.7669 | 0.8800 |
| kv1024-q1-float16 | 21.5200 | 20.8405 | 0.9684 | 0.8717 |
| kv1024-q1-bfloat16 | 21.4900 | 20.9405 | 0.9744 | 0.8720 |
| kv1024-q7-float16 | 34.6210 | 27.5310 | 0.7952 | 0.8731 |
| kv1024-q7-bfloat16 | 34.7410 | 27.7005 | 0.7973 | 0.8757 |

上版对照的收益主要来自较少输出行的任务：383 KV、每核 1 行约快 2%；D01 和部分 513/1023/1024 单 query 有小幅收益。7/8 行以及部分多 query 的区间跨 1，不能宣布这些用例有确定收益。没有用平均加速掩盖逐例结果，更不能由当前小幅收益推断 8193 已达标。

针对性检查先通过 52 项；最终完整检查使用格式调整后的构建。旧新版 9600-kernel 大批 profiler 采集发生框架关联解析错误并停滞，已终止且保留失败证据，不用于验收。分批采集 1..4 的退出码为 0、设备计时完整，但完整日志仍有框架关联解析错误，因此也未采用。各批增加一次明确记录并排除的原生 FIA setup kernel 后复验，补足 profiler 的 launch 关联记录；最终有效批次为 batch1-repro 至 batch4-repro，共 5760 次旧新版 kernel 计时，4 次 setup 不计入统计。异常原始目录全部保留，详见 `versions-summary.json`。CANN 主验收独立完成，未发生该解析错误。

## 证据与复现

持久证据根目录：`/home/zhangqijia/repo/flash-attention-npu/build/mla-m04b2b-validation/`。

- `takeover-audit.json`、`environment-before.json`、`device-before-final.log`：接手、环境与设备核对；`before/` 保存上版扩展和 MLA headers。
- `build-{1,2}-*`、`build-provenance-{1,2}.json`：两次构建，最终为 2；实际加载扩展与精度/性能报告中的哈希一致。
- `mla-targeted-*`、`mla-final-*`、`regression-final-*`：精确命令、退出码、日志/JUnit；`run_regression.py` 保留普通 FA3 节点。
- `cann-accuracy.py` / `.json`、`dispatch-accuracy.log`、`verify_dispatch.py` / `dispatch-proof.json`：94 组精度及 vendor binary 分派来源。
- `compare.py`、`compare-cann-final.json`、`profile-cann-final/`、`performance-acceptance.json`：CANN 同轮逐例计时及验收。
- `compare-versions*.py`、`versions-*.json`、`profile-versions-*/`、`summarize-versions.py`：旧新版同进程测量，包含未采用的采集与明确标记的 profiler setup kernel。
- `quality-staged.log`、`document-check.log`、`final-integrity.json`、`checksums.sha256`：质量、文档与证据完整性。

实机使用 `login=false`、沙箱外；`run_command.py` 保留 CANN PYTHONPATH，固定线程数、物理 7 号/进程内 npu:0 和隔离 cache/plog，只有 CANN/比较/分派标签加载固定 vendor。复跑前检查设备健康/占用，更换标签和输出路径，不覆盖既有证据。

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 \
  FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python setup.py build_ext --inplace

python build/mla-m04b2b-validation/run_command.py mla-repro \
  timeout 240 python -m pytest -q tests/test_flash_attn_npu_v3_mla_online.py \
  tests/test_flash_attn_npu_v3_mla_short.py tests/test_flash_attn_npu_v3_mla_contract.py

python build/mla-m04b2b-validation/run_command.py compare-repro \
  python build/mla-m04b2b-validation/compare.py \
  --output /tmp/mla-m04b2b1-repro.json --profile-dir /tmp/mla-m04b2b1-repro-profile \
  --iterations 50 --rounds 7
```

质量工具沿用 `/tmp/mla-m01-precommit`（前置 PYTHONPATH）与 `/tmp/mla-m01-precommit-cache`（PRE_COMMIT_HOME），执行 `ci/code_quality.sh --staged`，不绕过提交钩子。

## 环境与继承基线

| 项目 | 本轮结果（2026-09-24） |
| --- | --- |
| Python / 包 | `/usr/local/python3.11.15/bin/python` 3.11.15；torch `2.7.1+cpu`，torch_npu `2.7.1` |
| CANN / 固定 FIA 源 | `/usr/local/Ascend/cann-9.0.0-beta.2`；`ca18aee610729ca5270a44e2d84c751bc687fbf0` |
| CATLASS | `769cd40a8716b28650b6bebb08db4834eea4462f`，干净且未修改 |
| 实际扩展 | 本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`；SHA256 `4ca5cd1d11ebdc226348a1eb5effe2c3b018c999c2afbaed6564ce7e917be9f0` |
| 芯片 | Ascend910B3，物理 7 号；验收前健康且空闲 |

固定源码包仍为 `build/mla-m02b-validation/opp-dense/`，长 KV 基线在 `build/mla-m02b2-validation/`，上一步闭环在 `build/mla-m04b2-validation/`。本轮未修改 reference/comparator、CANN 基线执行器/包、CATLASS 或 CANN 源仓；源仓其他目录既有删除/未跟踪状态保留。下一步仍须同轮交替比较。当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。
