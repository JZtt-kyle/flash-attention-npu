# MLA 当前状态

本轮完成 **M06c2a1：TND v3 记录与独立 AICPU 生成**。分支 `feat/mla-cann-migration`，本轮起点 `242bee6`。原 M06c2a 已按生成/消费拆为两个独立提交；公开能力仍停留在 M06b4。

唯一下一步为 **M06c2a2：TND v3 设备校验消费者**；本轮未开始该步。按 [plan](plan.md#m06非分页长度tnd-与-stride) 验证起点、长度、当前前缀和、调度与资源，并完成旧/篡改/变更记录真实 Trap、拒绝不写输出和生命周期验收。继续在已确认范围内推进，无需重复确认。

## 当前支持边界

| 项目 | 当前代码行为 |
| --- | --- |
| 设备与表示 | 910B C1V2、FP16/BF16、512+64 共享 KV、Hkv=1。目标仍含 910C，合成 C1V1 调度不代表 910C 实机验收 |
| 输入 | 连续 BSND，允许非零 storage offset；B/Sq/Sk 可为零，Hq>0，KV 容量 `0..16384`；Q 使用整容量，支持 S1G/S2 多块 |
| 公开长度 | 只读 `flash_attn_with_kvcache` 接受 None（整容量）、`0..Sk` 整数或同设备连续 `int32[B]`；支持短/非均匀/混合与全零长度 |
| 公开记录 | `get_scheduler_metadata` 的 64/512 分支生成 v2 `int32[401+B]`，含实际长度快照；真实/fake/C++ 外形一致。消费时复核当前长度，省略长度表示整容量；旧 v1、损坏或过期记录拒绝 |
| 内部 dense | 无长度且无显式记录时仍使用 v1，包括 `flash_attn_func`。只读 kvcache 精简 dispatcher 复用已有 Kernel / Block / Tile，不调用原 CANN |
| 空行与返回 | 非零容量的零 KV 样本 `O=0、LSE=-inf`，含端点外零尾部；全局空 KV 为 `O=0、LSE=+inf`。空 batch/query 保留空 shape，仍校验记录。kvcache 的 LSE 返回四项，未 split 的 accumulator 为空 tensor |
| 调度限制 | 显式 `num_splits=1`、无 mask、`softcap=0`、`pack_gqa=None`、`sm_margin=0`；scale 由当次前向提供。静态字段上界见 [design 6.2](design.md#62-调度字段与边界)，v2 另要求 `B<=2^32-1-401` |
| 尚未接通 | TND、Q-used、非连续外层 stride、PA、causal/window、自动或多 split、192/128 prefill、其他调度选项、梯度；追加 KV、内置 rotary 等范围外选项继续明确拒绝。完整图 replay 生命周期留在 M16 |

生产入口在 TPipe、attention 事件或输出写入前完成 v2 校验。非空 v2 使用平台全核 blockDim，工作核数由设备记录决定；workspace 按源最大 M 基块 256 预算，覆盖非均匀长度使 M 增大的情形。长度与记录无 D2H，使用既有双事件和 allocator stream 生命周期。实现见 [生产生成/发射](../../csrc/ascend910/flash_attn_npu_3/mla/mla_forward.cpp)、[设备入口](../../csrc/ascend910/flash_attn_npu_3/mla/mla_launch_impl.hpp) 和 [接口校验](../../flash_attn_npu_3/flash_attn_npu_interface.py)。

## 本轮组件与证据

新增 [`mla_tnd_metadata.h/.cpp`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_metadata.h)、[`mla_tnd_metadata_args.h`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_metadata_args.h) 和 [`mla_tnd_metadata.aicpu`](../../csrc/ascend910/flash_attn_npu_3/mla/mla_tnd_metadata.aicpu)。v3/kind=3 记录为 `int32[405+4B]`：沿用 401 字调度编码，增加 Q/KV 总容量、调用方 max 上界及逐样本 Q/KV 起点/长度。完整协议和错误优先级见 [design 5.4](design.md#54-v3紧凑-tnd-记录与独立-aicpu-生成)。直接复用 M06c1 的解析与调度；该算法仅将私有 Align 重命名，避免 AICPU 同编译单元名称冲突。

生成前检查输出字数/对齐/地址加法和双输入重叠；生成失败清零整个载荷。成功记录从拥有存储的调度快照序列化，不重新读取前缀；输入仍须在生成期间不可变。独立驱动沿用双事件与 allocator stream，无前缀 D2H。v1/v2 CPU 消费者拒绝 v3；**TND 设备消费者、真实 Trap、attention 寻址及公开 TND 本步均未接通**。TND Q + BSND KV、used 与 stride 按后续步骤实现。

证据目录 **`build/mla-m06c2a1-tnd-metadata/`**，总索引 `acceptance.json`，由 `audit.py` 核验。原始命令、退出码、JUnit、完整输入/输出、固定源 blobs、编译命令和源码快照均保存于此；交接环境时须一并保留。临时目录中的构建与公开回归产物已归档到 `build-tests-artifacts/`、`regression-artifacts/`，旧 AICPU oracle 另以正确的持久目录完整复验。

| 验收面 | 结果 | 证据（相对上述目录） |
| --- | --- | --- |
| CPU v3 与 M06c1 回归 | 63 + 111 项通过；v3 1,206 组固定源差分，原生非 FD 782 组、无 split 映射 424 组；含 12 组端点外零成本后缀 | `final.xml`、`final/`、`final-schedule/` |
| 完整 ASan/UBSan | 同一 174 项通过；序列化入口、缓冲守卫、调度与固定源 oracle 均启用 address/undefined 和不可恢复错误，未关闭 LeakSanitizer；常规与 sanitizer 完整差分输出逐份哈希相同 | `sanitized.xml`、`sanitized/`、`sanitized-schedule/` |
| 910B TND AICPU | 最终 17 项通过，FP16/BF16、600 组固定源差分（388 原生非 FD / 212 无 split 映射）、空容量/非法前缀、失败后复用、逐项前缀更新、共享输入与跨流释放/分配复用 | `device-final.xml`、`device-final/`、`device-final-oracle/`，加载 `device/metadata.so` |
| 旧 AICPU 与构建 | v1/v2 91 项、构建 9 项通过；新增 TND 调度头和 ABI 头的增量失效、缺少依赖证据/编译选项变更、各后端 AICPU 来源清单均覆盖 | `legacy-final.xml`、`legacy-final-v1/`、`legacy-final-v2/`、`legacy-final-*-source/`、`build-tests.xml` |
| 普通 FA3 与公开 MLA | 923 项通过：普通 metadata/compile/签名 149 项、MLA 契约 582 项、公开 BSND 192 项；后者含双 dtype 源 O/LSE、不同 scale、真实/fake/C++、长度/记录复用与流生命周期 | `regressions.xml`、`regression-nodes.json`、`regression-artifacts/` |

合计 **1,214 项去重 pytest 通过，另有同一 TND 套件的 174 项 sanitizer 复验通过；最终选入项无失败/跳过**。前轮 CPU 63 项、设备 15 项、生成对照 2 项和旧 AICPU 91 项的通过结果保留，不重复计数。

生成耗时仅作诊断：最终在设备 5、CPU 32–39、OMP/MKL=8 下，对三组调度全字段相同的均匀用例交替测量 v1/v3，双 dtype 均预热 5 次、记录 20 次 wall 样本。v3 中位数 **237–248 µs**，v1 **188–198 µs**，比值 **1.254–1.271**；计入 ctypes、事件、两组前缀登记和逐次同步。非均匀/空容量另有五组诊断。初轮并行其他 NPU 回归时绝对耗时较高，保留 `comparison/` 与 `device/`，最终数据在 `device-final/comparison-*.json` / `timing-*.json`。这不是 TND attention 精度/性能或公开 E2E 验收，不宣称公开性能达标。

## 环境与上一步复核

Python `/usr/local/python3.11.15/bin/python3` 3.11.15、torch `2.7.1+cpu`、torch_npu `2.7.1`、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`、bisheng clang 15.0.5、g++ 11.4.0。八张 Ascend910B3 健康，运行前重新核对占用；本步 TND 用设备 5、普通/BSND 回归用 4、旧 AICPU 用 7。CPU 普通/SAN 分别用 16–23 / 24–31，普通回归 40–47，旧 AICPU 8–15，构建检查 48–55。沙箱内 npu-smi 驱动初始化失败，必要 NPU 和完整 sanitizer 验证在沙箱外执行。合成 C1V1 仅说明调度一致，不代表 910C 实机验收。

CATLASS 固定 `769cd40a8716b28650b6bebb08db4834eea4462f` 且干净；CANN 固定 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，原有 67 条删除/未跟踪记录未变。本步不修改两者。新 AICPU 单元已加入 `setup.py` 并实际重建/链接：`production-build-manifest.json` 核对 **218 个源码依赖、19 个链接对象**，原有 18 个对象哈希均未变，新增 `mla_tnd_metadata.o`。实际加载的 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so` 与本轮链接产物一致，SHA256 为 **4c87d3124a51162d4f0bb7d9801c918d8811cca8c31716853f6eccbebceb90db**。

已复核 M06c1 的 `acceptance.json`、源码/文档哈希和 `final/sanitized/regressions` 完整命令与退出码；其最终无未解决失败。本轮 TND 调度源码发生私有命名变化，因此完整重跑受影响 111 项 CPU/SAN；扩展重链接后完整重跑公开 BSND 套件，覆盖此前最小公开精度检查。

初始 precheck 沿用 M06b4 文档哈希，遇到 M06c1 的正常文档变更而失败，改为当前交接哈希后通过；初始 Ruff E731 已修复。原始失败见 `precheck*.log`、`format.log`，后续通过见 `precheck-fixed-exit.json`、`format-fixed-exit.json`。质量工具沿用固定 pre-commit 4.6.2、Ruff 0.16.5、clang-format 21.1.8，命令环境为 `PYTHONPATH=/tmp/mla-m01-precommit:$PYTHONPATH PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache`，保留 CANN PYTHONPATH。提交前 staged 钩子、文档链接/锚点/步骤状态/公开签名和证据复核见 `quality-staged.log`、`doc-checks.json`、`acceptance.json`。

## 最小复现

新证据写到其他目录，不覆盖本轮结果；完整命令和环境见各 `*-command.json` / `*-exit.json`。

```bash
MLA_TND_METADATA_CPU_DIR=build/mla-tnd-wire-recheck MLA_TND_SCHEDULE_ARTIFACT_DIR=build/mla-tnd-wire-oracle-recheck python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_metadata.py
MLA_SCHEDULE_SANITIZE=1 MLA_TND_METADATA_CPU_DIR=build/mla-tnd-wire-asan MLA_TND_SCHEDULE_ARTIFACT_DIR=build/mla-tnd-wire-oracle-asan python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_metadata.py tests/test_flash_attn_npu_v3_mla_tnd_schedule.py
ASCEND_RT_VISIBLE_DEVICES=5 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c 32-39 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_tnd_metadata_aicpu.py
ASCEND_RT_VISIBLE_DEVICES=4 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_metadata_signature.py tests/test_flash_attn_npu_v3_mla_bsnd_public.py
```

sanitizer 须在支持完整 ASan/UBSan/LeakSanitizer 的环境执行，不能把关闭检查当作通过。当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位。
