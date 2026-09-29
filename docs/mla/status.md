# MLA 当前状态

## 当前能力与下一步

- 分支 `feat/mla-cann-migration`，接手 HEAD `142dd96`，接手工作区干净。本步完成 M05c2，M05 已闭环。
- 公开 dense MLA metadata 生成改走内部 `_mla_dense_metadata` C++ dispatcher；真实/fake 独立校验形状、dtype 和长度张量。只读 dense kvcache 的 NPU qv 路径由既有 `_mla_dense_forward` 完成张量及 scale 校验，整数长度转换前仍检查几何与整容量值。公开签名、401 字记录和设备消费校验保持原契约。
- 已支持范围仍为 910B、FP16/BF16、512+64 共享 KV、连续 BSND、B>0、`1<=Sq*Hq<=256`、KV 容量 `1..512`、Hkv=1、无 mask、显式 `num_splits=1` 和整容量长度。追加 KV、TND/PA、部分/零长度及梯度等范围外组合明确拒绝。
- AICPU、调度算法、Kernel/Block/Tile、metadata 协议和 stream/allocator 生命周期沿用原路径。显式复用跳过 AICPU；无 metadata 的 kvcache 内部生成；普通 FA3 和 `flash_attn_func` 保持原入口。
- **唯一下一步：M04c1，按源 M 切块/分核接通 dense S1G 多块，解除 `Sq*Hq<=256` 限制，KV 仍为 1..512。** 本步提交后停止。

## 最近一次闭环证据

最终功能与构建证据：`build/mla-m05c2-final/`；固定 CPU 亲和性的最终性能证据：`build/mla-m05c2-affinity/`；物理 7 号 Ascend910B3。最终源码与扩展的 **1,789 项检查全部通过，无失败或 skip**。初版检查及诊断另存 `build/mla-m05c2-metadata/`，不计入此总数。

| 项目 | 结果 |
| --- | --- |
| 上步复核 | PASS：2,122 份证据哈希、1,595 个去重测试、17 个生产对象、实际扩展及 303 份固定 CANN 构建文件一致；源仓状态未变，见 `previous-verified.json` |
| CPU | PASS：90 项消费者解码、启动桥及构建依赖检查，含 ASan/UBSan，见 `cpu-final*` |
| dispatcher | PASS：418 项 metadata/前向独立边界、公开坏输入、schema/fake、梯度拒绝、返回 dtype 与 compile 检查；含编译后读取更新长度、清零失败 payload 及快照独立性，见 `dispatch-final*` |
| 公开 MLA | PASS：512 项，含双 dtype 共 48 组形状/scale 的三路径源对照、舍入、跨流释放/分配压力、主机线程和 snapshot；O/LSE 与固定源逐元素一致且通过独立数学对照，见 `public-final*`、`producer-final/` |
| 设备拒绝 | PASS：32 个隔离进程，覆盖非法生成、隐式生成、旧记录复用时的长度与损坏记录；专属进程组无存活进程，见 `errors-final*`、`process-groups.json` |
| 契约与普通 FA3 | PASS：588 项 MLA 契约、129 项普通 metadata 签名/兼容/精度/compile、20 项普通 FA3 BSND/TND/分页/split/metadata/mask/softcap/scale/compile，见对应 `*-final*` |
| 构建与加载 | PASS：本步只增量重建 `mla_forward.o`，其余 16 个对象含 Kernel/AICPU 哈希与接手时一致；最终 Python 路由复用该扩展。60 份构建源码和 127 份运行源码快照、对象及实际加载库核对通过，见 `build-provenance.json`、`production-artifacts/` |
| 性能 | PASS：两种 dtype 各五组，复用 E2E、生成加一次前向及生成加四次前向逐例均 <= 1.0；见下表及 `build/mla-m05c2-affinity/performance-*/` |

最终扩展位于 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 为 `0e5539452bbd7fa06ee6857bbaab95cc9afa30c332f75be5d56eaa40bd27c29d`。全部最终功能与性能证据对应同一最终 Python 源码和该扩展。

## 性能口径与结果

FA3 与 CANN 在同一进程、相同 CPU 亲和性（67,69,70,74,79,80,89,95）下比较；CPU 组在采样前按 NPU 7 所在 NUMA 节点的利用率选定，见 `host-affinity-selection.json` 和 `preflight-resume.json`。两种 dtype 各五组，每组五轮 × 20 个 wall 样本，比较显式复用、CANN 单次、metadata 生成、隐式生成 kvcache、生成加 1/4 次前向及 CANN 四次调用，共 **7,000 个 wall 样本**。R=1/4 在采集前固定，完整段只在末尾同步，两边前向次数一致。输入 H2D 不计时；入口检查、准备、输出/workspace 分配、发射、依赖与完成同步计时。另按五轮 × 10 个样本交替采集两边 attention Kernel，共 **1,000 个 Kernel 样本**；显式复用的 AICPU 生成任务为 0。

下表均为每例 100 个样本的 FA3 / 同轮 CANN E2E 中位数比值，逐例通过 1.0 门槛：

| `(B,Sq,Hq,Sk)` / scale | FP16 复用 | BF16 复用 | FP16 生成+1次 | BF16 生成+1次 |
| --- | --- | --- | --- | --- |
| `(2,1,8,257)` / 默认 | 0.587830 | 0.581967 | 0.878740 | 0.885168 |
| `(2,1,8,257)` / 0.23 | 0.594037 | 0.626537 | 0.894813 | 0.919310 |
| `(1,7,16,65)` / 默认 | 0.595253 | 0.635363 | 0.978508 | 0.848634 |
| `(1,1,128,512)` / 默认 | 0.612526 | 0.626381 | 0.940772 | 0.909854 |
| `(2,2,128,511)` / 默认 | 0.598907 | 0.587244 | 0.867207 | 0.917642 |

| 其他指标（五组中位数范围） | FP16 | BF16 |
| --- | --- | --- |
| metadata 生成 E2E | 236.17–275.82 μs | 232.47–260.90 μs |
| 隐式生成 kvcache / CANN 单次 | 0.693–0.775 | 0.672–0.780 |
| 一次生成+4次前向 / CANN 四次 | 0.536–0.574 | 0.502–0.574 |
| attention Kernel / CANN | 0.970–1.032 | 0.972–1.029 |

全部逐例值、轮均值和原始样本保留在 `profile.json`、`wall-samples.json` 及 profiler CSV。Profiler 与原始 CANN CSV 逐行核对，PID/物理 7 号映射及采集前后独占快照通过，见性能目录 `performance-device-proof.json`。未改变五组、预热、轮数或 R=1/4 的口径来替代优化。

## 环境、诊断与限制

- Python `/usr/local/python3.11.15/bin/python3` 3.11.15；torch `2.7.1+cpu`、torch_npu `2.7.1`；CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；bisheng clang 15.0.5、`dav-2201`、`-O2`。CATLASS `769cd40a8716b28650b6bebb08db4834eea4462f`，固定 CANN `ca18aee610729ca5270a44e2d84c751bc687fbf0`，官方 FA3 `d15f1531a460ba456f41b01a774f33ab2db8febf`；子模块、CANN 既有改动及 950 后端未改。7 号设备结束时健康 OK、ECC 全零、无设备进程。
- 初版仅增加 metadata dispatcher，功能 1,705 项通过，但 FP16 R=1 五例比值为 1.026/1.014/1.001/1.003/0.999，未闭环。Python profile 与后续诊断定位到通用 custom-op 及重复 Python 张量校验开销；最终同时处理两处。平台信息查询仅数微秒，未采用缓存。初版 `pilot*` 与公开测试进程重叠，只作诊断。
- 最终源码的未固定 CPU 首轮性能采集成功，但 R=1 未全部达标（FP16 0.909/1.327/1.043/1.828/1.309；BF16 约 0.893/1.030/0.934/0.869/1.001），原始数据保留在功能证据目录 `performance-*/`。256 核主机 load 约 375–405，CANN 自身 E2E 升至约 500–700 μs；按采样前选定的相同 CPU 组复测后达到上表门槛。固定亲和性结果仍有长尾波动，未删除样本或改变中位数口径，不把它解释为每一轮都更快。环境检查导入扩展会初始化 NPU，首次预检和综合核验把检查进程自身误判为残留占用；运行时日志已核对 namespace/host PID，退出后用不导入 torch 的进程确认 7 号设备空闲。原始失败和纠正证据均保留，见 `preflight-blocked.json`、`environment-process-proof.json` 及 `acceptance.log`；性能采样前后始终只有各自采样进程。
- 首次正式 profiler 因文件系统可用空间为 0 拒绝采集，未生成设备 CSV，保留失败日志。最终在 `/dev/shm/mla-m05c2-*` 采集后逐文件校验并归档到各轮持久 `performance-*/`；`performance-*-archive.json` 保存原始哈希。`profile.original.json` 保留原报告，`profile.json` 只将两个 CSV 路径改为持久归档路径，采样值未改。
- 首次沙箱内构建 133 秒无输出后主动终止，随后沙箱外构建成功；初版 `build-initial*` 与 `build-dispatch*` 保留两次记录。最终首轮 ruff-format 自动调整一处排版，复核通过。沿用非登录 shell、沙箱外实机/sanitizer 执行；本轮 `kernel_meta` 归档至最终证据目录。
- 图捕获生命周期仍留给 M16；本轮 Torch compile 验证 eager backend/fullgraph。Trap 后不承诺继续复用执行上下文；错误测试禁用 core dump 并清理专属进程组。

## 证据与复现

三个证据目录中的精确命令、环境、日志和 JUnit 见 `*-command.json`、`*-exit.json`、对应日志及 XML；源码与生产产物快照、构建清单及综合核验见 `build-source-manifest.json`、`runtime-source-manifest.json`、`acceptance.json`、`checksums.sha256`。最终测试节点见功能目录 `run_device.py`，性能采集与归档见性能目录 `resume_controlled.py`、`run_performance.py`。复测脚本引用最终功能证据并在采样前核验 127 份源码哈希；两种 dtype 命令均记录相同实际 CPU 亲和性。接手复核和初次构建记录在两个证据目录中保留原命令与时间戳。

从仓库根目录复现，选择空闲健康的 910B 并使用新 artifact/JUnit 路径，避免覆盖证据：

```bash
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTN_BUILD_NPU=910 FLASH_ATTN_SKIP_SUBMODULE_INIT=1 python3 setup.py build_ext --inplace
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MLA_CONSUMER_SANITIZE=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_metadata_consumer.py tests/test_flash_attn_npu_v3_mla_build.py
ASCEND_RT_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 -m pytest -x -q tests/test_flash_attn_npu_v3_mla_metadata_dispatch.py tests/test_flash_attn_npu_v3_mla_dispatch.py tests/test_flash_attn_npu_v3_mla_kvcache.py
```

提交前检查：

```bash
PYTHONPATH=/tmp/mla-m01-precommit PRE_COMMIT_HOME=/tmp/mla-m01-precommit-cache bash ci/code_quality.sh --staged
python3 build/mla-m05c2-final/verify_acceptance.py
git diff --check
git diff --cached --check
```

当前闭环提交用 `git log -1 -- docs/mla/status.md` 定位。
