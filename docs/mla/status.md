# MLA 当前状态

**M15e（BSND Q + ND PA KV softcap）已完成，必要验收及适用质量检查通过。唯一下一步是 [M15f：TND PA softcap](plan.md#唯一下一步-m15f-tnd-pa-softcap)，本轮不启动。** 本轮基线为 `feat/mla-cann-migration` 的 `f10ce93`，接手时工作树干净；这是专门文档轮之后的第一个开发轮。

## 当前能力与边界

- 两种表示（512+64、192/128）的 BSND PA 正 cap 接通 host/AICPU、Kernel/launcher、metadata、公开 kvcache/raw C++ 和 real/fake/compile；三 mask 与单/显式/自动 split 同步支持。v41/v42/v43 沿用既有分页字段和调度，只绑定 softcap 启用状态；正 cap 改值可复用记录。页表/cache 只读，独立 stride、乱序/重复页及全部有效页检查保留。
- TND PA 正 cap、非默认 `pack_gqa` / `sm_margin` 仍拒绝；完整 graph replay 未验收。生产平台仍只接受 Ascend910B，910C 实现与资源兼容留待后续，实机目标继续为 910B。
- 新增所属 CPU 单元、公开专项及日常 E2E，代表/日常集合包含 None/整数 KV 长度。复用源执行器、数学缓存及性能协议；prefill 汇总器补充指定组汇总，默认五布局阶段门槛不变。四个 FA3 公开函数的签名与默认值不变。

## 最近一次验收证据

证据根 `/tmp/mla-m15e/`；以下均为修复后最终来源的实际执行，未继承 M15d E2E、历史单元或性能。21 个组全部通过，无跳过或失败；通过统一驱动现有 `suite_sources`、`suite_dependencies`、`validate_result`、`reusable` 和构建 manifest 校验依赖、用例、实际加载及原始证据，记录在 `evidence-check-completed.log`、`evidence-check-compressed.log`，没有另写整树审计。最终集合数学缓存记录 1,352 hit、83 miss/write_ok；待测实现均实际执行。

| summary / attempt | 本次结果与耗时 |
| --- | --- |
| `cpu-final/summary.json` / `1792289262480209760` | 模块工具 45 passed / 34.1 s；prefill 汇总工具 208 passed / 111.0 s；CPU 112 passed / 97.6 s；ASan/UBSan 112 passed / 309.9 s |
| `device-final/summary.json` / `1792289391805695740` | 代表 48 passed / 36.1 s；完整专项 198 passed / 412.5 s（含 24 项隔离 Trap）；固定源 36 passed / 703.4 s；累计 E2E 647 passed / 125.3 s；ordinary 508 passed / 26.0 s |
| 同上，性能准备及正 cap 采样 | native BSND/TND prefill 各 1 passed / 207.1、204.7 s；正 cap 夹具 4 passed / 111.8 s；正 cap 正式采样 4 passed / 122.2 s |
| `perf-prefill/summary.json` / `1792291411690061790` | BSND/TND PA 各 1 passed / 246.3、252.4 s |
| `perf-compressed/summary.json` / `1792291944434092400` | BSND/TND PA 单片各 1 passed / 515.7、519.2 s；显式各 1 passed / 273.4、272.1 s；自动各 1 passed / 240.8、239.0 s |

固定源 76 份 O/LSE 对照逐元素一致且通过数学阈值，无新增源精度限制；显式 S=256 的原生 FD 最终归并不支持，只比较支持的 partial 并验收独立数学，不记为完整源归并通过。固定源六个含探针构建的首用例合计 545.7 s；压缩性能单片组也包含探针准备及缺失 reference 的首次计算，没有删减配置或重复计算同一 reference。旧全矩阵/历史单元与完整阶段性能未运行，不作为本轮通过项。

性能采样前固定 70 配置（含 dtype），R=1/4、预热 5、wall 5×20、kernel 5×10 不变。两个指定组汇总都通过，scope 为 `selected-groups`，不是 M10/M14 整阶段收尾。每个配置内入口/R 等权，再按配置等权；所有单例、分段、FD 排除项及原始样本保留在 summary 绑定的报告中。

| 性能结果 | 实测 / 适用 / FD 单列 | 平均公开耗时比与退化 |
| --- | --- | --- |
| `prefill-performance.json` | 24 / 12 / 12 | 0.776170 ≤ 1；0 配置均值、17 调用分段、24/24 kernel 对照退化 |
| `compressed-performance.json` | 36 / 24 / 12 | 0.580926 ≤ 1；4 配置均值、24 调用分段、36/36 kernel 对照退化 |
| 正 cap 的 `device-final/.../softcap-paged-bsnd-profile/*/profile.json` | 10 / 无同义 CANN / 不适用 | 正 cap/cap=0 平均开销比 1.057139，7/10 配置高于 1；仅作开销参照，不纳入同义源门槛 |

各集合均先 dry-run/collect-only，设备使用卡 7。复现命令：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export MLA_NATIVE_ROOT=/tmp/mla-m13b/native
python3 -m tests.mla --device 7 --suite softcap-paged-bsnd-tools prefill-stage-tools unit-softcap_paged_bsnd softcap-paged-bsnd-sanitize --output /tmp/mla-m15e/cpu-final
python3 -m tests.mla --device 7 --suite softcap-paged-bsnd-check detail-softcap_paged_bsnd prefill-paged-bsnd-profile-check prefill-paged-tnd-profile-check softcap-paged-bsnd-profile-check softcap-paged-bsnd-source e2e ordinary softcap-paged-bsnd-profile --output /tmp/mla-m15e/device-final
python3 -m tests.mla --device 7 --prefill-stage-performance --suite prefill-paged-bsnd-profile prefill-paged-tnd-profile --output /tmp/mla-m15e/perf-prefill
python3 -m tests.mla --device 7 --split-stage-performance --suite window-paged-profile window-paged-tnd-profile split-paged-bsnd-profile split-paged-tnd-profile auto-split-paged-bsnd-profile auto-split-paged-tnd-profile --output /tmp/mla-m15e/perf-compressed
python3 -m tests.mla.support.mla_prefill_stage_performance --evidence /tmp/mla-m15e/perf-prefill/summary.json --suite prefill-paged-bsnd-profile prefill-paged-tnd-profile --output /tmp/mla-m15e/prefill-performance.json
python3 -m tests.mla.support.mla_split_stage_performance --evidence /tmp/mla-m15e/perf-compressed/summary.json --suite window-paged-profile window-paged-tnd-profile split-paged-bsnd-profile split-paged-tnd-profile auto-split-paged-bsnd-profile auto-split-paged-tnd-profile --output /tmp/mla-m15e/compressed-performance.json
```

## 已处理失败

- 普通沙箱未映射 NPU 驱动，初次收集失败；设备环境重新收集成功。首次生产编译/链接成功，但准备期间补充汇总工具触发 `Source changed during preparation`，该尝试失败（558.9 s）；固定来源后按 manifest 核对有效构建。最终设备命令构建复核 1.3 s、实际加载检查 21.2 s。
- 正 cap 性能预检曾因符号名替换未维护 C++ ABI 名称长度而失败；改为直接校验实际 BSND PA 符号，增加成功/失败夹具，最终预检和正式采样均通过。原始尝试保留在 `preflight/`。
- 首次专项在 162 passed 后，单片正 cap + 整数 KV 长度 + 显式 metadata 失败：Python 整数便利分支漏传特征，误按旧记录长度预检。已补传并加入代表/日常集合，最终代表、完整专项与累计 E2E 均通过。原始失败在 `final/attempt-1792288937228803260/`，不计为最终通过。

## 环境与收尾

Python 3.11.15、torch 2.7.1+cpu、torch_npu 2.7.1、CANN `/usr/local/Ascend/cann-9.0.0-beta.2`；卡 7 Ascend910B3，卡 6 有其他负载。CATLASS 为 `769cd40a8716b28650b6bebb08db4834eea4462f`，固定 CANN 来源为 `ca18aee610729ca5270a44e2d84c751bc687fbf0`，两者未修改。

实际扩展为本仓 `flash_attn_npu_3/flash_attn_npu_3.cpython-311-aarch64-linux-gnu.so`，SHA256 `e71297ad6f35f739acf44d899d5a1fad7ec3f0a5c8a9d79883a654729233a3fd`。默认工作根 `/tmp/flash-attention-npu-mla-46928a793876/` 保存构建 manifest、探针与数学缓存；有效 native 包为 `/tmp/mla-m13b/native`，配方及文件校验通过。`build/` 仅保留无头循环日志和状态。上述 `/tmp` 路径非持久存储，缺失或内容变化后不得继承。

本次文档契约、32 处改动链接/锚点、差异及 `git diff --check` 已核对；43 个本步文件通过仓库质量入口 `PYTHONPATH=/tmp/mla-m10c8a/quality ci/code_quality.sh --staged`，格式化工具未改写源码。记录在 `/tmp/mla-m15e/docs-check.log` 与 `quality.log`。提交一次后立即停止，不开始 M15f。当前闭环提交可用 `git log -1 -- docs/mla/status.md` 定位；历史从 Git 查询。
