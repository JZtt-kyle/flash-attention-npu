# MLA 迭代计划

本页只记录实施顺序、产物和完成条件。**非量化 M17 与已授权的 INT8 M18–M21 均已完成；R4 测试整理也已完成，当前无后续已授权步骤。** 用户于 2026-10-18 批准 M18–M21 的 INT8 扩展，无需各步重复确认；范围以 [design 3.4](design.md#34-int8-压缩-mla-扩展) 为准，当前能力、限制和证据以 [status](status.md) 为准。2026-10-19 最新授权为图模式性能暂不做要求，适用于当前 INT8 M19–M21；eager、图正确性及其他必要验收保持不变，详见 design 3.4。

## R4：公开 E2E 测试整理（已完成）

用户于 2026-10-19 要求按仓库原有测试风格，仅新增一个 MLA 测试文件，并确认保留全部公开功能 E2E，移除 MLA 专用单元测试和迁移验收工具。以 `d433904` 为基线，将公开 API 的数值、布局、mask/split、空值/stride/长度、metadata 复用、生命周期、参数拒绝、fake/compile 与图重放集中到 `tests/test_flash_attn_npu_v3_mla.py`；重复的迁移阶段检查合并为参数化用例。删除 `tests/mla/` 和旧兼容验收入口；清理公共 conftest、README、文档及无头脚本对旧工具的依赖。加入既有全量 CI 的文件选项，并对非 910B/C 设备跳过 MLA 测试。生产 Python 仅修正 docstring 的测试路径；生产逻辑、C++、构建和 CATLASS 不变。

实施前保存旧 E2E 收集清单和能力映射于仓库外工作根的 `mla-test-cleanup/`。新文件使用直接 pytest 与仓库公共比较规则，不依赖 CANN 源仓、组件 probe 或本机 `/tmp/mla-*` 夹具。公开数值必须有独立 reference，重用/图/编译一致性不能单独替代数值判定；INT8 的既有数学限制须保留说明，不能放宽比较阈值来消除失败。

必跑检查：旧驱动 dry-run/collect 核对日常基线；新文件 `python3 -m pytest --collect-only -q tests/test_flash_attn_npu_v3_mla.py`，对照保留/合并/移除的覆盖目的；先实跑双 dtype、两种表示、五布局、INT8、fake/compile/图的代表及长 KV 成本边界，再固定实现运行新文件完整集合及普通 FA3/图回归。使用 910B 设备 7 和已验收扩展，先核对 Python/torch/torch_npu/CANN、芯片、扩展 SHA256 与生产源码身份；不因纯测试整理重建未变的扩展。继承 M21 的生产数值与性能记录，仅作生产基线，本轮测试结果另记，旧节点通过不能冒充新节点执行。

完成条件已满足：单个 MLA 测试文件可直接 pytest 运行，无旧工具导入或失效文档链接；完整公开功能集合与受影响普通回归通过。覆盖映射见 design 第 7 节，最新命令、耗时和证据见 status。质量检查后只提交本步改动并停止。

## 已完成步骤

这里只保留交付边界；具体历史命令和测试工具在对应 Git 提交中。各布局步均包含完整调用链和必要验收；非量化拒绝项见 design 第 2 节，INT8 边界见 3.4，旧步骤的命令、数量、耗时、失败修正及证据从对应提交的 status 查询。

| 步骤 | 已交付产物 |
| --- | --- |
| M00–M03 | 固定迁移交集/FA3 契约、独立双 reference 与固定源执行器、入口识别/校验/未实现拦截 |
| M04–M06 | 512+64 dense 数值与非 PA BSND/TND/混合生产前向，v1–v5、allocation/used、完整 storage/stride 与生命周期 |
| M07 | ND PA 地址/Copy/调度、v6、BSND/TND PA 公开 metadata/cache 及阶段收尾 |
| M08–M09 | 五布局右下 causal 与 window，v7–v16、公开接入及各阶段收尾 |
| M10 | partial/FD 归并与真实 accumulator；五布局显式/原生自动 split，v17–v26、公开入口及组合/性能阶段收尾 |
| M11–M14 | 192/128 独立 K/V、GQA 数值组件及五布局完整公开前向，v27–v31；三 mask、全部 split、真实 accumulator、allocation/used、stride、页安全与组合收尾 |
| M15a–f | softcap 独立数值组件及两种表示、五布局完整前向，v32–v46；三 mask、全部 split、allocation/used、分页安全与公开性能 |
| M15g1–g2 | packed/unpacked GQA 坐标与资源组件、pack_gqa/sm_margin 公开接入，v47；修复共享 softcap 比较尾部 |
| M16–M17 | 固定/动态图重放、发射优化、910B/C 配置兼容、双 dtype 用户示例及非量化阶段收尾；图性能失败按当次明确授权接受，默认门槛未变 |
| M18–M21 | INT8 独立数值与 ND PA 适配、BSND/TND 公开前向、metadata/real/fake/compile/图、示例与阶段收尾；graph 性能按授权暂缓 |
| R4 | 公开功能 E2E 合并为单个普通 pytest 文件，退役 MLA 私有单元、固定源/Trap、性能和验收工具，更新现有 CI 与文档入口 |
| 工具整理、R1、R2 | 统一验收与缓存/证据管理、模块归属与共享 helper、共用前向分派、文档轮和无头交接；原 R3 的内核可读性核对合入 R2 |

非量化收尾为 `ebdf3eb`，M18 开发闭环为 `619787b`。组合映射见 [512+64](design.md#75-51264-组合覆盖)、[192/128](design.md#76-192128-组合覆盖) 和 [非量化收尾](design.md#77-非量化收尾映射)。历史查询：

```bash
git log --oneline -- docs/mla/status.md
git show ebdf3eb:docs/mla/status.md
git show 619787b:docs/mla/status.md
```

## 迭代与收尾规则

遵循 [AGENTS.md](../../AGENTS.md#一次工作只完成一个提交) 和 [design 第 7 节](design.md#7-验收口径)。实现前明确影响路径、必要用例、继承依据和完成条件；验证失败时保留交接，不提交为完成。测试整理后的入口为普通 pytest，只有一个 MLA E2E 文件，不恢复迁移阶段的独立测试框架。每成功完成五轮开发插入专门文档轮；文档轮不计开发轮数。一次提交完成后立即停止。
