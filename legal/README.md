# Legal files

[English](./README.md) | [中文](./README.zh.md)

This directory groups the repository-wide third-party license inventory.

- The canonical first-party [Apache License 2.0](../LICENSE) and
  [NOTICE](../NOTICE) remain at the repository root so source archives,
  packaging tools, and hosting services can discover them.
- [Third-party notices](./THIRD_PARTY_NOTICES.md) records external source,
  runtimes, models, services, and assets used by the project.
- The independently installable material package keeps its own
  [license](../tools/qwen_material_pipeline/LICENSE).
- Vendored source keeps its license beside the corresponding code, including
  [MVInverse](../tools/qwen_material_pipeline/third_party/mvinverse/LICENSE) and
  its [DINOv2 notice](../tools/qwen_material_pipeline/third_party/mvinverse/DINOV2_LICENSE).

Do not consolidate or delete a license that belongs to an independently built
package or a vendored dependency. Those files define separate distribution
boundaries even when their text duplicates a root-level license.
