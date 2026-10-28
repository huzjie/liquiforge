# Changelog

All notable changes to liquiforge are documented here.

## [0.1.0] - 2026-10-28

### Added
- Tri-hybrid omni backbone (linear attention + depthwise conv + sparse MoE).
- Unified tokenization: byte-level BPE / vision patches / audio STFT-mel-VQ.
- Five task heads (text / vision / audio / omni / decision) + LoRA adapters.
- Decision-head policy-gradient RL with a deterministic reward oracle.
- Deterministic skill-driven training + full benchmark across five modalities.
- Zero-dependency stdlib HTTP serving, Docker/K8s/Helm/CI packaging.
