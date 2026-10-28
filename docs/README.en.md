# liquiforge (English quick reference)

A zero-dependency Python reference implementation of an LFM3-style **tri-hybrid**
omni-modal backbone.

- **Tri-hybrid block** = linear attention (global, O(n·d²)) + depthwise causal
  convolution (local) + sparse MoE (capacity, top-k routing), gated and residual.
- **Unified token space**: text BPE / image patches / audio STFT→mel→VQ all map into
  one shared embedding table via `(modality, local_id)`.
- **One base → five variants**: shared trunk + lightweight heads + LoRA (rank≈8).
- **Decision modality**: REINFORCE policy gradient with a deterministic reward oracle.

Run: `python -m liquiforge.cli doctor|bench|train|serve`. See the main README.
