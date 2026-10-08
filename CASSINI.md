# Cassini fork maintenance

This repository keeps the application release separate from the upstream review:

- `v1.13.7-cassini`: upstream v1.13.7 plus the Parakeet v3 reference frontend,
  the `+cassini-parakeet-v3-reference-v1` runtime suffix, and regression tests.
- `fix/parakeet-v3-reference-frontend`: the same frontend and tests on current
  upstream `master`, without Cassini branding or packaging. The fork-only draft
  is https://github.com/codemyriad/sherpa-onnx/pull/1. Do not open an upstream PR
  until that review is complete.

Cassini consumes `codemyriad/sherpa-onnx-go-linux v1.13.7-cassini.4` with the
unmodified upstream `sherpa-onnx-go v1.13.7` wrapper. The wrapper fork is no
longer needed. All Go/C declarations match the native 1.13.7 release.
The patched amd64/arm64 binaries and Cassini's checksum-verified source builder
use commit `832bfe50d1e45929e47c9d6e7a65e8a00a855820`. Later commits on this
release branch add tests, formatting and documentation only. Build recipes and
binary checksums live in the Go Linux fork's `CASSINI.md` and `scripts/`.

The runtime suffix selects Cassini's matching boundary policy. An unbranded
current-master library still works in Cassini, but Cassini conservatively warns
and uses its standard boundary policy. Adoption of an upstream release will
need an explicit capability/version contract before removing that check.

Validation completed on Linux amd64:

- Both source branches pass `offline-stream-parakeet-test`: boundary/silence,
  repeatability, normalization, resampling and independent numerical reference.
- The current-master shared library, release package, stock v1.13.7 library and
  a fresh release source build each transcribed a public sample in Cassini.
- Cassini's full recorder suite and operator race tests pass.
- Cassini PR #312 has passed ARM64 image build and transcription smoke checks.

For any native behavior change, rerun the numerical tests and Cassini's actual
transcription checks, rebuild both CPU architectures, publish a new immutable
Go module tag, and update Cassini's module/source pins together. CUDA compilation
and GPU inference need separate validation; CPU success does not establish GPU
support. Do not rewrite published tags or modify the user's Go module cache.

## Nemotron-3-Diarization

The branch `feat/nemotron3-diarization-cassini` adds end-to-end speaker
diarization with NVIDIA Nemotron-3-Diarization (k2-fsa/sherpa-onnx#4006):

- The C++ runtime is a cherry-pick of the upstream contribution
  (`feat/nemotron3-diarization`). Upstream adds a `segmentation.sortformer`
  config, which changes the C struct layout.
- Cassini keeps the stock sherpa-onnx-go v1.13.7 wrapper. The C API accepts a
  Sortformer model in `Segmentation.Pyannote.Model` and detects it from the
  ONNX metadata. No embedding model or clustering config is needed. Pyannote
  models are unaffected.
- The runtime version gains `.nemotron-diarization-v1`. The existing
  `+cassini-parakeet-v3-reference-v1` marker is unchanged, so Contains checks
  keep working.

Validated on Linux amd64 only: parity with the transformers reference (max
probability difference 1.3e-5 on 227 s), the cache unit tests, and Go through
the stock wrapper with a locally built library. No binaries, Go module tag or
Cassini pin have been published for it.

## Nemotron interface version 2

The reviewed Nemotron implementation adds the scalar `num_frames` model input
and masks padded frames before the output convolution. It requires ONNX
metadata version 2 and reports `.nemotron-diarization-v2`. The stock v1.13.7
Go/C API shim remains unchanged: set `Segmentation.Pyannote.Model` to the
Nemotron model. The Parakeet frontend remains unchanged.

Backported from codemyriad/sherpa-onnx commit
`1a07f97662c8765784472c10d0c23ba24378f4cc`, including strict metadata parsing,
finite silence embeddings, overflow-safe cache updates and boundary handling.
Version-1 Nemotron exports require the older Cassini runtime.
