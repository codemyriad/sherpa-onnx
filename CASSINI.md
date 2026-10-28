# Cassini fork of sherpa-onnx

This branch, `v2.0.1-cassini`, is the native speech runtime that Cassini
(codemyriad/gocassini) ships. It is upstream sherpa-onnx **v2.0.1**
(`af2d8d81dbfa61c4cea4c568da78f3982edca005`) plus three commits. Releases are
annotated tags `v2.0.1-cassini-N` on this branch; tags are never moved or
rewritten. A fix is a new commit and a new `N`.

It replaces the `v1.13.7-cassini` branch (tags `v1.13.7-cassini-nemotron-v2`,
`v1.13.7-cassini-parakeet-ultra-v1`), which stays as it was for the
`sherpa-onnx-go-* v1.13.7-cassini.*` modules.

## Patches on top of v2.0.1

1. **Nemotron-3-Diarization (Sortformer) speaker diarization,
   k2-fsa/sherpa-onnx#4032.** The PR's net diff at head
   `6a4ad68b2cf4c69b684c040db1a5f7227675c93b`, taken against its merge base with
   upstream master `bfbc6ec2306a3cffd1817a32d165f9b4eb78a284`, squashed into one
   commit (72 files, +3586/-22). The PR is open upstream. Drop this commit when
   the fork moves to an upstream release that contains it.
   - Models: interface version 2 only (`model_type=nemotron3_diarization`,
     `version=2`; inputs `features`, `cached_embeds`, `num_frames`; outputs
     `probs`, `chunk_embeds`). Cassini's `nemotron-3-diarization-int8` and fp32
     exports on dist.gocassini.com are version 2.
   - API: the model goes in **`segmentation.sortformer.model`** (C:
     `SherpaOnnxOfflineSpeakerSegmentationModelConfig.sortformer`, Go:
     `Segmentation.Sortformer.Model`). The 1.13.7 fork accepted it in
     `segmentation.pyannote.model`; that shim is gone. The new member is at the
     end of the C struct, so a Go/C wrapper must come from this tree
     (`scripts/go/sherpa_onnx.go`, `sherpa-onnx/c-api/c-api.h`).
2. **TDT detection from the joiner output size.** `InitJoiner` in
   `offline-transducer-nemo-model.cc` treats a NeMo joiner with more outputs
   than the vocabulary as TDT, even when the encoder `url` metadata does not
   contain `tdt`. Without it the production Parakeet Ultra bundles
   (`url = https://huggingface.co/moondream/parakeet-ultra`) exit at load with
   `vocab_size: 8193 != output_size: 8198`. Model-agnostic; no URL list.
   Unit test: `offline-transducer-nemo-joiner-test`. Meant to go upstream on its
   own, after which this commit is dropped.
3. **Version marker and this file.**

Not carried any more (compared with `v1.13.7-cassini`):

- The Parakeet "reference frontend" (centred Hann window, global preemphasis,
  Nyquist mel edge, additive log guard, N-1 variance) and its
  `offline-stream-parakeet-test`. Upstream #3999 (in v2.0.1) switches NeMo
  models with `normalize_type` metadata to an additive log guard, which fixes
  quiet audio (the -60 dB probe decodes in full). On Ultra int8 greedy it
  measured equal or better than the reference frontend
  (parakeet-ultra-eval `reports/fork-update-2026-10-28.md` §3).
- The list of Moondream Ultra URLs that selected that frontend and TDT.
- The Sortformer-through-Pyannote C API shim.
- The old Nemotron commits (#4032 contains them; native code is identical).

## Runtime version string

`GetVersionStr()` / `SherpaOnnxGetVersionStr()` / Go `sherpa.GetVersion()`:

```
2.0.1+cassini.parakeet-tdt-joiner-v1.nemotron-diarization-v2
```

Cassini matches markers with substring checks, so each marker is a stable
contract. Its meaning:

| Marker | Meaning |
|---|---|
| `2.0.1` (prefix) | Upstream base. Implies #3999 (NeMo additive log guard), also in stock v2.0.1 libraries. |
| `+cassini.` | A Cassini build of this branch, so also the #4032 C API layout. Carries no feature check by itself. |
| `.parakeet-tdt-joiner-v1` | NeMo TDT is detected from the joiner output size (patch 2): plain-URL Parakeet Ultra exports load and decode as TDT. |
| `.nemotron-diarization-v2` | Sortformer diarization with interface-version-2 Nemotron models is available. Unchanged from the 1.13.7 fork, so existing checks keep working. On this base the model is passed through `segmentation.sortformer`, not the Pyannote slot; a caller that may meet both bases must also check the version prefix. |

Removed on purpose: `+cassini-parakeet-v3-reference-v1` and
`.parakeet-ultra-v1`. Both claimed the Parakeet reference frontend, which this
branch no longer has. Code that keyed a "reference" boundary policy on them
must move to a different test (for example, the `2.0.1` base, which every
v2.0.1 library including stock Linux arm32 satisfies, since #3999 is
upstream). `GetGitSha1()` keeps upstream's release value.

## Reproduce the source

```sh
git clone https://github.com/k2-fsa/sherpa-onnx.git && cd sherpa-onnx
git fetch origin pull/4032/head:pr4032          # head 6a4ad68b
git switch -c v2.0.1-cassini v2.0.1             # af2d8d81
git diff bfbc6ec2 6a4ad68b | git apply --3way --index
git commit -m 'Add Nemotron-3-Diarization (Sortformer) speaker diarization (k2-fsa/sherpa-onnx#4032)'
git fetch https://github.com/codemyriad/sherpa-onnx.git v2.0.1-cassini-1
git diff --stat HEAD FETCH_HEAD   # only patches 2 and 3 remain
```

The tree after the `git apply` step is identical to this branch's first
commit, `a3ea1fce` (`git diff --quiet a3ea1fce` exits 0). Patch 2 is
`4dad81fd`.

## Build (Linux, CPU)

The release libraries use the CMake options of gocassini's
`deployment/sherpa/build.sh` (shared libraries, C API, diarization on, TTS
off). To also run the native tests:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 \
  -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
  -DCMAKE_INSTALL_RPATH='$ORIGIN' \
  -DSHERPA_ONNX_ENABLE_GPU=OFF -DSHERPA_ONNX_ENABLE_C_API=ON \
  -DSHERPA_ONNX_ENABLE_BINARY=ON -DSHERPA_ONNX_ENABLE_TESTS=ON \
  -DSHERPA_ONNX_ENABLE_PYTHON=OFF -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
  -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF -DSHERPA_ONNX_ENABLE_TTS=OFF \
  -DSHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=ON \
  -DSHERPA_ONNX_BUILD_C_API_EXAMPLES=OFF
cmake --build build --parallel 4
(cd build && ctest --output-on-failure)
```

v2.0.1 pins ONNX Runtime **1.28.2** (1.27.1 on the 1.13.7 fork); the CUDA
build uses `onnxruntime-linux-x64-gpu_cuda12-1.28.2-patched.zip`
(`cmake/onnxruntime-linux-x86_64-gpu.cmake`). CPU success does not establish
CUDA support; a CUDA build needs its own probe on a GPU.

## Validation of v2.0.1-cassini-1

Linux x86_64, CPU, built with the options above (GCC 14.2, `-j4`), ONNX
Runtime 1.28.2 (`libonnxruntime.so` sha256 `4b3607ae…`, the same file as the
upstream `sherpa-onnx-go-linux/v2 v2.0.1` module):

- **Native tests:** `ctest` 30/30 pass, including
  `offline-transducer-nemo-joiner-test` and `sortformer-speaker-cache-test`.
- **Quiet-audio probe:** public `en.wav` (3.85 s) at gain 0.001 (-60 dB) and
  1, greedy, through the Go C API. Cassini production Parakeet v3 int8 bundle
  and production Parakeet Ultra int8 bundle (encoder
  `url = https://huggingface.co/moondream/parakeet-ultra`, which stock v2.0.1
  rejects with exit 255): 27 tokens and the full sentence in all four runs.
  Tokens and timestamps are identical to the research candidate of the same
  stack. `sherpa-onnx-offline --debug=1` on the Ultra bundle logs
  `TDT model. vocab_size: 8193, num_durations: 5`.
- **Sortformer diarization:** `nemotron-3-diarization-int8` from
  dist.gocassini.com (version 2, sha256 `47c221ea…`) through
  `segmentation.sortformer`, from Cassini's Go `speakers diarize` and from
  `sherpa-onnx-offline-speaker-diarization --segmentation.sortformer-model`.
  On a 24 s clip (4 turns) and a 21.4 min meeting track (31 turns) the turns
  are identical to the 1.13.7 fork (`sherpa-onnx-go-linux v1.13.7-cassini.8`).

Not validated here: CUDA, macOS, Windows, Linux arm64, the fp32 Ultra and fp32
diarization models, and beam search. Those belong to the Go platform modules
and the CUDA image.

## Downstream

- Go platform modules `codemyriad/sherpa-onnx-go-{linux,macos,windows}`: branch
  `v2.0.1-cassini`, tags `v2.0.1-cassini.N`, module path
  `github.com/k2-fsa/sherpa-onnx-go-<os>/v2`, Go source copied from this tree.
  Each must report the version string above.
- gocassini `deployment/sherpa/build.sh` pins this branch's tag commit and
  codeload tarball sha256 for the CUDA image.

When the native code changes: rerun the native tests, the quiet-audio probe
(Parakeet v3 int8 and plain-URL Ultra int8), and a Sortformer diarization run;
then publish a new tag and update every pin together.
