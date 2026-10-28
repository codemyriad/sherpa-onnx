// sherpa-onnx/csrc/offline-transducer-nemo-joiner-test.cc
//
// Copyright (c)  2026  Cassini contributors

#include "gtest/gtest.h"
#include "sherpa-onnx/csrc/offline-transducer-nemo-model.h"

namespace sherpa_onnx {

// vocab_size includes the blank token, as in OfflineTransducerNeMoModel.

TEST(ClassifyNeMoJoiner, PlainTransducer) {
  // e.g., parakeet-rnnt / parakeet-ctc-hybrid transducer heads
  EXPECT_EQ(ClassifyNeMoJoiner(false, 1025, 1025), NeMoJoinerKind::kTransducer);
}

TEST(ClassifyNeMoJoiner, TDTNamedByUrl) {
  // nvidia/parakeet-tdt-0.6b-v3: 8192 tokens + blank, 5 durations
  EXPECT_EQ(ClassifyNeMoJoiner(true, 8193, 8198), NeMoJoinerKind::kTDT);
}

TEST(ClassifyNeMoJoiner, TDTDetectedFromJoinerOutput) {
  // https://huggingface.co/moondream/parakeet-ultra: the url does not
  // contain "tdt", but the joiner has 5 duration logits. Before this
  // detection the model exited with "vocab_size: 8193 != output_size: 8198".
  EXPECT_EQ(ClassifyNeMoJoiner(false, 8193, 8198), NeMoJoinerKind::kTDT);
  EXPECT_EQ(ClassifyNeMoJoiner(false, 1025, 1026), NeMoJoinerKind::kTDT);
}

TEST(ClassifyNeMoJoiner, UrlSaysTDTButJoinerHasNoDurations) {
  EXPECT_EQ(ClassifyNeMoJoiner(true, 8193, 8193), NeMoJoinerKind::kInvalid);
}

TEST(ClassifyNeMoJoiner, JoinerSmallerThanVocabulary) {
  EXPECT_EQ(ClassifyNeMoJoiner(false, 8193, 8192), NeMoJoinerKind::kInvalid);
}

}  // namespace sherpa_onnx
