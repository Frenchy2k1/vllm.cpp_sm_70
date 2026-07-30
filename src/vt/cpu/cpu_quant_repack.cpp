// CIQ G7 repack-at-load — portable transform + eligibility (arch-independent).
//
// The q8_0 weight repack and the shared 4-row interleave. Both mirror
// llama.cpp @ 237ad9b96:
//   make_block_q8_0x4          repack.cpp:2725  (the 8-byte-chunk interleave)
//   repack_q8_0_to_q8_0_4_bl   repack.cpp:3480  (the whole-tensor loop)
// The weight repack (nrows_interleaved = 4, interleave_block = 8) is exactly
// what `ggml_repack_get_optimal_repack_type` selects for q8_0 on NEON + i8mm
// (repack.cpp:4683 -> q8_0_4x8_q8_0). It is a byte permutation only, so the
// resident buffer keeps the same total size and the quant math is unchanged.
#include "cpu_quant_repack.h"

#include <cstring>
#include <vector>

#include "vt/dtype.h"
#include "vt/quant.h"

namespace vt::cpu {

void InterleaveQ8_0Rows4(const BlockQ8_0* row0, const BlockQ8_0* row1,
                         const BlockQ8_0* row2, const BlockQ8_0* row3,
                         BlockQ8_0x4* dst, int64_t nblocks) {
  const BlockQ8_0* rows[4] = {row0, row1, row2, row3};
  for (int64_t x = 0; x < nblocks; ++x) {
    BlockQ8_0x4& out = dst[x];
    for (int i = 0; i < kQ8_0xNrowsInterleaved; ++i) {
      out.d[i] = rows[i][x].d;
    }
    // make_block_q8_0x4(in, blck_size_interleave = 8): 128/8 = 16 chunks,
    // chunk i takes 8 bytes from row (i % 4) at source offset (i / 4) * 8.
    const int end = kQK8_0 * kQ8_0xNrowsInterleaved / kQ8_0xInterleaveBlock;  // 16
    for (int i = 0; i < end; ++i) {
      const int src_id = i % kQ8_0xNrowsInterleaved;
      const int src_offset = (i / kQ8_0xNrowsInterleaved) * kQ8_0xInterleaveBlock;
      const int dst_offset = i * kQ8_0xInterleaveBlock;
      std::memcpy(&out.qs[dst_offset], &rows[src_id][x].qs[src_offset],
                  kQ8_0xInterleaveBlock);
    }
  }
}

bool QuantRepackEligible(DType weight_dtype, int64_t n, int64_t k) {
  if (!QuantRepackActive()) return false;
  // G7 repacks q8_0 only (the profile's kMatmulBTQuant is q8_0). The k-quants
  // keep the mmla tier; adding their repack layouts is a follow-up row.
  if (weight_dtype != DType::kQ8_0) return false;
  // repack_q8_0_to_q8_0_4_bl guard: ne[1] % 4 == 0 (N rows) and ne[0] % 8 == 0
  // (K); K % 32 == 0 is the whole-q8_0-block rule and subsumes K % 8.
  return n > 0 && k > 0 && n % kQ8_0xNrowsInterleaved == 0 && k % kQK8_0 == 0;
}

void QuantRepackWeight(DType weight_dtype, uint8_t* blocks, int64_t n,
                       int64_t k) {
  VT_CHECK(QuantRepackEligible(weight_dtype, n, k),
           "quant_repack_weight: weight is not repack-eligible");
  const int64_t nblocks = k / kQK8_0;
  const int64_t ngroups = n / kQ8_0xNrowsInterleaved;
  // The transform is not in-place-safe, so snapshot the plain rows first. Same
  // total size (block_q8_0x4 == 4 * block_q8_0), so the dest fits `blocks`.
  const size_t total = static_cast<size_t>(n) * nblocks * sizeof(BlockQ8_0);
  std::vector<uint8_t> src(blocks, blocks + total);
  const BlockQ8_0* srcb = reinterpret_cast<const BlockQ8_0*>(src.data());
  BlockQ8_0x4* dst = reinterpret_cast<BlockQ8_0x4*>(blocks);

  for (int64_t g = 0; g < ngroups; ++g) {
    const int64_t r0 = g * kQ8_0xNrowsInterleaved;
    InterleaveQ8_0Rows4(srcb + (r0 + 0) * nblocks, srcb + (r0 + 1) * nblocks,
                        srcb + (r0 + 2) * nblocks, srcb + (r0 + 3) * nblocks,
                        dst + g * nblocks, nblocks);
  }
}

// Brick 4 (DeepSeek-V4 last-mile): repack a Q8_0 weight into the CUDA
// coalesced-load layout. The in-place 34-byte block (`uint16 d + 32×int8 qs`)
// interleaves the scale into the qs stream and starts qs at offset 2 (2-byte
// aligned), which precludes wide aligned loads on the decode Q8_0 matvec. This
// DEINTERLEAVES the whole tensor into two contiguous sections — `[all qs | all
// scales]` — so every block's 32 qs sit at a 32-byte-aligned offset (b*32) and a
// warp lane reads them via aligned `int4` (128-bit) loads. Byte permutation only:
// same total size (n*nb*32 qs + n*nb*2 scales = n*nb*34), same scale + int8
// values, so the integer dot is BIT-IDENTICAL to the in-place BlockQ8_0 path.
// dst layout for global block index i = r*nb+b (r in [0,n), b in [0,nb)):
//   qs bytes  : blocks[ i*32 .. i*32+32 )
//   scale d   : *(uint16*)(blocks + n*nb*32 + i*2)
bool RepackQ8_0CudaEligible(DType weight_dtype, int64_t n, int64_t k) {
  return weight_dtype == DType::kQ8_0 && n > 0 && k > 0 && k % kQK8_0 == 0;
}

void RepackQ8_0Cuda(uint8_t* blocks, int64_t n, int64_t k) {
  VT_CHECK(RepackQ8_0CudaEligible(DType::kQ8_0, n, k),
           "repack_q8_0_cuda: weight is not Q8_0 / K not a whole number of blocks");
  const int64_t nb = k / kQK8_0;
  const int64_t nblk = n * nb;
  const size_t total = static_cast<size_t>(nblk) * sizeof(BlockQ8_0);  // n*nb*34
  // Not in-place-safe (the qs section overlaps the source interleave), so
  // snapshot the plain blocks first; the deinterleaved sections fit `blocks`.
  std::vector<uint8_t> src(blocks, blocks + total);
  const BlockQ8_0* sb = reinterpret_cast<const BlockQ8_0*>(src.data());
  uint8_t* qs = blocks;                                                    // [nblk*32]
  uint8_t* dd = blocks + static_cast<size_t>(nblk) * kQK8_0;               // [nblk*2]
  for (int64_t i = 0; i < nblk; ++i) {
    std::memcpy(qs + static_cast<size_t>(i) * kQK8_0, sb[i].qs, kQK8_0);
    std::memcpy(dd + static_cast<size_t>(i) * sizeof(uint16_t), &sb[i].d, sizeof(uint16_t));
  }
}

}  // namespace vt::cpu
