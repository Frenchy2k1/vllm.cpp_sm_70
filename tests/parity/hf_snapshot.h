#ifndef VLLM_TESTS_PARITY_HF_SNAPSHOT_H_
#define VLLM_TESTS_PARITY_HF_SNAPSHOT_H_

// Resolving a Hugging Face cache snapshot for a checkpoint-gated test.
//
// Every one of these gates used to take the FIRST entry `directory_iterator`
// yielded under `<repo>/snapshots/`. That is only safe while a repo has exactly
// one cached revision, and `unsloth/Qwen3.6-27B-NVFP4` does not:
//
//   @890bdef7  genuine NVFP4 - `weight_packed` U8 + `weight_scale` F8_E4M3 +
//              `weight_global_scale` F32. Every committed 27B golden was
//              captured against it (see the `oracle.model` field of
//              tests/parity/goldens/qwen36_*_27b/manifest.json).
//   @ccdaab7e  the SAME repo name, silently re-quantized to FP8 W8A8
//              throughout, with every NVFP4-specific `*_global_scale` tensor
//              gone.
//
// So the filesystem decided which model the SACRED gate measured, and a
// token-exact pass against an FP8 model would have been recorded as an NVFP4
// pass. Publishers re-quantize in place; a correctness gate must name the
// revision its golden belongs to.

#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

namespace parity {

// The revision the committed 27B goldens were captured against.
inline constexpr const char* kQwen27NvfP4Revision =
    "890bdef7a42feba6d83b6e17a03315c694112f2a";

// Snapshot directory for `<repo>` at `revision`, or "" when it is not cached
// (the caller then emits its loud SKIP). `env_override`, when set and non-empty,
// names an explicit snapshot directory for a deliberate different-checkpoint
// run and is the ONLY way to gate a revision other than the pinned one -- a
// cache holding some other revision skips rather than silently substituting it.
inline std::string HfSnapshot(const char* repo_dir, const char* revision,
                              const char* env_override) {
  namespace fs = std::filesystem;
  std::error_code ec;
  if (env_override != nullptr) {
    const char* over = std::getenv(env_override);
    if (over != nullptr && *over != '\0') {
      if (fs::exists(fs::path(over) / "config.json", ec)) return over;
      return "";
    }
  }
  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";
  const fs::path snap = fs::path(home) / ".cache/huggingface/hub" / repo_dir /
                        "snapshots" / revision;
  if (!fs::exists(snap / "config.json", ec)) return "";
  return snap.string();
}

// The 27B NVFP4 gate model, pinned to the goldens' revision.
inline std::string Qwen27NvfP4Snapshot() {
  return HfSnapshot("models--unsloth--Qwen3.6-27B-NVFP4",
                    kQwen27NvfP4Revision, "VT_QWEN27_SNAPSHOT");
}

// GATE-27B-FP8-TOWER-GOLDEN (issue #466). The OTHER dense 27B -- a different
// model, not a second spelling of the one above. `nvidia/Qwen3.6-27B-NVFP4` is
// `modelopt_mixed`: a W4A16 NVFP4 MLP plus an FP8 W8A8 tower
// (`linear_attn.in_proj_qkv`, `in_proj_z`, `out_proj`, and every
// `self_attn.{q,k,v,o}_proj`) and an NVFP4 head.
//
// It is the checkpoint every fp8-tower lever targets and every recorded 27B
// performance ratio was taken on, and until #466 nothing gated it
// token-for-token. The unsloth revision above lists
// `linear_attn.in_proj_{qkv,z,a,b}` in its `ignore` set and ships ZERO
// `*.input_scale` tensors, so there the fp8 arm is not merely differently tuned
// -- it cannot execute at all, and a gate pinned there can never fail for an
// fp8 defect.
//
// Deliberately NOT named `kQwen27NvfP4Revision<suffix>`:
// tests/tools/test_online_gate_server_binary.py parses that exact identifier
// out of this header and asserts a SINGLE pin equal to
// MODEL_GATE_CONTRACTS["test_qwen27_paged_engine"]["golden_revision"]. That
// assertion is correct and is left untouched.
inline constexpr const char* kQwen27nFp8TowerRevision =
    "0893e1606ff3d5f97a441f405d5fc541a6bdf404";

// The FP8-tower 27B gate model, pinned to ITS goldens' revision
// (tests/parity/goldens/qwen36_*_27n). An absent cache yields "" and the caller
// emits its loud refusal; a cache holding a DIFFERENT revision of the same repo
// skips rather than being substituted, exactly as above.
inline std::string Qwen27nFp8TowerSnapshot() {
  return HfSnapshot("models--nvidia--Qwen3.6-27B-NVFP4",
                    kQwen27nFp8TowerRevision, "VT_QWEN27N_SNAPSHOT");
}

}  // namespace parity

#endif  // VLLM_TESTS_PARITY_HF_SNAPSHOT_H_
