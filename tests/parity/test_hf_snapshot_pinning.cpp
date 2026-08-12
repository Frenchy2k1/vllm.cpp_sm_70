// vllm.cpp original. GATE-PIN-UNPINNED-SNAPSHOTS (issue #471): the resolver
// gate. Proves, on any box, with no GPU and no real checkpoint, that a pinned
// accessor SELECTS its revision out of a cache that also holds a decoy of the
// SAME repo, and REFUSES (returns "" -> the caller's loud skip) rather than
// substituting when only the decoy is present.
//
// Why this test exists rather than a run on the gate host. The three DFlash
// gates check `HasCuda()` BEFORE they resolve anything, so on a CPU box they can
// never demonstrate selection, and on the GPU box they only demonstrate it while
// the cache happens to be in the two-revision state. Synthesising the cache
// makes the hazardous state available on demand and makes the proof independent
// of what anyone has downloaded.
//
// The hazard being modelled is real and specific: `unsloth/Qwen3.6-27B-NVFP4`
// caches @890bdef7 (NVFP4 W4A4, bf16 GDN tower) and @ccdaab7e (the same repo
// name, silently re-quantized to FP8 W8A8). MEASURED on the gate host, readdir
// yields @890bdef7 first today -- so an unpinned resolver gets the right answer
// there RIGHT NOW and would get the wrong one after any reordering. A test that
// only ran on that host would therefore have passed either way. This one cannot.
#include <doctest/doctest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "hf_snapshot.h"

namespace fs = std::filesystem;

namespace {

// A scratch $HOME that is restored on scope exit, so one case's cache layout can
// never leak into another's.
class ScopedHome {
 public:
  explicit ScopedHome(const fs::path& root) : root_(root) {
    const char* prev = std::getenv("HOME");
    had_prev_ = prev != nullptr;
    if (had_prev_) prev_ = prev;
    setenv("HOME", root_.string().c_str(), /*overwrite=*/1);
  }
  ~ScopedHome() {
    if (had_prev_)
      setenv("HOME", prev_.c_str(), /*overwrite=*/1);
    else
      unsetenv("HOME");
    std::error_code ec;
    fs::remove_all(root_, ec);
  }

  ScopedHome(const ScopedHome&) = delete;
  ScopedHome& operator=(const ScopedHome&) = delete;

 private:
  fs::path root_;
  std::string prev_;
  bool had_prev_ = false;
};

// Same for an env override, so an unset variable stays unset afterwards.
class ScopedEnv {
 public:
  ScopedEnv(const char* name, const std::string& value) : name_(name) {
    const char* prev = std::getenv(name);
    had_prev_ = prev != nullptr;
    if (had_prev_) prev_ = prev;
    setenv(name_, value.c_str(), /*overwrite=*/1);
  }
  ~ScopedEnv() {
    if (had_prev_)
      setenv(name_, prev_.c_str(), /*overwrite=*/1);
    else
      unsetenv(name_);
  }

  ScopedEnv(const ScopedEnv&) = delete;
  ScopedEnv& operator=(const ScopedEnv&) = delete;

 private:
  const char* name_;
  std::string prev_;
  bool had_prev_ = false;
};

fs::path ScratchRoot(const std::string& tag) {
  const fs::path root =
      fs::temp_directory_path() / ("vt_hf_snapshot_pin_" + tag);
  std::error_code ec;
  fs::remove_all(root, ec);
  return root;
}

// Materialise `<home>/.cache/huggingface/hub/<repo>/snapshots/<rev>/config.json`.
// `config.json` is exactly what HfSnapshot probes for, so this is the minimum
// that makes a revision look cached.
fs::path PlantRevision(const fs::path& home, const std::string& repo,
                       const std::string& rev) {
  const fs::path dir =
      home / ".cache/huggingface/hub" / repo / "snapshots" / rev;
  fs::create_directories(dir);
  std::ofstream(dir / "config.json") << "{}\n";
  return dir;
}

// The real second revision of the real repo. Not a placeholder: this is the FP8
// re-quant that shares `unsloth/Qwen3.6-27B-NVFP4`'s name.
constexpr const char* kUnsloth27bRepo = "models--unsloth--Qwen3.6-27B-NVFP4";
constexpr const char* kUnsloth27bDecoyRevision =
    "ccdaab7e68af2409599b8949a8f2685703c9bae5";

}  // namespace

TEST_CASE("hf_snapshot: the pinned revision is selected out of a two-revision cache") {
  const fs::path home = ScratchRoot("select");
  ScopedHome guard(home);

  // Plant the decoy FIRST, so a resolver that took "whatever came back first"
  // from an insertion-ordered listing would pick the wrong one. Readdir order is
  // not insertion order, which is the whole point -- neither order may matter.
  PlantRevision(home, kUnsloth27bRepo, kUnsloth27bDecoyRevision);
  const fs::path want =
      PlantRevision(home, kUnsloth27bRepo, parity::kQwen27NvfP4Revision);

  const std::string got = parity::Qwen27NvfP4Snapshot();
  CHECK(got == want.string());
  // Stated separately from the equality above: an assertion that only compared
  // paths would still pass if BOTH constants were changed to the decoy.
  CHECK(got.find(parity::kQwen27NvfP4Revision) != std::string::npos);
  CHECK(got.find(kUnsloth27bDecoyRevision) == std::string::npos);
}

TEST_CASE("hf_snapshot: a cache holding only ANOTHER revision refuses, never substitutes") {
  const fs::path home = ScratchRoot("refuse");
  ScopedHome guard(home);

  PlantRevision(home, kUnsloth27bRepo, kUnsloth27bDecoyRevision);

  // The repo is cached. The pinned revision is not. The only correct answer is
  // "" -- which is what makes the caller emit its skip instead of gating the
  // wrong model. An unpinned `directory_iterator` returns the decoy here.
  CHECK(parity::Qwen27NvfP4Snapshot().empty());
}

TEST_CASE("hf_snapshot: every pinned accessor selects, refuses, and skips alike") {
  struct Arm {
    const char* name;
    const char* repo;
    const char* revision;
    std::string (*resolve)();
  };
  // Every accessor in the header. A new pin that forgets to appear here is
  // caught by test_snapshot_pins.py, which parses the header and this list.
  const Arm arms[] = {
      {"27B unsloth NVFP4", kUnsloth27bRepo, parity::kQwen27NvfP4Revision,
       &parity::Qwen27NvfP4Snapshot},
      {"35B-A3B nvidia NVFP4", "models--nvidia--Qwen3.6-35B-A3B-NVFP4",
       parity::kQwen36A3bNvfP4Revision, &parity::Qwen36A3bNvfP4Snapshot},
      {"27B z-lab DFlash draft", "models--z-lab--Qwen3.6-27B-DFlash",
       parity::kQwen27DFlashDraftRevision, &parity::Qwen27DFlashDraftSnapshot},
  };

  // A decoy revision that is not any real pin, used per-repo.
  const std::string decoy(40, 'a');

  for (const Arm& arm : arms) {
    CAPTURE(arm.name);

    {  // Nothing cached at all -> refuse.
      const fs::path home = ScratchRoot("empty");
      ScopedHome guard(home);
      CHECK(arm.resolve().empty());
    }
    {  // Only a decoy revision of the right repo -> refuse.
      const fs::path home = ScratchRoot("decoy");
      ScopedHome guard(home);
      PlantRevision(home, arm.repo, decoy);
      CHECK(arm.resolve().empty());
    }
    {  // Pinned revision present alongside the decoy -> select the pin.
      const fs::path home = ScratchRoot("both");
      ScopedHome guard(home);
      PlantRevision(home, arm.repo, decoy);
      const fs::path want = PlantRevision(home, arm.repo, arm.revision);
      CHECK(arm.resolve() == want.string());
    }
    {  // Directory present but no config.json -> refuse. An empty or partially
       // downloaded snapshot is not a checkpoint.
      const fs::path home = ScratchRoot("noconfig");
      ScopedHome guard(home);
      fs::create_directories(home / ".cache/huggingface/hub" / arm.repo /
                             "snapshots" / arm.revision);
      CHECK(arm.resolve().empty());
    }
  }
}

TEST_CASE("hf_snapshot: the env override is the ONLY way to gate another checkpoint") {
  const fs::path home = ScratchRoot("override");
  ScopedHome guard(home);

  // A deliberate different-checkpoint run: an explicit directory, named by a
  // human, outside the cache entirely.
  const fs::path elsewhere = home / "deliberate-checkpoint";
  fs::create_directories(elsewhere);
  std::ofstream(elsewhere / "config.json") << "{}\n";

  {
    ScopedEnv over("VT_QWEN27_SNAPSHOT", elsewhere.string());
    CHECK(parity::Qwen27NvfP4Snapshot() == elsewhere.string());
  }
  {
    // An override pointing somewhere without a config.json refuses rather than
    // falling back to the cache -- otherwise a typo'd override would silently
    // gate the pinned model and the run would be misattributed.
    PlantRevision(home, kUnsloth27bRepo, parity::kQwen27NvfP4Revision);
    ScopedEnv over("VT_QWEN27_SNAPSHOT", (home / "nonexistent").string());
    CHECK(parity::Qwen27NvfP4Snapshot().empty());
  }
  {
    // An empty override is not an override; the pin still applies.
    ScopedEnv over("VT_QWEN27_SNAPSHOT", "");
    CHECK(parity::Qwen27NvfP4Snapshot().find(parity::kQwen27NvfP4Revision) !=
          std::string::npos);
  }
}
