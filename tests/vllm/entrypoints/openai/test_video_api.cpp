// `/v1/videos` serving surface: request contract + job lifecycle.
//
// The library never spawns a process (developer-ratified: the ffmpeg invocation
// lives in examples/), so these two pieces ARE the endpoint's logic and both are
// pure, which is what makes them testable here.
#include "vllm/entrypoints/openai/video_api.h"

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <string>
#include <thread>
#include <vector>

using vllm::openai::ParseVideoRequest;
using vllm::openai::VideoJob;
using vllm::openai::VideoJobStatus;
using vllm::openai::VideoJobStatusJson;
using vllm::openai::VideoJobStore;
using vllm::openai::VideoRequest;

TEST_CASE("video api: request parsing applies H3 defaults and rejects bad input") {
  const VideoRequest minimal = ParseVideoRequest(R"({"prompt": "a cat"})");
  CHECK(minimal.prompt == "a cat");
  CHECK(minimal.task.empty());              // resolved later from partition + inputs
  CHECK(minimal.num_inference_steps == 50);  // H3's documented step count
  CHECK(minimal.flow_shift == doctest::Approx(12.0));       // video
  CHECK(minimal.audio_flow_shift == doctest::Approx(3.0));  // audio
  CHECK(minimal.duration_seconds == doctest::Approx(0.0));  // => per-task default
  CHECK_FALSE(minimal.has_seed);

  // vLLM-Omni nests the knobs under extra_params; both spellings must work.
  const VideoRequest nested = ParseVideoRequest(R"({
    "prompt": "a dog", "height": 768, "width": 1344,
    "extra_params": {"duration": 6.0, "num_inference_steps": 30,
                     "flow_shift": 9.5, "audio_flow_shift": 2.5, "seed": 1234}
  })");
  CHECK(nested.height == 768);
  CHECK(nested.width == 1344);
  CHECK(nested.duration_seconds == doctest::Approx(6.0));
  CHECK(nested.num_inference_steps == 30);
  CHECK(nested.flow_shift == doctest::Approx(9.5));
  CHECK(nested.audio_flow_shift == doctest::Approx(2.5));
  CHECK(nested.has_seed);
  CHECK(nested.seed == 1234);

  const VideoRequest flat =
      ParseVideoRequest(R"({"prompt": "x", "duration": 4.0, "num_inference_steps": 10})");
  CHECK(flat.duration_seconds == doctest::Approx(4.0));
  CHECK(flat.num_inference_steps == 10);

  // Malformed input must be a specific error, never a silent default.
  CHECK_THROWS(ParseVideoRequest("not json"));
  CHECK_THROWS(ParseVideoRequest("[]"));
  CHECK_THROWS(ParseVideoRequest(R"({})"));                        // no prompt
  CHECK_THROWS(ParseVideoRequest(R"({"prompt": 5})"));             // wrong type
  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","num_inference_steps":0})"));
  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","flow_shift":0})"));
  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","duration":-1})"));
  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","height":-8})"));
}

// ---------------------------------------------------------------------------
// OpenAI's Sora video shape (developers.openai.com/api/docs/guides/video-generation):
// {model, prompt, size:"<w>x<h>", seconds}. These are ALIASES
// onto the native fields, so an OpenAI client works unmodified while every
// existing body keeps its exact meaning.
//
// Every value below DIFFERS from the field's default, so a passing check proves
// the parser was reached rather than that a default happened to match.
// ---------------------------------------------------------------------------

TEST_CASE("video api: OpenAI `size` round-trips to width/height") {
  const VideoRequest r = ParseVideoRequest(R"({"prompt":"a cat","size":"1280x720"})");
  CHECK(r.width == 1280);   // default is 0
  CHECK(r.height == 720);   // default is 0
  // width x height, in that order: a portrait size must not come back landscape.
  const VideoRequest portrait = ParseVideoRequest(R"({"prompt":"x","size":"720x1280"})");
  CHECK(portrait.width == 720);
  CHECK(portrait.height == 1280);
  // The separator is case-insensitive; nothing else about the spelling is.
  const VideoRequest upper = ParseVideoRequest(R"({"prompt":"x","size":"1024X1792"})");
  CHECK(upper.width == 1024);
  CHECK(upper.height == 1792);

  // And the direct helper, so the geometry contract is gated without a body.
  int64_t w = 0, h = 0;
  vllm::openai::ParseVideoSize("1792x1024", &w, &h);
  CHECK(w == 1792);
  CHECK(h == 1024);
}

TEST_CASE("video api: a malformed `size` is a 400, never a silent default") {
  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","size":"1280"})"));       // no separator
  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","size":"1280x"})"));      // no height
  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","size":"x720"})"));       // no width
  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","size":"1280x720x2"})")); // two separators
  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","size":"1280 x 720"})")); // spaces
  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","size":"1280x720p"})"));  // trailing junk
  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","size":"-1280x720"})"));  // signed
  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","size":"0x720"})"));      // zero extent
  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","size":"1280.0x720"})")); // not whole
  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","size":""})"));           // empty
  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","size":720})"));          // not a string
  // An UNREADABLE size stays a 400 even when width/height would override it: the
  // client is never told the whole request was understood when half of it wasn't.
  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","size":"nonsense","width":8,"height":8})"));
}

TEST_CASE("video api: OpenAI `seconds` maps to the duration, as number or string") {
  const VideoRequest number = ParseVideoRequest(R"({"prompt":"x","seconds":8})");
  CHECK(number.duration_seconds == doctest::Approx(8.0));  // default is 0.0
  // OpenAI's schema types `seconds` as a STRING enum ("4"/"8"/"12"), so the
  // literal client shape must parse too.
  const VideoRequest text = ParseVideoRequest(R"({"prompt":"x","seconds":"12"})");
  CHECK(text.duration_seconds == doctest::Approx(12.0));
  const VideoRequest fractional = ParseVideoRequest(R"({"prompt":"x","seconds":"4.5"})");
  CHECK(fractional.duration_seconds == doctest::Approx(4.5));

  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","seconds":"soon"})"));   // not a number
  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","seconds":"8s"})"));     // trailing junk
  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","seconds":0})"));        // must be > 0
  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","seconds":-4})"));
  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","seconds":[8]})"));      // wrong type
  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","seconds":"inf"})"));    // stod takes it; we must not
}

TEST_CASE("video api: the native spelling WINS over the OpenAI alias") {
  // Documented precedence: `width`/`height` beat `size`, `duration` beats
  // `seconds`. That direction is what guarantees every body which parses today
  // keeps its exact meaning once the aliases exist.
  const VideoRequest both = ParseVideoRequest(R"({
    "prompt":"x", "size":"1280x720", "width":640, "height":480,
    "seconds":8, "duration":3.0
  })");
  CHECK(both.width == 640);
  CHECK(both.height == 480);
  CHECK(both.duration_seconds == doctest::Approx(3.0));

  // Per-axis, not all-or-nothing: an explicit width alone still lets `size`
  // supply the height it did not specify.
  const VideoRequest partial =
      ParseVideoRequest(R"({"prompt":"x","size":"1280x720","width":640})");
  CHECK(partial.width == 640);
  CHECK(partial.height == 720);

  // extra_params carries `duration` for vLLM-Omni clients; it wins there too.
  const VideoRequest nested =
      ParseVideoRequest(R"({"prompt":"x","seconds":8,"extra_params":{"duration":2.5}})");
  CHECK(nested.duration_seconds == doctest::Approx(2.5));
  // ... and with no `duration` anywhere, the top-level `seconds` applies even
  // though extra_params is present (an OpenAI field is never nested).
  const VideoRequest alias_only =
      ParseVideoRequest(R"({"prompt":"x","seconds":8,"extra_params":{"flow_shift":9.5}})");
  CHECK(alias_only.duration_seconds == doctest::Approx(8.0));
  CHECK(alias_only.flow_shift == doctest::Approx(9.5));
}

TEST_CASE("video api: `model` is recorded, never a parse failure") {
  const VideoRequest r = ParseVideoRequest(R"({"prompt":"x","model":"sora-2-pro"})");
  CHECK(r.model == "sora-2-pro");  // default is empty
  // Whether it names something this server serves is the ROUTE's call (a warning
  // on the job); the parser only refuses shapes it cannot record.
  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","model":7})"));
  CHECK_THROWS(ParseVideoRequest(R"({"prompt":"x","model":""})"));
  // Absent stays absent.
  CHECK(ParseVideoRequest(R"({"prompt":"x"})").model.empty());
}

TEST_CASE("video api: a whole OpenAI-shaped body parses, keeping our rich fields") {
  // The exact shape an unmodified OpenAI client sends, plus our native knobs
  // alongside: this is ADDITIVE compatibility, not a replacement.
  const VideoRequest r = ParseVideoRequest(R"({
    "model": "sora-2-pro",
    "prompt": "a cat on a skateboard",
    "size": "1280x720",
    "seconds": "8",
    "task": "t2va",
    "extra_params": {"num_frames": 97, "num_inference_steps": 30, "flow_shift": 9.5,
                     "audio_flow_shift": 2.5, "seed": 4242}
  })");
  CHECK(r.model == "sora-2-pro");
  CHECK(r.prompt == "a cat on a skateboard");
  CHECK(r.width == 1280);
  CHECK(r.height == 720);
  CHECK(r.duration_seconds == doctest::Approx(8.0));
  CHECK(r.task == "t2va");
  CHECK(r.num_frames == 97);
  CHECK(r.num_inference_steps == 30);   // default 50
  CHECK(r.flow_shift == doctest::Approx(9.5));        // default 12.0
  CHECK(r.audio_flow_shift == doctest::Approx(2.5));  // default 3.0
  CHECK(r.has_seed);
  CHECK(r.seed == 4242);
}

TEST_CASE("video api: the job store enforces its lifecycle") {
  VideoJobStore store;
  CHECK(store.Size() == 0);

  const std::string id = store.Create();
  CHECK(!id.empty());
  CHECK(store.Size() == 1);

  VideoJob job;
  REQUIRE(store.Get(id, &job));
  CHECK(job.id == id);
  CHECK(job.status == VideoJobStatus::kQueued);

  // An unknown id is reported, not invented -- so the route can 404.
  CHECK_FALSE(store.Get("vid_does_not_exist", &job));

  store.MarkRunning(id);
  REQUIRE(store.Get(id, &job));
  CHECK(job.status == VideoJobStatus::kRunning);

  store.MarkSucceeded(id, "/tmp/out.mp4");
  REQUIRE(store.Get(id, &job));
  CHECK(job.status == VideoJobStatus::kSucceeded);
  CHECK(job.output_path == "/tmp/out.mp4");

  // Illegal transitions throw rather than corrupting a finished record.
  CHECK_THROWS(store.MarkRunning(id));
  CHECK_THROWS(store.MarkSucceeded(id, "/tmp/other.mp4"));
  CHECK_THROWS(store.MarkFailed(id, "late"));
  CHECK_THROWS(store.MarkRunning("vid_nope"));

  // A succeeded job must carry a path; an empty one is a caller bug.
  const std::string second = store.Create();
  store.MarkRunning(second);
  CHECK_THROWS(store.MarkSucceeded(second, ""));
  store.MarkFailed(second, "ffmpeg exited 1");
  REQUIRE(store.Get(second, &job));
  CHECK(job.status == VideoJobStatus::kFailed);
  CHECK(job.error == "ffmpeg exited 1");

  // A queued job may fail directly (rejected before it ever ran).
  const std::string third = store.Create();
  store.MarkFailed(third, "");
  REQUIRE(store.Get(third, &job));
  CHECK(job.status == VideoJobStatus::kFailed);
  CHECK(job.error == "unspecified error");  // never an empty reason
}

TEST_CASE("video api: status JSON reports exactly what the client needs") {
  VideoJobStore store;
  const std::string id = store.Create();
  VideoJob job;
  REQUIRE(store.Get(id, &job));

  nlohmann::json queued = nlohmann::json::parse(VideoJobStatusJson(job));
  CHECK(queued.at("id") == id);
  CHECK(queued.at("status") == "queued");
  CHECK_FALSE(queued.contains("output_path"));  // nothing to hand back yet
  CHECK_FALSE(queued.contains("error"));

  store.MarkRunning(id);
  store.MarkSucceeded(id, "/tmp/a.mp4");
  REQUIRE(store.Get(id, &job));
  nlohmann::json done = nlohmann::json::parse(VideoJobStatusJson(job));
  CHECK(done.at("status") == "succeeded");
  CHECK(done.at("output_path") == "/tmp/a.mp4");
  CHECK_FALSE(done.contains("error"));

  const std::string bad = store.Create();
  store.MarkRunning(bad);
  store.MarkFailed(bad, "boom");
  REQUIRE(store.Get(bad, &job));
  nlohmann::json failed = nlohmann::json::parse(VideoJobStatusJson(job));
  CHECK(failed.at("status") == "failed");
  CHECK(failed.at("error") == "boom");
  CHECK_FALSE(failed.contains("output_path"));
}

TEST_CASE("video api: the requested model and its mismatch note ride the job") {
  VideoJobStore store;
  const std::string id = store.Create("sora-2-pro", "not a served model");
  VideoJob job;
  REQUIRE(store.Get(id, &job));
  CHECK(job.model == "sora-2-pro");
  CHECK(job.warning == "not a served model");

  // They survive to the terminal state, so a poller that only ever sees the
  // FINISHED record still learns the model it asked for was not the one used.
  store.MarkRunning(id);
  store.MarkSucceeded(id, "/tmp/a.mp4");
  REQUIRE(store.Get(id, &job));
  nlohmann::json done = nlohmann::json::parse(VideoJobStatusJson(job));
  CHECK(done.at("model") == "sora-2-pro");
  CHECK(done.at("warning") == "not a served model");

  // A job with neither omits both keys rather than emitting empty strings.
  VideoJob plain;
  REQUIRE(store.Get(store.Create(), &plain));
  nlohmann::json plain_json = nlohmann::json::parse(VideoJobStatusJson(plain));
  CHECK_FALSE(plain_json.contains("model"));
  CHECK_FALSE(plain_json.contains("warning"));
}

TEST_CASE("video api: the job store is safe under concurrent creation") {
  // The HTTP worker pool touches this from several threads, so ids must stay
  // unique and no record may be lost.
  VideoJobStore store;
  constexpr int kThreads = 8, kPerThread = 32;
  std::vector<std::thread> threads;
  std::vector<std::vector<std::string>> ids(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&store, &ids, t]() {
      for (int i = 0; i < kPerThread; ++i) ids[t].push_back(store.Create());
    });
  }
  for (std::thread& thread : threads) thread.join();

  CHECK(store.Size() == kThreads * kPerThread);
  std::vector<std::string> flat;
  for (const auto& per_thread : ids) flat.insert(flat.end(), per_thread.begin(), per_thread.end());
  std::sort(flat.begin(), flat.end());
  CHECK(std::adjacent_find(flat.begin(), flat.end()) == flat.end());  // all unique
}
