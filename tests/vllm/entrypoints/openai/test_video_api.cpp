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
