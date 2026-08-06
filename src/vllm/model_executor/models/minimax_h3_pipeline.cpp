// MiniMax-H3 t2va pipeline assembly — the path from prompt embeddings to frames
// and a waveform.
//
// Everything this file calls was ported and gated separately; W6 is the WIRING:
//
//   prompt_embeds ──► packed sequence (fl2va/t2va layout)
//                     │
//                     ├─► denoise loop: 50 x DiT forward + euler-eta0 step
//                     │      (sigmas from the rectified-flow time-shift schedule)
//                     ▼
//              video rows / audio rows
//                     │
//        unpatchify ──┤── unpack audio
//                     ▼
//        video latent ──► video VAE ViT3D decoder ──► frames  [C, T, H, W]
//        audio latent ──► audio VAE BigVGAN       ──► waveform per channel
//
// Latents are DENORMALIZED before decode (`latent * std + mean`), mirroring
// vae.py:252-270 / :341-357.
//
// NOISE IS AN INPUT, deliberately. Upstream seeds a torch CPU generator
// (pipeline_minimax_h3.py:813-843); reproducing torch's RNG bit-exactly is a
// separate concern that decides WHICH sample you get, not whether the pipeline is
// correct, so the caller supplies the initial rows. See the spec's open items.
#include "vllm/model_executor/models/minimax_h3.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"

namespace vllm {

// Turn supplied KEYFRAME IMAGES into the packed conditioning rows the denoise loop
// pins. This is the one step that stands between "the fl2va primitives are ported"
// and "a caller can pass a first frame".
//
// Each image is [3, H, W] in [-1, 1] and is encoded as a ONE-FRAME clip: the 3D CNN
// is causal in time, so a single frame is a valid input and yields latent_t 1.
// Rows are then patchified with the DiT's own patch sizes, which is what makes them
// interchangeable with the video rows the loop carries.
//
// `noise_aug` blends toward the supplied noise (1.0 pins the frame exactly). Noise
// is an INPUT for the same reason it is in the t2va path: reproducing upstream's RNG
// decides WHICH sample you get, not whether the pipeline is right.
void MiniMaxH3VideoDenormalizePixels(std::vector<float>& frames, int64_t channels,
                                     int64_t per_channel) {
  VT_CHECK(channels == 3, "minimax_h3: ImageNet de-normalization expects 3 channels");
  VT_CHECK(static_cast<int64_t>(frames.size()) == channels * per_channel,
           "minimax_h3: frame buffer does not match [C, ...]");
  for (int64_t c = 0; c < channels; ++c) {
    const float mean = kMiniMaxH3ImagenetMean[c], std_dev = kMiniMaxH3ImagenetStd[c];
    for (int64_t i = 0; i < per_channel; ++i) {
      float& v = frames[static_cast<size_t>(c * per_channel + i)];
      // clamp in [0,1] BEFORE the [-1,1] map, matching vae.py:693's order --
      // clamping afterwards would let out-of-gamut values survive rescaled.
      const float pixel = std::min(1.0F, std::max(0.0F, v * std_dev + mean));
      v = pixel * 2.0F - 1.0F;
    }
  }
}

void MiniMaxH3VideoNormalizePixels(std::vector<float>& frames, int64_t channels,
                                   int64_t per_channel) {
  VT_CHECK(channels == 3, "minimax_h3: ImageNet normalization expects 3 channels");
  VT_CHECK(static_cast<int64_t>(frames.size()) == channels * per_channel,
           "minimax_h3: frame buffer does not match [C, ...]");
  for (int64_t c = 0; c < channels; ++c) {
    const float mean = kMiniMaxH3ImagenetMean[c], std_dev = kMiniMaxH3ImagenetStd[c];
    for (int64_t i = 0; i < per_channel; ++i) {
      float& v = frames[static_cast<size_t>(c * per_channel + i)];
      v = ((v + 1.0F) * 0.5F - mean) / std_dev;  // vae.py:659
    }
  }
}

std::vector<float> MiniMaxH3EncodeReferenceVideo(
    const MiniMaxH3EncoderFcn3dConfig& encoder_config,
    const MiniMaxH3AudioVaeWeights& encoder_weights, const MiniMaxH3DitParams& dit_params,
    const std::vector<float>& frames, int64_t frame_count, int64_t frame_h, int64_t frame_w,
    MiniMaxH3RefBlock* out_block) {
  VT_CHECK(frame_count > 0, "minimax_h3 ref2va: a video reference needs at least one frame");
  VT_CHECK(static_cast<int64_t>(frames.size()) ==
               encoder_config.in_channels * frame_count * frame_h * frame_w,
           "minimax_h3 ref2va: reference video is not [in_channels, T, H, W]");

  // The 3D CNN is CAUSAL in time, so a clip encodes in one call -- this is the same
  // encoder the single-image path uses, just with t > 1, which is why a video
  // reference needed no new porting once the image path existed.
  // Same ImageNet normalization the image path needs (vae.py:659).
  std::vector<float> norm = frames;
  MiniMaxH3VideoNormalizePixels(norm, encoder_config.in_channels, frame_count * frame_h * frame_w);
  MiniMaxH3EncoderFcn3dConfig cfg = encoder_config;
  cfg.t = frame_count;
  cfg.h = frame_h;
  cfg.w = frame_w;
  MiniMaxH3VideoFrameShape ls{};
  const std::vector<float> latent =
      MiniMaxH3VideoVaeEncodeToLatent(cfg, encoder_weights, norm, &ls);
  std::vector<float> rows = MiniMaxH3PatchifyVideoLatent(
      latent, /*batch=*/1, dit_params.latents_dim, ls.t, ls.h, ls.w, dit_params.patch_size_t,
      dit_params.patch_size_h, dit_params.patch_size_w);

  if (out_block != nullptr) {
    // kVideoAudio is the only kind that carries a temporal extent -- kImage counts
    // exactly one frame regardless of latent_t. ref_audio_t stays 0: silent.
    MiniMaxH3RefBlock b;
    b.kind = MiniMaxH3RefBlock::Kind::kVideoAudio;
    b.ref_audio_t = 0;
    b.latent_t = ls.t / dit_params.patch_size_t;
    b.latent_h = ls.h / dit_params.patch_size_h;
    b.latent_w = ls.w / dit_params.patch_size_w;
    *out_block = b;
  }
  return rows;
}

// Turn a supplied REFERENCE WAVEFORM into the packed audio conditioning rows, the
// audio counterpart of MiniMaxH3EncodeReferenceImages. This is what makes a
// `kAudio` block (and a `kVideoAudio` block with sound) expressible at all.
//
// `waveform` is CHANNEL-MAJOR and already at kMiniMaxH3AudioSampleRate; a mono
// source is REPEATED up to kMiniMaxH3AudioChannels, exactly as upstream does
// before it encodes (vae.py:305-313). Rows come back normalized by the audio VAE's
// own latent statistics, because that is the space the DiT's rows live in.
//
// `noise_aug` blends toward the supplied noise the way the visual side does
// (1.0 pins the reference exactly); noise is an INPUT for the same reason it is
// everywhere else in this pipeline.
std::vector<float> MiniMaxH3EncodeReferenceAudio(
    const MiniMaxH3AudioVaeEncoderConfig& encoder_config,
    const MiniMaxH3AudioVaeWeights& encoder_weights, const std::vector<float>& waveform,
    int64_t channels, int64_t samples_per_channel, const std::vector<float>& latents_mean,
    const std::vector<float>& latents_std, double noise_aug,
    const std::vector<float>& noise_rows, MiniMaxH3RefBlock* out_block) {
  VT_CHECK(channels == kMiniMaxH3AudioChannels,
           "minimax_h3 ref2va: reference audio must be stereo (mono is repeated by the caller)");
  int64_t audio_t = 0;
  std::vector<float> rows = MiniMaxH3AudioVaeEncodeToRows(
      encoder_config, encoder_weights, waveform, channels, samples_per_channel, latents_mean,
      latents_std, &audio_t);
  VT_CHECK(audio_t > 0, "minimax_h3 ref2va: the reference waveform encoded to zero latent frames");

  if (out_block != nullptr) {
    // A bare audio reference is a kAudio block: no visual extent, `ref_audio_t`
    // latent frames per channel.
    MiniMaxH3RefBlock b;
    b.kind = MiniMaxH3RefBlock::Kind::kAudio;
    b.ref_audio_t = audio_t;
    *out_block = b;
  }
  if (noise_aug >= 1.0 || noise_rows.empty()) return rows;
  return MiniMaxH3AudioCondNoiseAug(rows, {audio_t}, noise_aug, noise_rows);
}

std::vector<float> MiniMaxH3EncodeReferenceImages(
    const MiniMaxH3EncoderFcn3dConfig& encoder_config,
    const MiniMaxH3AudioVaeWeights& encoder_weights, const MiniMaxH3DitParams& dit_params,
    const std::vector<std::vector<float>>& images, int64_t image_h, int64_t image_w,
    std::vector<MiniMaxH3RefBlock>* out_blocks) {
  VT_CHECK(!images.empty(), "minimax_h3 ref2va: no reference images supplied");
  std::vector<float> rows;
  if (out_blocks != nullptr) out_blocks->clear();
  for (const std::vector<float>& img : images) {
    VT_CHECK(static_cast<int64_t>(img.size()) == encoder_config.in_channels * image_h * image_w,
             "minimax_h3 ref2va: reference image is not [in_channels, H, W]");
    // The encoder consumes IMAGENET-NORMALIZED pixels (vae.py:659), not [-1, 1].
    std::vector<float> norm = img;
    MiniMaxH3VideoNormalizePixels(norm, encoder_config.in_channels, image_h * image_w);
    MiniMaxH3EncoderFcn3dConfig cfg = encoder_config;
    cfg.t = 1;
    cfg.h = image_h;
    cfg.w = image_w;
    MiniMaxH3VideoFrameShape ls{};
    const std::vector<float> latent =
        MiniMaxH3VideoVaeEncodeToLatent(cfg, encoder_weights, norm, &ls);
    const std::vector<float> patched = MiniMaxH3PatchifyVideoLatent(
        latent, /*batch=*/1, dit_params.latents_dim, ls.t, ls.h, ls.w, dit_params.patch_size_t,
        dit_params.patch_size_h, dit_params.patch_size_w);
    rows.insert(rows.end(), patched.begin(), patched.end());
    if (out_blocks != nullptr) {
      // The block declares the PATCHED grid, which is what occupies packed rows.
      MiniMaxH3RefBlock b;
      b.kind = MiniMaxH3RefBlock::Kind::kImage;
      b.latent_t = ls.t / dit_params.patch_size_t;
      b.latent_h = ls.h / dit_params.patch_size_h;
      b.latent_w = ls.w / dit_params.patch_size_w;
      out_blocks->push_back(b);
    }
  }
  return rows;
}

std::vector<float> MiniMaxH3EncodeKeyframeCondRows(
    const MiniMaxH3EncoderFcn3dConfig& encoder_config, const MiniMaxH3AudioVaeWeights& encoder_weights,
    const MiniMaxH3DitParams& dit_params, const std::vector<std::vector<float>>& images,
    int64_t image_h, int64_t image_w, int64_t target_latent_t, double noise_aug,
    const std::vector<float>& noise_rows) {
  VT_CHECK(!images.empty(), "minimax_h3 keyframe: no images supplied");
  std::vector<float> rows;
  std::vector<int64_t> condition_shapes;
  for (const std::vector<float>& img : images) {
    VT_CHECK(static_cast<int64_t>(img.size()) == encoder_config.in_channels * image_h * image_w,
             "minimax_h3 keyframe: image is not [in_channels, H, W]");
    // The encoder consumes IMAGENET-NORMALIZED pixels (vae.py:659), not [-1, 1].
    std::vector<float> norm = img;
    MiniMaxH3VideoNormalizePixels(norm, encoder_config.in_channels, image_h * image_w);
    MiniMaxH3EncoderFcn3dConfig cfg = encoder_config;
    cfg.t = 1;  // a single frame is a valid causal clip
    cfg.h = image_h;
    cfg.w = image_w;
    MiniMaxH3VideoFrameShape ls{};
    const std::vector<float> latent =
        MiniMaxH3VideoVaeEncodeToLatent(cfg, encoder_weights, norm, &ls);
    // [C, t, h, w] -> packed rows with the DiT's patch volume.
    std::vector<float> patched = MiniMaxH3PatchifyVideoLatent(
        latent, /*batch=*/1, dit_params.latents_dim, ls.t, ls.h, ls.w, dit_params.patch_size_t,
        dit_params.patch_size_h, dit_params.patch_size_w);
    rows.insert(rows.end(), patched.begin(), patched.end());
    condition_shapes.push_back(ls.t);
    condition_shapes.push_back(ls.h);
    condition_shapes.push_back(ls.w);
  }
  if (noise_aug >= 1.0 || noise_rows.empty()) return rows;
  return MiniMaxH3ImgvidCondNoiseAug(rows, condition_shapes, target_latent_t,
                                     static_cast<int64_t>(images.size()), noise_aug, noise_rows);
}

MiniMaxH3DenoiseResult MiniMaxH3DenoiseT2va(vt::Device device, const MiniMaxH3T2vaRequest& request,
                                            const MiniMaxH3DitParams& dit_params,
                                            const MiniMaxH3DitWeights& dit_weights,
                                            const std::vector<float>& prompt_embeds,
                                            const std::vector<float>& initial_video_rows,
                                            const std::vector<float>& initial_audio_rows,
                                            vt::DType compute_dtype,
                                            const MiniMaxH3DitDeviceWeights* prestaged) {
  VT_CHECK(request.text_len > 0, "minimax_h3 t2va: text_len must be positive");
  VT_CHECK(request.num_steps >= 1, "minimax_h3 t2va: num_steps must be >= 1");
  VT_CHECK(static_cast<int64_t>(prompt_embeds.size()) == request.text_len * dit_params.text_dim,
           "minimax_h3 t2va: prompt_embeds must be [text_len, text_dim]");

  // --- 1. packed layout. fl2va is the SAME layout with keyframe conditioning
  // switched on, so t2va is just the empty-keyframe case rather than a separate
  // path -- upstream models it the same way (packed_sequence.py:116-239).
  const bool has_keyframes = !request.keyframe_frame_indices.empty();
  const bool has_refs = !request.ref_blocks.empty();
  VT_CHECK(!(has_keyframes && has_refs),
           "minimax_h3: keyframe (fl2va) and reference (ref2va) conditioning are exclusive");
  MiniMaxH3DenoiseBranch branch;
  int64_t ref_audio_rows = 0;
  if (has_refs) {
    // AUDIO-bearing reference blocks claim packed audio rows, and every claimed row
    // must be BACKED by an encoded reference -- otherwise the layout would grow and
    // the loop would pin whatever happened to be in the buffer. The rows come from
    // MiniMaxH3EncodeReferenceAudio (the audio VAE's encoder half).
    for (const MiniMaxH3RefBlock& b : request.ref_blocks) {
      if (b.kind == MiniMaxH3RefBlock::Kind::kImage) continue;
      VT_CHECK(b.ref_audio_t >= 0, "minimax_h3 ref2va: ref_audio_t cannot be negative");
      ref_audio_rows += b.ref_audio_t * request.audio_channel;
    }
    VT_CHECK(static_cast<int64_t>(request.audio_ref_rows.size()) ==
                 ref_audio_rows * dit_params.audio_latents_dim,
             "minimax_h3 ref2va: audio_ref_rows must supply exactly the rows the reference blocks "
             "claim (ref_audio_t * audio_channel per block) -- see "
             "MiniMaxH3EncodeReferenceAudio");
    branch.packed = BuildMiniMaxH3PackedSequenceRef2va(
        request.text_len, request.latent_t, request.latent_h, request.latent_w, request.audio_t,
        request.ref_blocks, request.audio_channel);
  } else {
    branch.packed = BuildMiniMaxH3PackedSequence(
        request.text_len, request.latent_t, request.latent_h, request.latent_w, request.audio_t,
        request.audio_channel, has_keyframes, request.keyframe_frame_indices,
        has_keyframes ? request.num_frames : 0);
  }
  branch.text_embeddings = prompt_embeds;
  branch.token_tags = branch.packed.token_tags;

  // --- 2. the rectified-flow sigma schedules (video and audio shift differently) ---
  const std::vector<double> sigmas_video =
      MiniMaxH3TimeShiftSigmas(request.num_steps, request.video_shift);
  const std::vector<double> sigmas_audio =
      MiniMaxH3TimeShiftSigmas(request.num_steps, request.audio_shift);
  VT_CHECK(sigmas_video.size() == sigmas_audio.size(),
           "minimax_h3 t2va: the two sigma schedules must have equal length");

  // --- 3. the denoise loop (one DiT forward per step) ---
  // Keyframe conditioning ADDS condition rows to the packed layout, and the loop
  // wants initial rows for every img position. Callers supply noise for the TARGET
  // rows -- that is what they can know -- so the condition slots are filled in
  // here. Their contents do not matter: the loop pins them to the keyframe anchors
  // before the first step.
  const int64_t video_width = dit_params.video_row_width();
  const int64_t num_img = static_cast<int64_t>(branch.packed.img_pos.size());
  std::vector<float> video_rows = initial_video_rows;
  if (static_cast<int64_t>(video_rows.size()) != num_img * video_width) {
    VT_CHECK(has_keyframes || has_refs,
             "minimax_h3 t2va: initial video rows do not match the packed layout");
    std::vector<float> full(static_cast<size_t>(num_img * video_width), 0.0f);
    int64_t src = 0;
    for (int64_t r = 0; r < num_img; ++r) {
      if (!branch.packed.update_mask[static_cast<size_t>(r)]) continue;  // pinned anchor
      VT_CHECK((src + 1) * video_width <= static_cast<int64_t>(initial_video_rows.size()),
               "minimax_h3 t2va: too few initial video rows for the denoise targets");
      std::copy(initial_video_rows.begin() + src * video_width,
                initial_video_rows.begin() + (src + 1) * video_width,
                full.begin() + r * video_width);
      ++src;
    }
    video_rows = std::move(full);
  }

  // The audio side has the SAME shape of problem once a reference carries sound:
  // reference audio adds PINNED rows to the layout, and callers supply noise only
  // for the denoise TARGETS (pipeline_minimax_h3.py:937-955). Scatter theirs into
  // the updating slots and leave the pinned ones to the anchors below.
  const int64_t audio_width = dit_params.audio_latents_dim;
  const int64_t num_audio = static_cast<int64_t>(branch.packed.audio_pos.size());
  std::vector<float> audio_rows = initial_audio_rows;
  if (static_cast<int64_t>(audio_rows.size()) != num_audio * audio_width) {
    VT_CHECK(ref_audio_rows > 0,
             "minimax_h3 t2va: initial audio rows do not match the packed layout");
    std::vector<float> full(static_cast<size_t>(num_audio * audio_width), 0.0f);
    int64_t src = 0;
    for (int64_t r = 0; r < num_audio; ++r) {
      if (!branch.packed.audio_update_mask.empty() &&
          !branch.packed.audio_update_mask[static_cast<size_t>(r)]) {
        continue;  // a pinned reference row
      }
      VT_CHECK((src + 1) * audio_width <= static_cast<int64_t>(initial_audio_rows.size()),
               "minimax_h3 t2va: too few initial audio rows for the denoise targets");
      std::copy(initial_audio_rows.begin() + src * audio_width,
                initial_audio_rows.begin() + (src + 1) * audio_width,
                full.begin() + r * audio_width);
      ++src;
    }
    audio_rows = std::move(full);
  }

  // Both the keyframe rows and the reference-audio rows are PINNED: the loop
  // resets them to these anchors every step, so the supplied conditioning stays
  // put instead of being denoised away.
  return MiniMaxH3DenoiseLoop(device, dit_params, dit_weights, branch, video_rows, audio_rows,
                              request.keyframe_cond_rows, request.audio_ref_rows, sigmas_video,
                              sigmas_audio, compute_dtype, prestaged);
}

MiniMaxH3T2vaResult MiniMaxH3GenerateT2va(vt::Device device, const MiniMaxH3T2vaRequest& request,
                                          const MiniMaxH3DitParams& dit_params,
                                          const MiniMaxH3DitWeights& dit_weights,
                                          const MiniMaxH3VideoVaeDecoderConfig& video_config,
                                          const MiniMaxH3AudioVaeWeights& video_weights,
                                          const MiniMaxH3AudioVaeConfig& audio_config,
                                          const MiniMaxH3AudioVaeWeights& audio_weights,
                                          const std::vector<float>& prompt_embeds,
                                          const std::vector<float>& initial_video_rows,
                                          const std::vector<float>& initial_audio_rows,
                                          vt::DType compute_dtype,
                                          const MiniMaxH3DitDeviceWeights* prestaged) {
  const MiniMaxH3DenoiseResult denoised =
      MiniMaxH3DenoiseT2va(device, request, dit_params, dit_weights, prompt_embeds,
                           initial_video_rows, initial_audio_rows, compute_dtype, prestaged);

  // --- 4. rows -> latents ---
  const int64_t ph = request.latent_h / dit_params.patch_size_h;
  const int64_t pw = request.latent_w / dit_params.patch_size_w;
  std::vector<float> video_latent = MiniMaxH3UnpatchifyVideoTokens(
      denoised.video_rows, request.latent_t, ph, pw, dit_params.latents_dim,
      dit_params.patch_size_t, dit_params.patch_size_h, dit_params.patch_size_w);
  std::vector<float> audio_latent = MiniMaxH3UnpackAudioTokens(
      denoised.audio_rows, request.audio_t * request.audio_channel, request.audio_channel,
      dit_params.audio_latents_dim);

  // --- 5. denormalize (vae.py:252-270, :341-357) ---
  auto denormalize = [](std::vector<float>& latent, int64_t channels, int64_t per_channel,
                        const std::vector<float>& mean, const std::vector<float>& std_dev) {
    if (mean.empty() && std_dev.empty()) return;
    VT_CHECK(static_cast<int64_t>(mean.size()) == channels &&
                 static_cast<int64_t>(std_dev.size()) == channels,
             "minimax_h3 t2va: latents_mean/std must have one value per channel");
    for (int64_t c = 0; c < channels; ++c) {
      for (int64_t i = 0; i < per_channel; ++i) {
        float& value = latent[static_cast<size_t>(c * per_channel + i)];
        value = value * std_dev[static_cast<size_t>(c)] + mean[static_cast<size_t>(c)];
      }
    }
  };
  const int64_t video_per_channel = request.latent_t * request.latent_h * request.latent_w;
  denormalize(video_latent, dit_params.latents_dim, video_per_channel, request.video_latents_mean,
              request.video_latents_std);
  const int64_t audio_steps = request.audio_t / request.audio_channel;
  denormalize(audio_latent, dit_params.audio_latents_dim, audio_steps * request.audio_channel,
              request.audio_latents_mean, request.audio_latents_std);

  // --- 6. decode ---
  MiniMaxH3T2vaResult result;
  // post_quant_conv (the AutoencoderKL WRAPPER's Conv3d 1x1x1 channel mix) runs on
  // the latent BEFORE the decoder. It sits OUTSIDE ViT3DDecoder, which is why the
  // decoder's own 8.9e-8 gate never covered it. Applied only when the weights
  // carry it, so a synthetic/reduced-dimension weight set without it still runs --
  // the structural t2va test does not ship one.
  if (video_weights.Has("post_quant_conv.weight")) {
    video_latent = MiniMaxH3VideoVaePostQuantConv(video_weights, video_latent,
                                                  dit_params.latents_dim, video_per_channel);
  }
  // On a device, run the ViT3D decoder device-resident. The portable decoder is a
  // scalar reference; at real resolutions it is the stage that does not finish. It
  // stays the CPU path, and stays the thing the device path is gated against.
  if (device.type != vt::DeviceType::kCPU) {
    vt::Queue vq = vt::GetBackend(device.type).CreateQueue();
    const MiniMaxH3VideoVaeDeviceWeights staged_vae =
        StageMiniMaxH3VideoVaeWeights(vq, video_config, video_weights);
    // Upstream's video path is decode_base -> decode_temporal: chunked in TIME,
    // and NOT spatially tiled (decoder_tiling defaults false and the shipped
    // config does not set it).
    result.frames = MiniMaxH3VideoVaeDecodeTemporalDevice(
        device, video_config, staged_vae, video_latent, request.latent_t, request.latent_h,
        request.latent_w, request.num_frames, &result.frame_shape);
  } else {
    result.frames = MiniMaxH3VideoVaeDecode(video_config, video_weights, video_latent,
                                            request.latent_t, request.latent_h, request.latent_w,
                                            &result.frame_shape);
  }
  // The ViT decoder emits IMAGENET-NORMALIZED values; the wrapper de-normalizes
  // them (vae.py:693). Outside the decoder, so its own gate never covered it --
  // the same shape of omission as post_quant_conv.
  MiniMaxH3VideoDenormalizePixels(
      result.frames, result.frame_shape.channels,
      result.frame_shape.t * result.frame_shape.h * result.frame_shape.w);

  // The audio VAE decodes ONE channel at a time; the packed rows are channel-major.
  result.audio_channels = request.audio_channel;
  for (int64_t c = 0; c < request.audio_channel; ++c) {
    std::vector<float> channel(static_cast<size_t>(dit_params.audio_latents_dim * audio_steps));
    for (int64_t d = 0; d < dit_params.audio_latents_dim; ++d) {
      for (int64_t t = 0; t < audio_steps; ++t) {
        channel[static_cast<size_t>(d * audio_steps + t)] = audio_latent[static_cast<size_t>(
            (c * dit_params.audio_latents_dim + d) * audio_steps + t)];
      }
    }
    int64_t samples = 0;
    const std::vector<float> wave =
        MiniMaxH3AudioVaeDecode(audio_config, audio_weights, channel, audio_steps, &samples);
    if (c == 0) {
      result.audio_samples_per_channel = samples;
      result.waveform.reserve(static_cast<size_t>(samples * request.audio_channel));
    } else {
      VT_CHECK(samples == result.audio_samples_per_channel,
               "minimax_h3 t2va: audio channels decoded to different lengths");
    }
    result.waveform.insert(result.waveform.end(), wave.begin(), wave.end());
  }
  result.sample_rate = kMiniMaxH3AudioSampleRate;
  return result;
}

}  // namespace vllm
