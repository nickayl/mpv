# mpv — Quven fork

This is a public fork of [mpv](https://github.com/mpv-player/mpv) maintained by the
[Quven](https://quven.tv) media server project. The libmpv built from this fork ships inside the
Quven Android client; publishing the modified sources here satisfies libmpv's LGPL-2.1+ §6 source
offer.

Everything lives on the branch **`quven/android-vk-interop`**, based on upstream master and kept in sync via periodic merges (last: 2026-07-02). `master` tracks upstream and carries no changes.

## How this fork diverges from upstream

### `hwdec_aimagereader_vk` — Android MediaCodec zero-copy for Vulkan (`--gpu-context=androidvk`)

Upstream mpv can hardware-decode into a Vulkan swapchain on Android only through
`mediacodec-copy` (a per-frame CPU copy-back), because its only zero-copy interop
(`hwdec_aimagereader`) maps decoder buffers through EGL into GL external-OES textures.

This fork adds `video/out/hwdec/hwdec_aimagereader_vk.c`, a self-contained hwdec driver that keeps
plain `hwdec=mediacodec` fully on the GPU under Vulkan:

- MediaCodec renders into an `AImageReader`; each frame's `AHardwareBuffer` is imported into a
  `VkImage` (`VK_ANDROID_external_memory_android_hardware_buffer`, dedicated allocation, foreign-
  queue acquire barrier via `VK_EXT_queue_family_foreign`).
- Drivers that expose decoder buffers as opaque *external formats* (every driver we tested:
  Samsung Xclipse, Qualcomm Adreno) get a single compute pass that samples the image through an
  immutable `VkSamplerYcbcrConversion` (the driver-suggested model/range) into an owned RGBA16
  texture handed to libplacebo — one GPU repack, zero CPU copies, correct colors for 8-bit, 10-bit,
  HDR10 and Dolby Vision (profile 8) content.
- Drivers that report standard multiplanar formats (none seen so far) take a direct per-plane
  `pl_vulkan_wrap` path instead — true YUV planes, no repack.
- Synchronization uses one timeline semaphore against libplacebo plus a deferred one-frame
  teardown; the AImageReader acquire fence is imported as a temporary binary semaphore
  (`VK_KHR_external_semaphore_fd`) with a bounded CPU poll fallback.
- `mppl_create_vulkan` opportunistically enables the AHardwareBuffer import extensions on device
  creation.

Notes for other integrators:

- Dolby Vision: mpv applies DV metadata through libplacebo. On the repacked-RGB path the
  component reshape is intentionally disabled (it operates on the pre-matrix signal); the dynamic
  per-scene brightness (`color.hdr`) still drives tonemapping. Overriding the ycbcr conversion to
  `YCBCR_IDENTITY` to recover raw planes does NOT work on external formats — the Vulkan spec ties
  external-format conversions to the driver-suggested model, and drivers accept the override
  silently while returning broken chroma (tested on Xclipse; the experiment is preserved in this
  branch's history as reverted commits).
- The decoder must receive frame side data for DV to work with hardware decode: FFmpeg's
  `mediacodecdec` does not attach the DOVI RPU upstream. Quven carries a small FFmpeg patch for
  that (not part of this repo — see `tools/libmpv/quven-ffmpeg-dovi-mediacodec.patch` in the Quven
  distribution artifacts, or apply the equivalent to your FFmpeg build).

### Android embedder patches absorbed as commits

- `android: expose mpv_lavc_set_java_vm for embedders` — lets a host process hand the JavaVM to
  libavcodec before any decoder opens (required for MediaCodec when libmpv is loaded by a runtime
  that does not register the VM itself).
- `ao_aaudio: tolerate pre-34 NDK headers for IEC61937 passthrough` — back-fills AAudio IEC61937
  constants when building against older NDK sysroots.

## Building

The fork builds exactly like upstream mpv. The interop compiles when both `android-media-ndk` and
`vulkan` features are enabled; it registers as hwdec `aimagereader-vk` and is picked automatically
by `--gpu-context=androidvk` sessions with `--hwdec=mediacodec`. Requirements: libplacebo ≥ 7.360
built with Vulkan support (including `vk-proc-addr` — the loader link), NDK ≥ r26.

Quven's own reproducible build recipe (LGPL toolchain, arm64) lives in the Quven repository as
`tools/libmpv/build-libmpv-android.sh`, which consumes this fork via `QUVEN_MPV_REPO`/`QUVEN_MPV_REF`.

## License

Same as upstream mpv: LGPL-2.1+ (with the usual notes in the upstream `Copyright` file). The
changes in this fork are offered under the same terms.
