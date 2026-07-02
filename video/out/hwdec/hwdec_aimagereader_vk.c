/*
 * Android MediaCodec to Vulkan interop via AImageReader / AHardwareBuffer.
 *
 * Copyright (c) 2026 Quven contributors
 *
 * This file is part of mpv.
 *
 * mpv is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * mpv is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with mpv.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * The MediaCodec decoder renders into an AImageReader surface (same flow as
 * hwdec_aimagereader.c). Each frame's AHardwareBuffer is imported into a
 * VkImage, then one of two paths applies:
 *
 * - Direct: when the driver reports a standard multiplanar format for the
 *   buffer, the planes are wrapped as libplacebo textures directly and the
 *   frame stays true YUV (NV12 / P010).
 *
 * - Repack: drivers such as Samsung Xclipse expose decoder buffers only as
 *   opaque "external formats" that can solely be sampled through a
 *   VkSamplerYcbcrConversion, which libplacebo cannot consume, so a small
 *   compute pass samples the imported image once (the driver conversion
 *   yields RGB) into an owned RGBA16 storage image. Mirroring the GL interop,
 *   the result is treated as IMGFMT_RGB0 with the driver-suggested
 *   matrix/range applied.
 *
 * Both paths keep everything on the GPU (no CPU copy-back like
 * mediacodec-copy). The AImageReader acquire fence is imported as a temporary
 * binary semaphore when VK_KHR_external_semaphore_fd is available (bounded
 * CPU poll otherwise).
 */

#include <assert.h>
#include <dlfcn.h>
#include <poll.h>
#include <string.h>
#include <unistd.h>

#include <media/NdkImageReader.h>
#include <android/native_window_jni.h>
#include <android/hardware_buffer.h>
#include <libavcodec/mediacodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_mediacodec.h>

#include <vulkan/vulkan.h>
#include <vulkan/vulkan_android.h>
#include <libplacebo/vulkan.h>

#include "misc/jni.h"
#include "osdep/threads.h"
#include "osdep/timer.h"
#include "video/out/gpu/hwdec.h"
#include "video/out/vulkan/context.h"
#include "video/out/placebo/ra_pl.h"

#include "aimagereader_vk_shader.h"

#define AHB_VK_EXTENSION "VK_ANDROID_external_memory_android_hardware_buffer"
#define FOREIGN_VK_EXTENSION "VK_EXT_queue_family_foreign"
#define SEM_FD_VK_EXTENSION "VK_KHR_external_semaphore_fd"

struct priv_owner {
    struct mp_hwdec_ctx hwctx;
    AImageReader *reader;
    jobject surface;
    void *lib_handle;

    pl_gpu gpu;
    pl_vulkan vk;
    PFN_vkGetAndroidHardwareBufferPropertiesANDROID get_ahb_props;
    PFN_vkImportSemaphoreFdKHR import_sem_fd;
    uint32_t external_qf;   // VK_QUEUE_FAMILY_FOREIGN_EXT when available

    media_status_t (*AImageReader_newWithUsage)(
        int32_t, int32_t, int32_t, uint64_t, int32_t, AImageReader **);
    media_status_t (*AImageReader_getWindow)(
        AImageReader *, ANativeWindow **);
    media_status_t (*AImageReader_setImageListener)(
        AImageReader *, AImageReader_ImageListener *);
    media_status_t (*AImageReader_acquireLatestImage)(AImageReader *, AImage **);
    media_status_t (*AImageReader_acquireLatestImageAsync)(
        AImageReader *, AImage **, int *);
    void (*AImageReader_delete)(AImageReader *);
    media_status_t (*AImage_getHardwareBuffer)(const AImage *, AHardwareBuffer **);
    void (*AImage_delete)(AImage *);
    void (*AHardwareBuffer_describe)(const AHardwareBuffer *, AHardwareBuffer_Desc *);
    jobject (*ANativeWindow_toSurface)(JNIEnv *, ANativeWindow *);
};

struct priv {
    struct mp_log *log;

    AImage *image;
    AImage *prev_image;

    mp_mutex lock;
    mp_cond cond;
    bool image_available;

    VkDevice dev;
    VkQueue queue;
    uint32_t qf;
    VkCommandPool cmd_pool;
    VkCommandBuffer cmd;
    VkSemaphore timeline;
    uint64_t tl_value;         // last timeline value allocated
    uint64_t last_submit;      // timeline value signaled by our last submit
    VkSemaphore acquire_sem;   // temporary import target for acquire fences
    bool wait_acquire;         // acquire_sem holds a payload for this frame

    // Path choice, decided per buffer format.
    bool direct;

    // Repack path state, recreated when the buffer's format changes.
    uint64_t cur_external_format;
    VkFormat cur_format;
    VkSamplerYcbcrModelConversion cur_model;
    VkSamplerYcbcrConversion conv;
    VkSampler sampler;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout pll;
    VkPipeline pipe;
    VkDescriptorPool dpool;
    VkDescriptorSet dset;

    // Per-frame imported input; destroyed one frame late (after our compute
    // provably retired, via the timeline wait at the next map).
    VkImage in_image, prev_in_image;
    VkDeviceMemory in_mem, prev_in_mem;
    VkImageView in_view, prev_in_view;

    // Direct path: per-frame plane wrappers, freed one frame late.
    struct ra_tex *plane_ra[2], *prev_plane_ra[2];

    // Repack path: owned output image the compute pass writes.
    pl_tex out_tex;
    struct ra_tex *out_ra;
    VkImage out_image;
    VkImageView out_view;
    uint32_t out_w, out_h;
};

static const struct { const char *symbol; int offset; bool optional; } lib_functions[] = {
    { "AImageReader_newWithUsage", offsetof(struct priv_owner, AImageReader_newWithUsage), false },
    { "AImageReader_getWindow", offsetof(struct priv_owner, AImageReader_getWindow), false },
    { "AImageReader_setImageListener", offsetof(struct priv_owner, AImageReader_setImageListener), false },
    { "AImageReader_acquireLatestImage", offsetof(struct priv_owner, AImageReader_acquireLatestImage), false },
    { "AImageReader_acquireLatestImageAsync", offsetof(struct priv_owner, AImageReader_acquireLatestImageAsync), true },
    { "AImageReader_delete", offsetof(struct priv_owner, AImageReader_delete), false },
    { "AImage_getHardwareBuffer", offsetof(struct priv_owner, AImage_getHardwareBuffer), false },
    { "AImage_delete", offsetof(struct priv_owner, AImage_delete), false },
    { "AHardwareBuffer_describe", offsetof(struct priv_owner, AHardwareBuffer_describe), false },
    { "ANativeWindow_toSurface", offsetof(struct priv_owner, ANativeWindow_toSurface), false },
    { NULL, 0, false },
};

static AVBufferRef *create_mediacodec_device_ref(jobject surface)
{
    AVBufferRef *device_ref = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_MEDIACODEC);
    if (!device_ref)
        return NULL;

    AVHWDeviceContext *ctx = (void *)device_ref->data;
    AVMediaCodecDeviceContext *hwctx = ctx->hwctx;
    hwctx->surface = surface;

    if (av_hwdevice_ctx_init(device_ref) < 0)
        av_buffer_unref(&device_ref);

    return device_ref;
}

static bool load_lib_functions(struct priv_owner *p, struct mp_log *log)
{
    p->lib_handle = dlopen("libmediandk.so", RTLD_NOW | RTLD_GLOBAL);
    if (!p->lib_handle)
        return false;
    for (int i = 0; lib_functions[i].symbol; i++) {
        const char *sym = lib_functions[i].symbol;
        void *fun = dlsym(p->lib_handle, sym);
        if (!fun)
            fun = dlsym(RTLD_DEFAULT, sym);
        if (!fun && !lib_functions[i].optional) {
            mp_warn(log, "Could not resolve symbol %s\n", sym);
            return false;
        }

        *(void **) ((uint8_t*)p + lib_functions[i].offset) = fun;
    }
    return true;
}

static int init(struct ra_hwdec *hw)
{
    struct priv_owner *p = hw->priv;
    int level = hw->probing ? MSGL_V : MSGL_ERR;

    struct mpvk_ctx *vk = ra_vk_ctx_get(hw->ra_ctx);
    if (!vk)
        return -1;

    p->gpu = ra_pl_get(hw->ra_ctx->ra);
    if (!p->gpu)
        return -1;

    p->vk = pl_vulkan_get(p->gpu);
    if (!p->vk)
        return -1;

    bool have_ahb = false, have_foreign = false, have_sem_fd = false;
    for (int i = 0; i < p->vk->num_extensions; i++) {
        if (strcmp(p->vk->extensions[i], AHB_VK_EXTENSION) == 0)
            have_ahb = true;
        if (strcmp(p->vk->extensions[i], FOREIGN_VK_EXTENSION) == 0)
            have_foreign = true;
        if (strcmp(p->vk->extensions[i], SEM_FD_VK_EXTENSION) == 0)
            have_sem_fd = true;
    }
    if (!have_ahb) {
        MP_MSG(hw, level, "Device lacks %s\n", AHB_VK_EXTENSION);
        return -1;
    }
    p->external_qf = have_foreign ? VK_QUEUE_FAMILY_FOREIGN_EXT
                                  : VK_QUEUE_FAMILY_EXTERNAL;

    PFN_vkGetDeviceProcAddr get_dev_proc = (PFN_vkGetDeviceProcAddr)
        p->vk->get_proc_addr(p->vk->instance, "vkGetDeviceProcAddr");
    if (get_dev_proc) {
        p->get_ahb_props = (PFN_vkGetAndroidHardwareBufferPropertiesANDROID)
            get_dev_proc(p->vk->device, "vkGetAndroidHardwareBufferPropertiesANDROID");
        if (have_sem_fd) {
            p->import_sem_fd = (PFN_vkImportSemaphoreFdKHR)
                get_dev_proc(p->vk->device, "vkImportSemaphoreFdKHR");
        }
    }
    if (!p->get_ahb_props) {
        MP_MSG(hw, level, "vkGetAndroidHardwareBufferPropertiesANDROID unavailable\n");
        return -1;
    }

    JNIEnv *env = MP_JNI_GET_ENV(hw);
    if (!env)
        return -1;

    if (!load_lib_functions(p, hw->log))
        return -1;

    // dummy dimensions, AImageReader only transports hardware buffers
    media_status_t ret = p->AImageReader_newWithUsage(16, 16,
        AIMAGE_FORMAT_PRIVATE, AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE,
        5, &p->reader);
    if (ret != AMEDIA_OK) {
        MP_ERR(hw, "newWithUsage failed: %d\n", ret);
        return -1;
    }

    ANativeWindow *window;
    ret = p->AImageReader_getWindow(p->reader, &window);
    if (ret != AMEDIA_OK) {
        MP_ERR(hw, "getWindow failed: %d\n", ret);
        return -1;
    }

    jobject surface = p->ANativeWindow_toSurface(env, window);
    p->surface = (*env)->NewGlobalRef(env, surface);
    (*env)->DeleteLocalRef(env, surface);

    p->hwctx = (struct mp_hwdec_ctx) {
        .driver_name = hw->driver->name,
        .av_device_ref = create_mediacodec_device_ref(p->surface),
        .hw_imgfmt = IMGFMT_MEDIACODEC,
    };

    if (!p->hwctx.av_device_ref) {
        MP_VERBOSE(hw, "Failed to create hwdevice_ctx\n");
        return -1;
    }

    hwdec_devices_add(hw->devs, &p->hwctx);

    MP_VERBOSE(hw, "Using the Vulkan AImageReader interop (fence import: %s)\n",
               p->import_sem_fd ? "semaphore" : "cpu poll");
    return 0;
}

static void uninit(struct ra_hwdec *hw)
{
    struct priv_owner *p = hw->priv;

    if (p->surface) {
        JNIEnv *env = MP_JNI_GET_ENV(hw);
        mp_assert(env);
        (*env)->DeleteGlobalRef(env, p->surface);
        p->surface = NULL;
    }

    if (p->reader) {
        p->AImageReader_delete(p->reader);
        p->reader = NULL;
    }

    hwdec_devices_remove(hw->devs, &p->hwctx);
    av_buffer_unref(&p->hwctx.av_device_ref);

    if (p->lib_handle) {
        dlclose(p->lib_handle);
        p->lib_handle = NULL;
    }
}

static void image_callback(void *context, AImageReader *reader)
{
    struct priv *p = context;

    mp_mutex_lock(&p->lock);
    p->image_available = true;
    mp_cond_signal(&p->cond);
    mp_mutex_unlock(&p->lock);
}

static int mapper_init(struct ra_hwdec_mapper *mapper)
{
    struct priv *p = mapper->priv;
    struct priv_owner *o = mapper->owner->priv;

    p->log = mapper->log;
    mp_mutex_init(&p->lock);
    mp_cond_init(&p->cond);

    AImageReader_ImageListener listener = {
        .context = p,
        .onImageAvailable = image_callback,
    };
    o->AImageReader_setImageListener(o->reader, &listener);

    // Refined per frame once the buffer format is known (direct YUV planes or
    // repacked RGB0).
    mapper->dst_params = mapper->src_params;
    mapper->dst_params.imgfmt = IMGFMT_RGB0;
    mapper->dst_params.hw_subfmt = 0;

    p->dev = o->vk->device;
    p->qf = o->vk->queue_compute.index;
    vkGetDeviceQueue(p->dev, p->qf, 0, &p->queue);

    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = p->qf,
    };
    if (vkCreateCommandPool(p->dev, &pool_info, NULL, &p->cmd_pool) != VK_SUCCESS)
        return -1;

    VkCommandBufferAllocateInfo cmd_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = p->cmd_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    if (vkAllocateCommandBuffers(p->dev, &cmd_info, &p->cmd) != VK_SUCCESS)
        return -1;

    VkSemaphoreTypeCreateInfo tl_type = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = 0,
    };
    VkSemaphoreCreateInfo sem_info = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &tl_type,
    };
    if (vkCreateSemaphore(p->dev, &sem_info, NULL, &p->timeline) != VK_SUCCESS)
        return -1;

    if (o->import_sem_fd) {
        VkSemaphoreCreateInfo bin_info = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        };
        if (vkCreateSemaphore(p->dev, &bin_info, NULL, &p->acquire_sem) != VK_SUCCESS)
            return -1;
    }

    return 0;
}

static void destroy_format_objects(struct priv *p)
{
    if (p->dpool) {
        vkDestroyDescriptorPool(p->dev, p->dpool, NULL);
        p->dpool = VK_NULL_HANDLE;
        p->dset = VK_NULL_HANDLE;
    }
    if (p->pipe) {
        vkDestroyPipeline(p->dev, p->pipe, NULL);
        p->pipe = VK_NULL_HANDLE;
    }
    if (p->pll) {
        vkDestroyPipelineLayout(p->dev, p->pll, NULL);
        p->pll = VK_NULL_HANDLE;
    }
    if (p->dsl) {
        vkDestroyDescriptorSetLayout(p->dev, p->dsl, NULL);
        p->dsl = VK_NULL_HANDLE;
    }
    if (p->sampler) {
        vkDestroySampler(p->dev, p->sampler, NULL);
        p->sampler = VK_NULL_HANDLE;
    }
    if (p->conv) {
        vkDestroySamplerYcbcrConversion(p->dev, p->conv, NULL);
        p->conv = VK_NULL_HANDLE;
    }
}

static void destroy_input(struct priv *p, VkImageView *view, VkImage *image,
                          VkDeviceMemory *mem)
{
    if (*view) {
        vkDestroyImageView(p->dev, *view, NULL);
        *view = VK_NULL_HANDLE;
    }
    if (*image) {
        vkDestroyImage(p->dev, *image, NULL);
        *image = VK_NULL_HANDLE;
    }
    if (*mem) {
        vkFreeMemory(p->dev, *mem, NULL);
        *mem = VK_NULL_HANDLE;
    }
}

static void free_plane_wrappers(struct ra_hwdec_mapper *mapper,
                                struct ra_tex **planes)
{
    for (int i = 0; i < 2; i++) {
        if (planes[i])
            ra_tex_free(mapper->ra, &planes[i]);
    }
}

static void destroy_output(struct ra_hwdec_mapper *mapper)
{
    struct priv *p = mapper->priv;

    if (p->out_view) {
        vkDestroyImageView(p->dev, p->out_view, NULL);
        p->out_view = VK_NULL_HANDLE;
    }
    if (p->out_ra) {
        ra_tex_free(mapper->ra, &p->out_ra);
        p->out_tex = NULL;
        p->out_image = VK_NULL_HANDLE;
    }
    p->out_w = p->out_h = 0;
}

static bool ensure_format_objects(struct ra_hwdec_mapper *mapper,
    const VkAndroidHardwareBufferFormatPropertiesANDROID *fmt)
{
    struct priv *p = mapper->priv;

    if (p->conv && fmt->externalFormat == p->cur_external_format &&
        fmt->format == p->cur_format &&
        fmt->suggestedYcbcrModel == p->cur_model)
        return true;

    destroy_format_objects(p);
    p->cur_external_format = fmt->externalFormat;
    p->cur_format = fmt->format;
    p->cur_model = fmt->suggestedYcbcrModel;

    MP_VERBOSE(p, "AHB import: vkFormat=%d externalFormat=0x%llx model=%d range=%d\n",
               (int)fmt->format, (unsigned long long)fmt->externalFormat,
               (int)fmt->suggestedYcbcrModel, (int)fmt->suggestedYcbcrRange);

    VkExternalFormatANDROID ext_fmt = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID,
        .externalFormat = fmt->format == VK_FORMAT_UNDEFINED
                              ? fmt->externalFormat : 0,
    };
    VkSamplerYcbcrConversionCreateInfo conv_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_CREATE_INFO,
        .pNext = ext_fmt.externalFormat ? &ext_fmt : NULL,
        .format = fmt->format,
        .ycbcrModel = fmt->suggestedYcbcrModel,
        .ycbcrRange = fmt->suggestedYcbcrRange,
        .components = fmt->samplerYcbcrConversionComponents,
        .xChromaOffset = fmt->suggestedXChromaOffset,
        .yChromaOffset = fmt->suggestedYChromaOffset,
        .chromaFilter = VK_FILTER_LINEAR,
        .forceExplicitReconstruction = VK_FALSE,
    };
    if (!(fmt->formatFeatures &
          VK_FORMAT_FEATURE_SAMPLED_IMAGE_YCBCR_CONVERSION_LINEAR_FILTER_BIT))
        conv_info.chromaFilter = VK_FILTER_NEAREST;
    if (vkCreateSamplerYcbcrConversion(p->dev, &conv_info, NULL, &p->conv)
            != VK_SUCCESS) {
        MP_ERR(p, "vkCreateSamplerYcbcrConversion failed\n");
        return false;
    }

    VkSamplerYcbcrConversionInfo conv_ref = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO,
        .conversion = p->conv,
    };
    VkSamplerCreateInfo sampler_info = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .pNext = &conv_ref,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .unnormalizedCoordinates = VK_FALSE,
    };
    if (vkCreateSampler(p->dev, &sampler_info, NULL, &p->sampler) != VK_SUCCESS) {
        MP_ERR(p, "vkCreateSampler failed\n");
        return false;
    }

    VkDescriptorSetLayoutBinding bindings[] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = &p->sampler,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    VkDescriptorSetLayoutCreateInfo dsl_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2,
        .pBindings = bindings,
    };
    if (vkCreateDescriptorSetLayout(p->dev, &dsl_info, NULL, &p->dsl) != VK_SUCCESS)
        return false;

    VkPipelineLayoutCreateInfo pll_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &p->dsl,
    };
    if (vkCreatePipelineLayout(p->dev, &pll_info, NULL, &p->pll) != VK_SUCCESS)
        return false;

    VkShaderModuleCreateInfo shader_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof(aimagereader_vk_shader_spv),
        .pCode = aimagereader_vk_shader_spv,
    };
    VkShaderModule shader;
    if (vkCreateShaderModule(p->dev, &shader_info, NULL, &shader) != VK_SUCCESS)
        return false;

    VkComputePipelineCreateInfo pipe_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader,
            .pName = "main",
        },
        .layout = p->pll,
    };
    VkResult res = vkCreateComputePipelines(p->dev, VK_NULL_HANDLE, 1,
                                            &pipe_info, NULL, &p->pipe);
    vkDestroyShaderModule(p->dev, shader, NULL);
    if (res != VK_SUCCESS) {
        MP_ERR(p, "vkCreateComputePipelines failed: %d\n", res);
        return false;
    }

    VkDescriptorPoolSize pool_sizes[] = {
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1 },
    };
    VkDescriptorPoolCreateInfo dpool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1,
        .poolSizeCount = 2,
        .pPoolSizes = pool_sizes,
    };
    if (vkCreateDescriptorPool(p->dev, &dpool_info, NULL, &p->dpool) != VK_SUCCESS)
        return false;

    VkDescriptorSetAllocateInfo dset_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = p->dpool,
        .descriptorSetCount = 1,
        .pSetLayouts = &p->dsl,
    };
    if (vkAllocateDescriptorSets(p->dev, &dset_info, &p->dset) != VK_SUCCESS)
        return false;

    return true;
}

static bool ensure_output(struct ra_hwdec_mapper *mapper, uint32_t w, uint32_t h)
{
    struct priv *p = mapper->priv;
    struct priv_owner *o = mapper->owner->priv;

    if (p->out_tex && p->out_w == w && p->out_h == h)
        return true;

    destroy_output(mapper);

    pl_fmt fmt = pl_find_named_fmt(o->gpu, "rgba16");
    if (!fmt) {
        MP_ERR(p, "no rgba16 format\n");
        return false;
    }

    p->out_tex = pl_tex_create(o->gpu, pl_tex_params(
        .w = w,
        .h = h,
        .format = fmt,
        .sampleable = true,
        .storable = true,
    ));
    if (!p->out_tex) {
        MP_ERR(p, "output pl_tex_create failed\n");
        return false;
    }

    VkFormat out_format = VK_FORMAT_UNDEFINED;
    p->out_image = pl_vulkan_unwrap(o->gpu, p->out_tex, &out_format, NULL);

    VkImageViewCreateInfo view_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image = p->out_image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = out_format,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
        },
    };
    if (vkCreateImageView(p->dev, &view_info, NULL, &p->out_view) != VK_SUCCESS) {
        MP_ERR(p, "output view creation failed\n");
        return false;
    }

    p->out_ra = talloc_zero(NULL, struct ra_tex);
    if (!mppl_wrap_tex(mapper->ra, p->out_tex, p->out_ra)) {
        MP_ERR(p, "mppl_wrap_tex failed\n");
        return false;
    }

    p->out_w = w;
    p->out_h = h;
    return true;
}

static void wait_last_submit(struct priv *p)
{
    if (!p->last_submit)
        return;
    VkSemaphoreWaitInfo wait = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
        .semaphoreCount = 1,
        .pSemaphores = &p->timeline,
        .pValues = &p->last_submit,
    };
    vkWaitSemaphores(p->dev, &wait, 1000000000ull);
}

static void mapper_uninit(struct ra_hwdec_mapper *mapper)
{
    struct priv *p = mapper->priv;
    struct priv_owner *o = mapper->owner->priv;

    o->AImageReader_setImageListener(o->reader, NULL);

    if (p->dev) {
        wait_last_submit(p);

        free_plane_wrappers(mapper, p->prev_plane_ra);
        free_plane_wrappers(mapper, p->plane_ra);
        destroy_input(p, &p->prev_in_view, &p->prev_in_image, &p->prev_in_mem);
        destroy_input(p, &p->in_view, &p->in_image, &p->in_mem);
        destroy_format_objects(p);
        destroy_output(mapper);

        if (p->acquire_sem)
            vkDestroySemaphore(p->dev, p->acquire_sem, NULL);
        if (p->timeline)
            vkDestroySemaphore(p->dev, p->timeline, NULL);
        if (p->cmd_pool)
            vkDestroyCommandPool(p->dev, p->cmd_pool, NULL);
    }

    if (p->prev_image)
        o->AImage_delete(p->prev_image);
    if (p->image)
        o->AImage_delete(p->image);
    p->prev_image = p->image = NULL;

    mp_mutex_destroy(&p->lock);
    mp_cond_destroy(&p->cond);
}

static void mapper_unmap(struct ra_hwdec_mapper *mapper)
{
    // Teardown is deferred to the next map (or uninit): Vulkan has no
    // implicit sync, so the frame's input image and AImage stay alive until
    // the timeline semaphore proves the GPU consumed them.
}

static bool import_input(struct ra_hwdec_mapper *mapper, AHardwareBuffer *hwbuf,
    const VkAndroidHardwareBufferPropertiesANDROID *props,
    const VkAndroidHardwareBufferFormatPropertiesANDROID *fmt,
    uint32_t w, uint32_t h, bool with_view)
{
    struct priv *p = mapper->priv;

    VkExternalFormatANDROID ext_fmt = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_FORMAT_ANDROID,
        .externalFormat = fmt->format == VK_FORMAT_UNDEFINED
                              ? fmt->externalFormat : 0,
    };
    VkExternalMemoryImageCreateInfo ext_mem = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
        .pNext = &ext_fmt,
        .handleTypes =
            VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID,
    };
    VkImageCreateInfo img_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = &ext_mem,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = fmt->format,
        .extent = { w, h, 1 },
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
    if (vkCreateImage(p->dev, &img_info, NULL, &p->in_image) != VK_SUCCESS) {
        MP_ERR(p, "vkCreateImage(import) failed\n");
        return false;
    }

    int mem_type = -1;
    for (int i = 0; i < 32; i++) {
        if (props->memoryTypeBits & (1u << i)) {
            mem_type = i;
            break;
        }
    }
    if (mem_type < 0)
        return false;

    VkImportAndroidHardwareBufferInfoANDROID import_info = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID,
        .buffer = hwbuf,
    };
    VkMemoryDedicatedAllocateInfo dedicated = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO,
        .pNext = &import_info,
        .image = p->in_image,
    };
    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &dedicated,
        .allocationSize = props->allocationSize,
        .memoryTypeIndex = mem_type,
    };
    if (vkAllocateMemory(p->dev, &alloc_info, NULL, &p->in_mem) != VK_SUCCESS) {
        MP_ERR(p, "vkAllocateMemory(import) failed\n");
        return false;
    }
    if (vkBindImageMemory(p->dev, p->in_image, p->in_mem, 0) != VK_SUCCESS) {
        MP_ERR(p, "vkBindImageMemory failed\n");
        return false;
    }

    if (!with_view)
        return true;

    VkSamplerYcbcrConversionInfo conv_ref = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_YCBCR_CONVERSION_INFO,
        .conversion = p->conv,
    };
    VkImageViewCreateInfo view_info = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = &conv_ref,
        .image = p->in_image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = fmt->format,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
        },
    };
    if (vkCreateImageView(p->dev, &view_info, NULL, &p->in_view) != VK_SUCCESS) {
        MP_ERR(p, "vkCreateImageView(import) failed\n");
        return false;
    }

    return true;
}

// Submits the recorded acquire barrier (+ optional dispatch), waiting on the
// libplacebo hand-off value and the imported acquire fence, signaling
// `signal_value`.
static bool submit_work(struct ra_hwdec_mapper *mapper, uint64_t wait_value,
                        uint64_t signal_value, bool wait_pl)
{
    struct priv *p = mapper->priv;
    struct priv_owner *o = mapper->owner->priv;

    VkSemaphore wait_sems[2];
    uint64_t wait_values[2];
    VkPipelineStageFlags wait_stages[2];
    uint32_t num_waits = 0;

    if (wait_pl) {
        wait_sems[num_waits] = p->timeline;
        wait_values[num_waits] = wait_value;
        wait_stages[num_waits] = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        num_waits++;
    }
    if (p->wait_acquire) {
        wait_sems[num_waits] = p->acquire_sem;
        wait_values[num_waits] = 0;
        wait_stages[num_waits] = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        num_waits++;
        p->wait_acquire = false;
    }

    VkTimelineSemaphoreSubmitInfo tl_submit = {
        .sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO,
        .waitSemaphoreValueCount = num_waits,
        .pWaitSemaphoreValues = wait_values,
        .signalSemaphoreValueCount = 1,
        .pSignalSemaphoreValues = &signal_value,
    };
    VkSubmitInfo submit = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = &tl_submit,
        .waitSemaphoreCount = num_waits,
        .pWaitSemaphores = wait_sems,
        .pWaitDstStageMask = wait_stages,
        .commandBufferCount = 1,
        .pCommandBuffers = &p->cmd,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &p->timeline,
    };
    o->vk->lock_queue(o->vk, p->qf, 0);
    VkResult res = vkQueueSubmit(p->queue, 1, &submit, VK_NULL_HANDLE);
    o->vk->unlock_queue(o->vk, p->qf, 0);
    if (res != VK_SUCCESS) {
        MP_ERR(mapper, "vkQueueSubmit failed: %d\n", res);
        return false;
    }
    p->last_submit = signal_value;
    return true;
}

static void record_acquire_barrier(struct priv *p, uint32_t external_qf,
                                   VkImageLayout new_layout,
                                   VkPipelineStageFlags dst_stage)
{
    VkImageMemoryBarrier acquire = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = new_layout,
        .srcQueueFamilyIndex = external_qf,
        .dstQueueFamilyIndex = p->qf,
        .image = p->in_image,
        .subresourceRange = {
            .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
            .levelCount = 1,
            .layerCount = 1,
        },
    };
    vkCmdPipelineBarrier(p->cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, dst_stage,
        0, 0, NULL, 0, NULL, 1, &acquire);
}

// UNTESTED on external-format-only drivers' counterpart hardware: wraps the
// two planes of a standard multiplanar decoder buffer directly as libplacebo
// textures (true YUV, no repack). Falls back to the repack path on failure.
static bool map_direct(struct ra_hwdec_mapper *mapper,
    const VkAndroidHardwareBufferFormatPropertiesANDROID *fmt,
    uint32_t w, uint32_t h)
{
    struct priv *p = mapper->priv;
    struct priv_owner *o = mapper->owner->priv;

    bool p010 = fmt->format == VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16;
    VkFormat plane_fmts[2] = {
        p010 ? VK_FORMAT_R10X6_UNORM_PACK16 : VK_FORMAT_R8_UNORM,
        p010 ? VK_FORMAT_R10X6G10X6_UNORM_2PACK16 : VK_FORMAT_R8G8_UNORM,
    };
    VkImageAspectFlags aspects[2] = {
        VK_IMAGE_ASPECT_PLANE_0_BIT,
        VK_IMAGE_ASPECT_PLANE_1_BIT,
    };

    pl_tex planes[2] = {0};
    for (int i = 0; i < 2; i++) {
        planes[i] = pl_vulkan_wrap(o->gpu, pl_vulkan_wrap_params(
            .image = p->in_image,
            .aspect = aspects[i],
            .width = i ? (w + 1) / 2 : w,
            .height = i ? (h + 1) / 2 : h,
            .format = plane_fmts[i],
            .usage = VK_IMAGE_USAGE_SAMPLED_BIT,
        ));
        if (!planes[i]) {
            MP_WARN(mapper, "direct plane wrap failed, using repack\n");
            for (int j = 0; j < i; j++)
                pl_tex_destroy(o->gpu, &planes[j]);
            return false;
        }
    }

    vkResetCommandBuffer(p->cmd, 0);
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(p->cmd, &begin);
    record_acquire_barrier(p, o->external_qf,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    vkEndCommandBuffer(p->cmd);

    uint64_t signal_value = ++p->tl_value;
    if (!submit_work(mapper, 0, signal_value, false)) {
        for (int i = 0; i < 2; i++)
            pl_tex_destroy(o->gpu, &planes[i]);
        return false;
    }

    for (int i = 0; i < 2; i++) {
        pl_vulkan_release_ex(o->gpu, pl_vulkan_release_params(
            .tex = planes[i],
            .layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .qf = p->qf,
            .semaphore = (pl_vulkan_sem) {
                .sem = p->timeline,
                .value = signal_value,
            },
        ));
        p->plane_ra[i] = talloc_zero(NULL, struct ra_tex);
        if (!mppl_wrap_tex(mapper->ra, planes[i], p->plane_ra[i])) {
            MP_ERR(mapper, "mppl_wrap_tex(plane %d) failed\n", i);
            return false;
        }
        mapper->tex[i] = p->plane_ra[i];
    }

    mapper->dst_params = mapper->src_params;
    mapper->dst_params.imgfmt = p010 ? IMGFMT_P010 : IMGFMT_NV12;
    mapper->dst_params.hw_subfmt = 0;
    return true;
}

static int mapper_map(struct ra_hwdec_mapper *mapper)
{
    struct priv *p = mapper->priv;
    struct priv_owner *o = mapper->owner->priv;

    {
        if (mapper->src->imgfmt != IMGFMT_MEDIACODEC)
            return -1;
        AVMediaCodecBuffer *buffer = (AVMediaCodecBuffer *)mapper->src->planes[3];
        av_mediacodec_release_buffer(buffer, 1);
    }

    bool image_available = false;
    mp_mutex_lock(&p->lock);
    if (!p->image_available) {
        mp_cond_timedwait(&p->cond, &p->lock, MP_TIME_MS_TO_NS(100));
        if (!p->image_available)
            MP_WARN(mapper, "Waiting for frame timed out!\n");
    }
    image_available = p->image_available;
    p->image_available = false;
    mp_mutex_unlock(&p->lock);

    // Our previous submit provably retired before its resources are touched.
    wait_last_submit(p);
    free_plane_wrappers(mapper, p->prev_plane_ra);
    destroy_input(p, &p->prev_in_view, &p->prev_in_image, &p->prev_in_mem);
    if (p->prev_image) {
        o->AImage_delete(p->prev_image);
        p->prev_image = NULL;
    }

    AImage *image = NULL;
    int fence_fd = -1;
    media_status_t ret;
    if (o->AImageReader_acquireLatestImageAsync) {
        ret = o->AImageReader_acquireLatestImageAsync(o->reader, &image, &fence_fd);
    } else {
        ret = o->AImageReader_acquireLatestImage(o->reader, &image);
    }
    if (ret != AMEDIA_OK) {
        MP_ERR(mapper, "acquireLatestImage failed: %d\n", ret);
        // If we merely timed out waiting return success anyway to avoid
        // flashing frames of render errors.
        return image_available ? -1 : 0;
    }

    p->wait_acquire = false;
    if (fence_fd >= 0 && o->import_sem_fd && p->acquire_sem) {
        VkImportSemaphoreFdInfoKHR sem_import = {
            .sType = VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR,
            .semaphore = p->acquire_sem,
            .flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT,
            .handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT,
            .fd = fence_fd,
        };
        if (o->import_sem_fd(p->dev, &sem_import) == VK_SUCCESS) {
            p->wait_acquire = true;
            fence_fd = -1;  // ownership transferred to the semaphore
        }
    }
    if (fence_fd >= 0) {
        struct pollfd pfd = { .fd = fence_fd, .events = POLLIN };
        poll(&pfd, 1, 100);
        close(fence_fd);
    }

    p->prev_image = p->image;
    memcpy(p->prev_plane_ra, p->plane_ra, sizeof(p->plane_ra));
    memset(p->plane_ra, 0, sizeof(p->plane_ra));
    p->prev_in_view = p->in_view;
    p->prev_in_image = p->in_image;
    p->prev_in_mem = p->in_mem;
    p->in_view = VK_NULL_HANDLE;
    p->in_image = VK_NULL_HANDLE;
    p->in_mem = VK_NULL_HANDLE;
    p->image = image;

    AHardwareBuffer *hwbuf = NULL;
    ret = o->AImage_getHardwareBuffer(p->image, &hwbuf);
    if (ret != AMEDIA_OK || !hwbuf) {
        MP_ERR(mapper, "getHardwareBuffer failed: %d\n", ret);
        return -1;
    }

    AHardwareBuffer_Desc desc;
    o->AHardwareBuffer_describe(hwbuf, &desc);

    VkAndroidHardwareBufferFormatPropertiesANDROID fmt = {
        .sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID,
    };
    VkAndroidHardwareBufferPropertiesANDROID props = {
        .sType = VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID,
        .pNext = &fmt,
    };
    if (o->get_ahb_props(p->dev, hwbuf, &props) != VK_SUCCESS) {
        MP_ERR(mapper, "vkGetAndroidHardwareBufferPropertiesANDROID failed\n");
        return -1;
    }

    bool try_direct =
        fmt.format == VK_FORMAT_G8_B8R8_2PLANE_420_UNORM ||
        fmt.format == VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16;

    if (try_direct) {
        if (!import_input(mapper, hwbuf, &props, &fmt, desc.width, desc.height,
                          false))
            return -1;
        if (map_direct(mapper, &fmt, desc.width, desc.height)) {
            if (!p->direct)
                MP_VERBOSE(mapper, "using the direct multiplanar path\n");
            p->direct = true;
            return 0;
        }
        // Wrap failed: rebuild the input with a ycbcr view and repack.
        destroy_input(p, &p->in_view, &p->in_image, &p->in_mem);
    }
    p->direct = false;

    if (!ensure_format_objects(mapper, &fmt))
        return -1;
    if (!ensure_output(mapper, desc.width, desc.height))
        return -1;
    if (!import_input(mapper, hwbuf, &props, &fmt, desc.width, desc.height, true))
        return -1;

    mapper->dst_params = mapper->src_params;
    mapper->dst_params.imgfmt = IMGFMT_RGB0;
    mapper->dst_params.hw_subfmt = 0;
    mapper->tex[0] = p->out_ra;
    mapper->tex[1] = NULL;

    VkDescriptorImageInfo in_info = {
        .imageView = p->in_view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    VkDescriptorImageInfo out_info = {
        .imageView = p->out_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    VkWriteDescriptorSet writes[] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = p->dset,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .pImageInfo = &in_info,
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = p->dset,
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            .pImageInfo = &out_info,
        },
    };
    vkUpdateDescriptorSets(p->dev, 2, writes, 0, NULL);

    // libplacebo signals `wait_value` once its pending reads of the output
    // texture retire; the compute submit waits on it and signals
    // `signal_value`, which the release below hands back to libplacebo.
    uint64_t wait_value = ++p->tl_value;
    bool held = pl_vulkan_hold_ex(o->gpu, pl_vulkan_hold_params(
        .tex = p->out_tex,
        .layout = VK_IMAGE_LAYOUT_GENERAL,
        .qf = p->qf,
        .semaphore = (pl_vulkan_sem) {
            .sem = p->timeline,
            .value = wait_value,
        },
    ));
    if (!held) {
        MP_ERR(mapper, "pl_vulkan_hold_ex failed\n");
        return -1;
    }

    vkResetCommandBuffer(p->cmd, 0);
    VkCommandBufferBeginInfo begin = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(p->cmd, &begin);
    record_acquire_barrier(p, o->external_qf,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    vkCmdBindPipeline(p->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->pipe);
    vkCmdBindDescriptorSets(p->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p->pll,
                            0, 1, &p->dset, 0, NULL);
    vkCmdDispatch(p->cmd, (desc.width + 7) / 8, (desc.height + 7) / 8, 1);
    vkEndCommandBuffer(p->cmd);

    uint64_t signal_value = ++p->tl_value;
    if (!submit_work(mapper, wait_value, signal_value, true))
        return -1;

    pl_vulkan_release_ex(o->gpu, pl_vulkan_release_params(
        .tex = p->out_tex,
        .layout = VK_IMAGE_LAYOUT_GENERAL,
        .qf = p->qf,
        .semaphore = (pl_vulkan_sem) {
            .sem = p->timeline,
            .value = signal_value,
        },
    ));

    return 0;
}

const struct ra_hwdec_driver ra_hwdec_aimagereader_vk = {
    .name = "aimagereader-vk",
    .priv_size = sizeof(struct priv_owner),
    .imgfmts = {IMGFMT_MEDIACODEC, 0},
    .device_type = AV_HWDEVICE_TYPE_MEDIACODEC,
    .init = init,
    .uninit = uninit,
    .mapper = &(const struct ra_hwdec_mapper_driver){
        .priv_size = sizeof(struct priv),
        .init = mapper_init,
        .uninit = mapper_uninit,
        .map = mapper_map,
        .unmap = mapper_unmap,
    },
};
