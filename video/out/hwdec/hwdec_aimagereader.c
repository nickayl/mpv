/*
 * Copyright (c) 2021 sfan5 <sfan5@live.de>
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

#include <assert.h>
#include <dlfcn.h>
#include <unistd.h>
#include <EGL/egl.h>
#include <media/NdkImageReader.h>
#include <android/native_window_jni.h>
#include <libavcodec/mediacodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_mediacodec.h>

#include "misc/jni.h"
#include "osdep/threads.h"
#include "osdep/timer.h"
#include "video/out/gpu/hwdec.h"
#include "video/out/opengl/ra_gl.h"

typedef void *GLeglImageOES;
typedef void *EGLImageKHR;
typedef void *EGLSyncKHR;
#define EGL_NATIVE_BUFFER_ANDROID 0x3140
#define EGL_SYNC_NATIVE_FENCE_ANDROID 0x3144
#define EGL_SYNC_NATIVE_FENCE_FD_ANDROID 0x3145
#define EGL_NO_NATIVE_FENCE_FD_ANDROID -1

struct priv_owner {
    struct mp_hwdec_ctx hwctx;
    AImageReader *reader;
    jobject surface;
    void *lib_handle;

    media_status_t (*AImageReader_newWithUsage)(
        int32_t, int32_t, int32_t, uint64_t, int32_t, AImageReader **);
    media_status_t (*AImageReader_getWindow)(
        AImageReader *, ANativeWindow **);
    media_status_t (*AImageReader_setImageListener)(
        AImageReader *, AImageReader_ImageListener *);
    media_status_t (*AImageReader_acquireLatestImage)(AImageReader *, AImage **);
    // Optional: reports the decoder's fence.
    media_status_t (*AImageReader_acquireLatestImageAsync)(AImageReader *, AImage **, int *);
    void (*AImageReader_delete)(AImageReader *);
    media_status_t (*AImage_getHardwareBuffer)(const AImage *, AHardwareBuffer **);
    void (*AImage_delete)(AImage *);
    void (*AHardwareBuffer_describe)(const AHardwareBuffer *, AHardwareBuffer_Desc *);
    jobject (*ANativeWindow_toSurface)(JNIEnv *, ANativeWindow *);
};

struct priv {
    struct mp_log *log;

    GLuint gl_texture;
    AImage *image;
    EGLImageKHR egl_image;

    mp_mutex lock;
    mp_cond cond;
    bool image_available;

    unsigned frames_mapped, frames_repeated, frames_dropped;
    unsigned fences_waited, fence_import_failures;

    EGLImageKHR (EGLAPIENTRY *CreateImageKHR)(
        EGLDisplay, EGLContext, EGLenum, EGLClientBuffer, const EGLint *);
    EGLBoolean (EGLAPIENTRY *DestroyImageKHR)(EGLDisplay, EGLImageKHR);
    EGLClientBuffer (EGLAPIENTRY *GetNativeClientBufferANDROID)(
        const struct AHardwareBuffer *);
    void (EGLAPIENTRY *EGLImageTargetTexture2DOES)(GLenum, GLeglImageOES);

    // EGL_ANDROID_native_fence_sync; absent on a driver that cannot import the fence.
    EGLSyncKHR (EGLAPIENTRY *CreateSyncKHR)(EGLDisplay, EGLenum, const EGLint *);
    EGLBoolean (EGLAPIENTRY *DestroySyncKHR)(EGLDisplay, EGLSyncKHR);
    EGLint (EGLAPIENTRY *WaitSyncKHR)(EGLDisplay, EGLSyncKHR, EGLint);
};

static const struct { const char *symbol; int offset; } lib_functions[] = {
    { "AImageReader_newWithUsage", offsetof(struct priv_owner, AImageReader_newWithUsage) },
    { "AImageReader_getWindow", offsetof(struct priv_owner, AImageReader_getWindow) },
    { "AImageReader_setImageListener", offsetof(struct priv_owner, AImageReader_setImageListener) },
    { "AImageReader_acquireLatestImage", offsetof(struct priv_owner, AImageReader_acquireLatestImage) },
    { "AImageReader_delete", offsetof(struct priv_owner, AImageReader_delete) },
    { "AImage_getHardwareBuffer", offsetof(struct priv_owner, AImage_getHardwareBuffer) },
    { "AImage_delete", offsetof(struct priv_owner, AImage_delete) },
    { "AHardwareBuffer_describe", offsetof(struct priv_owner, AHardwareBuffer_describe) },
    { "ANativeWindow_toSurface", offsetof(struct priv_owner, ANativeWindow_toSurface) },
    { NULL, 0 },
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
        if (!fun) {
            mp_warn(log, "Could not resolve symbol %s\n", sym);
            return false;
        }

        *(void **) ((uint8_t*)p + lib_functions[i].offset) = fun;
    }

    // Optional, and outside the table because its absence is not a failure.
    p->AImageReader_acquireLatestImageAsync =
        dlsym(p->lib_handle, "AImageReader_acquireLatestImageAsync");
    if (!p->AImageReader_acquireLatestImageAsync) {
        mp_warn(log, "AImageReader_acquireLatestImageAsync is absent: frames are sampled without "
                     "waiting for the decoder's fence and may tear\n");
    }
    return true;
}

static int init(struct ra_hwdec *hw)
{
    struct priv_owner *p = hw->priv;

    if (!ra_is_gl(hw->ra_ctx->ra))
        return -1;
    if (!eglGetCurrentContext())
        return -1;

    const char *exts = eglQueryString(eglGetCurrentDisplay(), EGL_EXTENSIONS);
    if (!gl_check_extension(exts, "EGL_ANDROID_image_native_buffer"))
        return -1;

    JNIEnv *env = MP_JNI_GET_ENV(hw);
    if (!env)
        return -1;

    if (!load_lib_functions(p, hw->log))
        return -1;

    static const char *es2_exts[] = {"GL_OES_EGL_image_external", 0};
    static const char *es3_exts[] = {"GL_OES_EGL_image_external_essl3", 0};
    GL *gl = ra_gl_get(hw->ra_ctx->ra);
    if (gl_check_extension(gl->extensions, es3_exts[0]))
        hw->glsl_extensions = es3_exts;
    else
        hw->glsl_extensions = es2_exts;

    // dummy dimensions, AImageReader only transports hardware buffers
    media_status_t ret = p->AImageReader_newWithUsage(16, 16,
        AIMAGE_FORMAT_PRIVATE, AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE,
        5, &p->reader);
    if (ret != AMEDIA_OK) {
        MP_ERR(hw, "newWithUsage failed: %d\n", ret);
        return -1;
    }
    mp_assert(p->reader);

    ANativeWindow *window;
    ret = p->AImageReader_getWindow(p->reader, &window);
    if (ret != AMEDIA_OK) {
        MP_ERR(hw, "getWindow failed: %d\n", ret);
        return -1;
    }
    mp_assert(window);

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

static bool fence_wait_available(struct priv *p, struct priv_owner *o)
{
    return o->AImageReader_acquireLatestImageAsync && p->CreateSyncKHR &&
           p->DestroySyncKHR && p->WaitSyncKHR;
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
    GL *gl = ra_gl_get(mapper->ra);

    p->log = mapper->log;
    mp_mutex_init(&p->lock);
    mp_cond_init(&p->cond);

    p->CreateImageKHR = (void *)eglGetProcAddress("eglCreateImageKHR");
    p->DestroyImageKHR = (void *)eglGetProcAddress("eglDestroyImageKHR");
    p->GetNativeClientBufferANDROID =
        (void *)eglGetProcAddress("eglGetNativeClientBufferANDROID");
    p->EGLImageTargetTexture2DOES =
        (void *)eglGetProcAddress("glEGLImageTargetTexture2DOES");

    if (!p->CreateImageKHR || !p->DestroyImageKHR ||
        !p->GetNativeClientBufferANDROID || !p->EGLImageTargetTexture2DOES)
        return -1;

    p->CreateSyncKHR = (void *)eglGetProcAddress("eglCreateSyncKHR");
    p->DestroySyncKHR = (void *)eglGetProcAddress("eglDestroySyncKHR");
    p->WaitSyncKHR = (void *)eglGetProcAddress("eglWaitSyncKHR");
    MP_VERBOSE(mapper, "acquire fence: %s\n",
               fence_wait_available(p, o) ? "waited on the GPU" : "UNAVAILABLE, frames may tear");

    AImageReader_ImageListener listener = {
        .context = p,
        .onImageAvailable = image_callback,
    };
    o->AImageReader_setImageListener(o->reader, &listener);

    mapper->dst_params = mapper->src_params;
    mapper->dst_params.imgfmt = IMGFMT_RGB0;
    mapper->dst_params.hw_subfmt = 0;

    // texture creation
    gl->GenTextures(1, &p->gl_texture);
    gl->BindTexture(GL_TEXTURE_EXTERNAL_OES, p->gl_texture);
    gl->TexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    gl->TexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    gl->TexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    gl->TexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    gl->BindTexture(GL_TEXTURE_EXTERNAL_OES, 0);

    struct ra_tex_params params = {
        .dimensions = 2,
        .w = mapper->src_params.w,
        .h = mapper->src_params.h,
        .d = 1,
        .format = ra_find_unorm_format(mapper->ra, 1, 4),
        .render_src = true,
        .src_linear = true,
        .external_oes = true,
    };

    if (params.format->ctype != RA_CTYPE_UNORM)
        return -1;

    mapper->tex[0] = ra_create_wrapped_tex(mapper->ra, &params, p->gl_texture);
    if (!mapper->tex[0])
        return -1;

    return 0;
}

// Makes the GPU wait for the decoder before sampling; takes ownership of the fd.
static void wait_on_acquire_fence(struct ra_hwdec_mapper *mapper, int fence_fd)
{
    struct priv *p = mapper->priv;
    EGLDisplay display = eglGetCurrentDisplay();
    const EGLint attribs[] = {
        EGL_SYNC_NATIVE_FENCE_FD_ANDROID, fence_fd,
        EGL_NONE,
    };

    EGLSyncKHR sync = p->CreateSyncKHR(display, EGL_SYNC_NATIVE_FENCE_ANDROID, attribs);
    if (sync == NULL) {
        MP_WARN(mapper, "could not import the acquire fence, closing it unwaited\n");
        close(fence_fd);
        p->fence_import_failures++;
        return;
    }

    p->WaitSyncKHR(display, sync, 0);
    p->DestroySyncKHR(display, sync);
    p->fences_waited++;
}

// Not wired to unmap: an external texture with no EGLImage samples undefined memory.
static void release_bound_image(struct ra_hwdec_mapper *mapper)
{
    struct priv *p = mapper->priv;
    struct priv_owner *o = mapper->owner->priv;

    if (p->egl_image) {
        p->DestroyImageKHR(eglGetCurrentDisplay(), p->egl_image);
        p->egl_image = 0;
    }

    if (p->image) {
        o->AImage_delete(p->image);
        p->image = NULL;
    }
}

static void mapper_uninit(struct ra_hwdec_mapper *mapper)
{
    struct priv *p = mapper->priv;
    struct priv_owner *o = mapper->owner->priv;
    GL *gl = ra_gl_get(mapper->ra);

    o->AImageReader_setImageListener(o->reader, NULL);
    release_bound_image(mapper);

    MP_VERBOSE(mapper, "mapped %u, repeated %u, dropped %u, fences waited %u, fence imports "
               "failed %u\n", p->frames_mapped, p->frames_repeated, p->frames_dropped,
               p->fences_waited, p->fence_import_failures);

    gl->DeleteTextures(1, &p->gl_texture);
    p->gl_texture = 0;

    ra_tex_free(mapper->ra, &mapper->tex[0]);

    mp_mutex_destroy(&p->lock);
    mp_cond_destroy(&p->cond);
}

static int mapper_map(struct ra_hwdec_mapper *mapper)
{
    struct priv *p = mapper->priv;
    struct priv_owner *o = mapper->owner->priv;
    GL *gl = ra_gl_get(mapper->ra);

    {
        if (mapper->src->imgfmt != IMGFMT_MEDIACODEC) {
            MP_ERR(mapper, "expected a MediaCodec frame, got imgfmt %d\n", mapper->src->imgfmt);
            return -1;
        }
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

    AImage *image = NULL;
    int fence_fd = -1;
    media_status_t ret = fence_wait_available(p, o)
        ? o->AImageReader_acquireLatestImageAsync(o->reader, &image, &fence_fd)
        : o->AImageReader_acquireLatestImage(o->reader, &image);
    if (ret != AMEDIA_OK) {
        if (fence_fd >= 0)
            close(fence_fd);
        // A timeout repeats the bound frame; with nothing bound there is none.
        bool drop = image_available || !p->egl_image;
        if (drop)
            p->frames_dropped++;
        else
            p->frames_repeated++;
        MP_ERR(mapper, "acquireLatestImage failed: %d (signalled=%s, bound=%s) -> %s; "
               "mapped %u, repeated %u, dropped %u\n",
               ret, image_available ? "yes" : "no", p->egl_image ? "yes" : "no",
               drop ? "drop frame" : "repeat bound frame",
               p->frames_mapped, p->frames_repeated, p->frames_dropped);
        return drop ? -1 : 0;
    }
    mp_assert(image);
    p->frames_mapped++;

    release_bound_image(mapper);
    p->image = image;

    AHardwareBuffer *hwbuf = NULL;
    ret = o->AImage_getHardwareBuffer(p->image, &hwbuf);
    if (ret != AMEDIA_OK) {
        MP_ERR(mapper, "getHardwareBuffer failed: %d\n", ret);
        goto fail;
    }
    mp_assert(hwbuf);

    // Update texture size since it may differ
    AHardwareBuffer_Desc d;
    o->AHardwareBuffer_describe(hwbuf, &d);
    if (mapper->tex[0]->params.w != d.width || mapper->tex[0]->params.h != d.height) {
        MP_VERBOSE(p, "Texture dimensions changed to %dx%d\n", d.width, d.height);
        mapper->tex[0]->params.w = d.width;
        mapper->tex[0]->params.h = d.height;
    }

    EGLClientBuffer buf = p->GetNativeClientBufferANDROID(hwbuf);
    if (!buf) {
        MP_ERR(mapper, "eglGetNativeClientBufferANDROID returned nothing for a %dx%d buffer; "
               "mapped %u\n", d.width, d.height, p->frames_mapped);
        goto fail;
    }

    const int attribs[] = {EGL_NONE};
    p->egl_image = p->CreateImageKHR(eglGetCurrentDisplay(),
        EGL_NO_CONTEXT, EGL_NATIVE_BUFFER_ANDROID, buf, attribs);
    if (!p->egl_image) {
        MP_ERR(mapper, "eglCreateImageKHR failed for a %dx%d buffer (EGL 0x%x); mapped %u\n",
               d.width, d.height, eglGetError(), p->frames_mapped);
        goto fail;
    }

    gl->BindTexture(GL_TEXTURE_EXTERNAL_OES, p->gl_texture);
    p->EGLImageTargetTexture2DOES(GL_TEXTURE_EXTERNAL_OES, p->egl_image);
    gl->BindTexture(GL_TEXTURE_EXTERNAL_OES, 0);

    // Must precede the draw that samples the texture.
    if (fence_fd >= 0)
        wait_on_acquire_fence(mapper, fence_fd);

    return 0;

fail:
    // Nothing bound: the next timed-out acquire drops instead of repeating.
    if (fence_fd >= 0)
        close(fence_fd);
    return -1;
}


const struct ra_hwdec_driver ra_hwdec_aimagereader = {
    .name = "aimagereader",
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
    },
};
