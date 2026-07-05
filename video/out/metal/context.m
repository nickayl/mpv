/*
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
 * License along with mpv.  If not, see <https://www.gnu.org/licenses/>.
 */

#include <QuartzCore/CAMetalLayer.h>

#include <libplacebo/metal.h>

#include "common/msg.h"
#include "options/options.h"
#include "video/out/vo.h"
#include "video/out/gpu/context.h"
#include "video/out/placebo/ra_pl.h"
#include "video/out/placebo/utils.h"

#include "context.h"

struct priv {
    struct mpmtl_ctx mtl;
    CAMetalLayer *layer;
    struct ra_tex proxy_tex;
};

static const struct ra_swapchain_fns metal_swapchain;

struct mpmtl_ctx *ra_mtl_ctx_get(struct ra_ctx *ctx)
{
    if (!ctx->swapchain || ctx->swapchain->fns != &metal_swapchain)
        return NULL;

    struct priv *p = ctx->priv;
    return &p->mtl;
}

static void metal_uninit(struct ra_ctx *ctx)
{
    struct priv *p = ctx->priv;
    if (!p)
        return;

    struct mpmtl_ctx *mtl = &p->mtl;

    if (ctx->ra) {
        pl_gpu_finish(mtl->gpu);
        ctx->ra->fns->destroy(ctx->ra);
        ctx->ra = NULL;
    }

    pl_swapchain_destroy(&mtl->swapchain);
    pl_mtl_destroy(&mtl->mtl);
    pl_log_destroy(&mtl->pllog);

    talloc_free(ctx->swapchain);
    ctx->swapchain = NULL;
}

static bool metal_init(struct ra_ctx *ctx)
{
    struct priv *p = ctx->priv = talloc_zero(ctx, struct priv);
    struct mpmtl_ctx *mtl = &p->mtl;
    int msgl = ctx->opts.probing ? MSGL_V : MSGL_ERR;

    if (ctx->vo->opts->WinID == -1) {
        MP_MSG(ctx, msgl, "WinID missing\n");
        goto fail;
    }

    p->layer = (__bridge CAMetalLayer *)(intptr_t)ctx->vo->opts->WinID;
    if (![p->layer isKindOfClass:[CAMetalLayer class]]) {
        MP_MSG(ctx, msgl, "WinID %lld does not reference a CAMetalLayer\n",
               (long long)ctx->vo->opts->WinID);
        goto fail;
    }
    MP_VERBOSE(ctx, "metal: bound CAMetalLayer %p (drawableSize %dx%d, scale %.2f)\n",
               (void *)p->layer, (int)p->layer.drawableSize.width,
               (int)p->layer.drawableSize.height, (double)p->layer.contentsScale);

    mtl->pllog = mppl_log_create(ctx, ctx->log);
    if (!mtl->pllog)
        goto fail;

    mppl_log_set_probing(mtl->pllog, ctx->vo->probing);

    mtl->mtl = pl_mtl_create(mtl->pllog, pl_mtl_params());
    if (!mtl->mtl) {
        MP_MSG(ctx, msgl, "Failed creating Metal device\n");
        goto fail;
    }

    mtl->gpu = mtl->mtl->gpu;
    mppl_log_set_probing(mtl->pllog, false);

    mtl->swapchain = pl_mtl_create_swapchain(mtl->mtl, pl_mtl_swapchain_params(
        .layer = p->layer,
    ));
    if (!mtl->swapchain) {
        MP_MSG(ctx, msgl, "Failed creating Metal swapchain\n");
        goto fail;
    }

    ctx->ra = ra_create_pl(mtl->gpu, ctx->log);
    if (!ctx->ra)
        goto fail;

    struct ra_swapchain *sw = ctx->swapchain = talloc_zero(NULL, struct ra_swapchain);
    sw->ctx = ctx;
    sw->fns = &metal_swapchain;
    sw->priv = p;

    return true;

fail:
    metal_uninit(ctx);
    return false;
}

static bool metal_reconfig(struct ra_ctx *ctx)
{
    struct priv *p = ctx->priv;
    CGSize size = p->layer.drawableSize;
    int w = size.width, h = size.height;

    if (w <= 0 || h <= 0) {
        MP_WARN(ctx, "metal: ignoring reconfig to an empty drawable size %dx%d\n", w, h);
        return false;
    }

    if (!pl_swapchain_resize(p->mtl.swapchain, &w, &h)) {
        MP_ERR(ctx, "metal: pl_swapchain_resize to %dx%d failed\n", (int)size.width, (int)size.height);
        return false;
    }

    if (w != ctx->vo->dwidth || h != ctx->vo->dheight)
        MP_VERBOSE(ctx, "metal: reconfigure %dx%d -> %dx%d\n", ctx->vo->dwidth, ctx->vo->dheight, w, h);

    ctx->vo->dwidth = w;
    ctx->vo->dheight = h;
    return true;
}

static int metal_control(struct ra_ctx *ctx, int *events, int request, void *arg)
{
    struct priv *p = ctx->priv;

    switch (request) {
    case VOCTRL_CHECK_EVENTS: {
        // UIKit resizes the CAMetalLayer out-of-band (device rotation, fullscreen) and never tells
        // mpv, so poll the drawable size each VO iteration: when it changes, run a full reconfig and
        // raise VO_EVENT_RESIZE. Without this gpu-next keeps placing the video at the stale
        // dwidth/dheight and renders it into a corner of the resized surface.
        @autoreleasepool {
            const CGSize size = p->layer.drawableSize;
            const int w = size.width, h = size.height;
            if (w > 0 && h > 0 && (w != ctx->vo->dwidth || h != ctx->vo->dheight)) {
                MP_VERBOSE(ctx, "metal: drawable resized to %dx%d (was %dx%d)\n",
                           w, h, ctx->vo->dwidth, ctx->vo->dheight);
                if (metal_reconfig(ctx))
                    *events |= VO_EVENT_RESIZE;
            }
        }
        return VO_TRUE;
    }
    }

    return VO_NOTIMPL;
}

static bool start_frame(struct ra_swapchain *sw, struct ra_fbo *out_fbo)
{
    struct priv *p = sw->priv;
    struct pl_swapchain_frame frame;

    // If out_fbo is NULL, this was called from vo_gpu_next. Bail out.
    if (out_fbo == NULL)
        return true;

    if (!pl_swapchain_start_frame(p->mtl.swapchain, &frame))
        return false;
    if (!mppl_wrap_tex(sw->ctx->ra, frame.fbo, &p->proxy_tex)) {
        MP_ERR(sw->ctx, "metal: failed wrapping the swapchain framebuffer\n");
        return false;
    }

    *out_fbo = (struct ra_fbo) {
        .tex = &p->proxy_tex,
        .flip = frame.flipped,
    };

    return true;
}

static bool submit_frame(struct ra_swapchain *sw, const struct vo_frame *frame)
{
    struct priv *p = sw->priv;
    return pl_swapchain_submit_frame(p->mtl.swapchain);
}

static void swap_buffers(struct ra_swapchain *sw)
{
    struct priv *p = sw->priv;
    pl_swapchain_swap_buffers(p->mtl.swapchain);
}

static const struct ra_swapchain_fns metal_swapchain = {
    .start_frame  = start_frame,
    .submit_frame = submit_frame,
    .swap_buffers = swap_buffers,
};

const struct ra_ctx_fns ra_ctx_metal = {
    .type        = "metal",
    .name        = "metal",
    .description = "CAMetalLayer (Metal)",
    .reconfig    = metal_reconfig,
    .control     = metal_control,
    .init        = metal_init,
    .uninit      = metal_uninit,
};
