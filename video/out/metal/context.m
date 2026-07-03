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
        pl_swapchain_destroy(&mtl->swapchain);
        ctx->ra->fns->destroy(ctx->ra);
        ctx->ra = NULL;
    }

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

    if (!pl_swapchain_resize(p->mtl.swapchain, &w, &h))
        return false;

    ctx->vo->dwidth = w;
    ctx->vo->dheight = h;
    return true;
}

static int metal_control(struct ra_ctx *ctx, int *events, int request, void *arg)
{
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
    if (!mppl_wrap_tex(sw->ctx->ra, frame.fbo, &p->proxy_tex))
        return false;

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
