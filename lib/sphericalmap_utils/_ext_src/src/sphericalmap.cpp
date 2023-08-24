#include "utils.h"
#include "sphericalmap.h"
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

void sphericalmap_kernel_wrapper(int b, int n, int c, int res, const int *phi,
                                     const int *rho, const float *dis,
                                     const float *feat, float *dis_smap, float *feat_smap);

at::Tensor sphericalmap(
    at::Tensor phi,
    at::Tensor rho,
    at::Tensor dis,
    at::Tensor feat,
    const int res
) {

    CHECK_CONTIGUOUS(phi);
    CHECK_CONTIGUOUS(rho);
    CHECK_CONTIGUOUS(dis);
    CHECK_CONTIGUOUS(feat);

    CHECK_IS_INT(phi);
    CHECK_IS_INT(rho);
    CHECK_IS_FLOAT(dis);
    CHECK_IS_FLOAT(feat);

    at::Tensor feat_smap =
        torch::zeros({feat.size(0), res, res, feat.size(2)},
            at::device(feat.device()).dtype(at::ScalarType::Float));
    at::Tensor dis_smap =
        torch::zeros({feat.size(0), res, res},
            at::device(feat.device()).dtype(at::ScalarType::Float));

    if (feat.type().is_cuda()) {
        sphericalmap_kernel_wrapper(feat.size(0), feat.size(1), feat.size(2), res, phi.data<int>(), rho.data<int>(), dis.data<float>(), feat.data<float>(), dis_smap.data<float>(), feat_smap.data<float>());
    } else {
        TORCH_CHECK(false, "CPU not supported");
    }

    // sphericalmap_kernel_wrapper(feat.size(0), feat.size(1), feat.size(2), res, phi.data<int>(), rho.data<int>(), dis.data<float>(), feat.data<float>(), dis_smap.data<float>(), feat_smap.data<float>());

    return feat_smap;
}