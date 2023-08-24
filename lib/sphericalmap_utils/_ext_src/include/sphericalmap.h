#pragma once
#include <torch/extension.h>

at::Tensor sphericalmap(
    at::Tensor phi,
    at::Tensor rho,
    at::Tensor dis,
    at::Tensor feat,
    const int res
);