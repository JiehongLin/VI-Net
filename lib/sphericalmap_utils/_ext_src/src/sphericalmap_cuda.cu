#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.h"


__global__ void sphericalmap_kernel(int b, int n, int c, int res,
                                        const int *__restrict__ phi,
                                        const int *__restrict__ rho,
                                        const float *__restrict__ dis,
                                        const float *__restrict__ feat,
                                        float *__restrict__ dis_smap,
                                        float *__restrict__ feat_smap) {
    int batch_index = blockIdx.x;

    phi += batch_index*n;
    rho += batch_index*n;
    dis += batch_index*n;
    feat += batch_index*n*c;
    dis_smap += batch_index*res*res;
    feat_smap += batch_index*res*res*c;

    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int j = index; j < n; j += stride) {
        int phi_idx = phi[j];
        int rho_idx = rho[j];

        float d1 = dis[j];
        float d2 = dis_smap[rho_idx*res+phi_idx];
        if (d1 <= 0.6) {
            if (d1 > d2) {
                dis_smap[rho_idx*res+phi_idx] = d1;

                for(int k = 0; k < c; k +=1){
                    feat_smap[rho_idx*res*c+phi_idx*c+k] = feat[j*c+k];
                }
                // feat_smap[rho_idx*res*c+phi_idx*c] = feat[j*c];
            }
        }
    }
}


void sphericalmap_kernel_wrapper(int b, int n, int c, int res, const int *phi,
                                     const int *rho, const float *dis,
                                     const float *feat, float *dis_smap, float *feat_smap) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sphericalmap_kernel<<<b, opt_n_threads(n), 0, stream>>>(
        b, n, c, res, phi, rho, dis, feat, dis_smap, feat_smap);

    CUDA_CHECK_ERRORS();
}