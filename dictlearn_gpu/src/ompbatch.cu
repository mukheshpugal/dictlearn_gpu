__device__ void lsolve(float* mat, float* target, int* indices, int veclen, int matlen, float* result) {
    int i, j;
    float acc;
    for (i = 0; i < veclen; i++) {
        acc = 0;
        for (j = 0; j < i; j++)
            acc += mat[i * matlen + j] * result[j];
        result[i] = (target[indices[i]] - acc) / mat[(matlen + 1) * i];
    }
}

__device__ void usolve(float* mat, float* target, int* indices, int veclen, int matlen, float* result) {
    int i, j;
    float acc;
    for (i = veclen - 1; i >= 0; i--) {
        acc = 0;
        for (j = veclen - 1; j > i; j--)
            acc += mat[j * matlen + i] * result[indices[j]];
        result[indices[i]] = (target[i] - acc) / mat[(matlen + 1) * i];
    }
}

__device__ void printvec(float* vec, int veclen) {
    printf("[");
    for (int i = 0; i < veclen - 1; i++)
        printf("%0.4f, ", vec[i]);
    printf("%0.4f]\n", vec[veclen - 1]);
}

__global__ void omp_batch(float* _a_0, float* gram, int sparsity_target, int width, int* _I, float* _L, float* _w, float* _a, float* _gamma, float epsilon) {

    // Resolving pointers
    float* a_0 = _a_0 + width * blockIdx.x;
    int *I = _I + sparsity_target * blockIdx.x;
    float *L = _L + sparsity_target * sparsity_target * blockIdx.x;
    float *w = _w + sparsity_target * blockIdx.x;
    float *a = _a + width * blockIdx.x;
    float *gamma = _gamma + width * blockIdx.x;

    // Initializing
    L[0] = 1;
    float maxval, absval, acc;
    int n, i, j, maxindex;
    for (i = 0; i < width; i++) {
        a[i] = a_0[i];
        gamma[i] = 0;
    }

    // Main loop
    for (n = 0; n < sparsity_target; n++) {

        // Finding argmax
        maxval = 0;
        maxindex = 0;
        for (i = 0; i < width; i++) {
            absval = fabsf(a[i]);
            if (absval > maxval) {
                maxval = absval;
                maxindex = i;
            }
        }

        // Enforce orthogonality. Stop adding coeffs if maxindex is already used
        for (i = 0; i < n; i++)
            if (maxindex == I[i])
                return;

        // Cholesky update
        if (n > 0) {
            lsolve(L, gram + maxindex * width, I, n, sparsity_target, w);
            acc = 1;
            for (i = 0; i < n; i++) {
                L[n * sparsity_target + i] = w[i];
                acc -= w[i] * w[i];
            }
            if (acc <= epsilon)
                return;
            L[n * (sparsity_target + 1)] = sqrtf(acc);
        }
        I[n] = maxindex;

        lsolve(L, a_0, I, n+1, sparsity_target, w);
        usolve(L, w, I, n+1, sparsity_target, gamma);

        // Updating a for next iter
        for (i = 0; i < width; i++) {
            acc = 0;
            for (j = 0; j < n+1; j++)
                acc += gram[i * width + I[j]] * gamma[I[j]];
            a[i] = a_0[i] - acc;
        }
    }
}
