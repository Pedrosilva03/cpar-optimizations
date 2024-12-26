#include "fluid_solver.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include <iostream> // For debugging output

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
#define SWAP(x0, x)                                                            \
  {                                                                            \
    float *tmp = x0;                                                           \
    x0 = x;                                                                    \
    x = tmp;                                                                   \
  }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

int compute_size(int M, int N, int O) {
    return (M + 2) * (N + 2) * (O + 2);
}

void add_source(int M, int N, int O, float *x, float *s, float dt) {
    int size = (M + 2) * (N + 2) * (O + 2);
    for (int i = 0; i < size; i++) {
        x[i] += dt * s[i];
    }
}

void set_bnd(int M, int N, int O, int b, float *x) {
    int i, j;

    for (j = 1; j <= N; j++) {
        for (i = 1; i <= M; i++) {
            x[IX(i, j, 0)] = (b == 3) ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
            x[IX(i, j, O + 1)] = (b == 3) ? -x[IX(i, j, O)] : x[IX(i, j, O)];
        }
    }

    for (j = 1; j <= O; j++) {
        for (i = 1; i <= N; i++) {
            x[IX(0, i, j)] = (b == 1) ? -x[IX(1, i, j)] : x[IX(1, i, j)];
            x[IX(M + 1, i, j)] = (b == 1) ? -x[IX(M, i, j)] : x[IX(M, i, j)];
        }
    }

    for (j = 1; j <= O; j++) {
        for (i = 1; i <= M; i++) {
            x[IX(i, 0, j)] = (b == 2) ? -x[IX(i, 1, j)] : x[IX(i, 1, j)];
            x[IX(i, N + 1, j)] = (b == 2) ? -x[IX(i, N, j)] : x[IX(i, N, j)];
        }
    }

    x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]); 
    x[IX(M + 1, 0, 0)] = 0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]); 
    x[IX(0, N + 1, 0)] = 0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]); 
    x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] + x[IX(M + 1, N + 1, 1)]); 
}

__global__ void lin_solve_kernel(int M, int N, int O, int b, float* x, const float* x0, float a, float inv_c, float* max_change) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {
        int index = IX(i, j, k);
        float old_x = x[index];
        x[index] = (x0[index] +
                    a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                         x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                         x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) * inv_c;
        float change = fabs(x[index] - old_x);
        if (change > *max_change) *max_change = change;

        // Debug print to track execution
        if (index < 10) printf("x[%d] = %f\n", index, x[index]);
    }
}

void lin_solve(int M, int N, int O, int b, float* x, const float* x0, float a, float c) {
    float tol = 1e-7f;
    float* d_x = nullptr;
    float* d_x0 = nullptr;
    float* d_max_change = nullptr;
    float max_change;
    int size = compute_size(M, N, O) * sizeof(float);

    // Allocate GPU memory
    if (cudaMalloc((void**)&d_x, size) != cudaSuccess) {
        std::cerr << "Error allocating memory for d_x\n";
    }
    if (cudaMalloc((void**)&d_x0, size) != cudaSuccess) {
        std::cerr << "Error allocating memory for d_x0\n";
    }
    if (cudaMalloc((void**)&d_max_change, sizeof(float)) != cudaSuccess) {
        std::cerr << "Error allocating memory for d_max_change\n";
    }

    // Copy data to GPU
    if (cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "Error copying x to d_x\n";
    }
    if (cudaMemcpy(d_x0, x0, size, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::cerr << "Error copying x0 to d_x0\n";
    }

    // Debug: Check initial data on GPU
    float* debug_x = new float[size / sizeof(float)];
    cudaMemcpy(debug_x, d_x, size, cudaMemcpyDeviceToHost);
    delete[] debug_x;

    // Kernel configuration
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

    float inv_c = 1.0f / c;
    int iterations = 0;

    // Iterate until tolerance is met
    do {
        max_change = 0.0f;
        if (cudaMemcpy(d_max_change, &max_change, sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << "Error resetting d_max_change\n";
        }

        lin_solve_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d_x, d_x0, a, inv_c, d_max_change);
        cudaDeviceSynchronize();

        if (cudaMemcpy(&max_change, d_max_change, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::cerr << "Error copying d_max_change back to host\n";
        }
        iterations++;
    } while (max_change > tol && iterations < 20);

    // Copy results back to host
    if (cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::cerr << "Error copying d_x to x\n";
    }

    // Debug: Check final data on GPU
    float* debug_final_x = new float[size / sizeof(float)];
    cudaMemcpy(debug_final_x, d_x, size, cudaMemcpyDeviceToHost);
    delete[] debug_final_x;

    // Free GPU memory
    cudaFree(d_x);
    cudaFree(d_x0);
    cudaFree(d_max_change);
}


void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff, float dt) {
    int max = MAX(M, MAX(N, O));
    float a = dt * diff * max * max;
    lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v, float *w, float dt) {
    float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

    for (int k = 1; k <= O; k++){
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= M; i++) {
                int index = IX(i, j, k);
                float u_val = u[index], v_val = v[index], w_val = w[index];
                float x = i - dtX * u_val, y = j - dtY * v_val, z = k - dtZ * w_val;

                x = (x < 0.5f) ? 0.5f : (x > M + 0.5f) ? M + 0.5f : x;
                y = (y < 0.5f) ? 0.5f : (y > N + 0.5f) ? N + 0.5f : y;
                z = (z < 0.5f) ? 0.5f : (z > O + 0.5f) ? O + 0.5f : z;

                int i0 = (int)x, i1 = i0 + 1, j0 = (int)y, j1 = j0 + 1, k0 = (int)z, k1 = k0 + 1;
                float s1 = x - i0, s0 = 1 - s1, t1 = y - j0, t0 = 1 - t1, u1 = z - k0, u0 = 1 - u1;

                d[index] = 
                    s0 * (t0 * (u0 * d0[IX(i0, j0, k0)] + u1 * d0[IX(i0, j0, k1)]) + 
                          t1 * (u0 * d0[IX(i0, j1, k0)] + u1 * d0[IX(i0, j1, k1)])) +
                    s1 * (t0 * (u0 * d0[IX(i1, j0, k0)] + u1 * d0[IX(i1, j0, k1)]) + 
                          t1 * (u0 * d0[IX(i1, j1, k0)] + u1 * d0[IX(i1, j1, k1)]));
            }
        }
    }
    set_bnd(M, N, O, b, d);
}

// Projection step to ensure incompressibility (make the velocity field
// divergence-free)
void project(int M, int N, int O, float *u, float *v, float *w, float *p, float *div) {
    int max_dim = MAX(M, MAX(N, O));
    float inv_max_dim = 1.0f / max_dim;

    // Calculate divergence and initialize pressure field
    for (int k = 1; k <= O; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= M; i++) {
                int index = IX(i, j, k);
                div[index] =
                    -0.5f *
                    (u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] +
                     v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)] +
                     w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]) *
                    inv_max_dim;
                p[index] = 0;
            }
        }
    }

    set_bnd(M, N, O, 0, div);
    set_bnd(M, N, O, 0, p);
    lin_solve(M, N, O, 0, p, div, 1, 6);

    // Update velocity fields based on pressure
    for (int k = 1; k <= O; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= M; i++) {
                int index = IX(i, j, k);
                u[index] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
                v[index] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
                w[index] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
            }
        }
    }

    set_bnd(M, N, O, 1, u);
    set_bnd(M, N, O, 2, v);
    set_bnd(M, N, O, 3, w);
}

// Step function for density
void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v,
               float *w, float diff, float dt) {
  add_source(M, N, O, x, x0, dt);
  SWAP(x0, x);
  diffuse(M, N, O, 0, x, x0, diff, dt);
  SWAP(x0, x);
  advect(M, N, O, 0, x, x0, u, v, w, dt);
}

// Step function for velocity
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0,
              float *v0, float *w0, float visc, float dt) {
  add_source(M, N, O, u, u0, dt);
  add_source(M, N, O, v, v0, dt);
  add_source(M, N, O, w, w0, dt);
  SWAP(u0, u);
  diffuse(M, N, O, 1, u, u0, visc, dt);
  SWAP(v0, v);
  diffuse(M, N, O, 2, v, v0, visc, dt);
  SWAP(w0, w);
  diffuse(M, N, O, 3, w, w0, visc, dt);
  project(M, N, O, u, v, w, u0, v0);
  SWAP(u0, u);
  SWAP(v0, v);
  SWAP(w0, w);
  advect(M, N, O, 1, u, u0, u0, v0, w0, dt);
  advect(M, N, O, 2, v, v0, u0, v0, w0, dt);
  advect(M, N, O, 3, w, w0, u0, v0, w0, dt);
  project(M, N, O, u, v, w, u0, v0);
}