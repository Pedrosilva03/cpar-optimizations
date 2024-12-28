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

__global__ void add_source_kernel(int M, int N, int O, float *x, float *s, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = (M + 2) * (N + 2) * (O + 2);

    if (idx < size) {
        x[idx] += dt * s[idx];
    }
}

void add_source_original(int M, int N, int O, float *x, float *s, float dt){
    float* d_x;
    float* d_s;

    int size = compute_size(M, N, O) * sizeof(float);
    int threadsPerBlock = 256;
    int numBlocks = (compute_size(M, N, O) + threadsPerBlock - 1) / threadsPerBlock;

    cudaMalloc((void**)&d_x, size);
    cudaMalloc((void**)&d_s, size);

    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_s, s, size, cudaMemcpyHostToDevice);

    add_source_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, d_x, d_s, dt);

    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_s);
}

// Para já mais rapida que o kernel então é a utilizada
void add_source(int M, int N, int O, float *x, float *s, float dt) {
    int size = (M + 2) * (N + 2) * (O + 2);
    for (int i = 0; i < size; i++) {
        x[i] += dt * s[i];
    }
}

__global__ void set_bnd_kernel(int M, int N, int O, int b, float* x) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    // Bordas em k = 0 e k = O+1
    if (i >= 1 && i <= M && j >= 1 && j <= N) {
        if (k == 0) x[IX(i, j, 0)] = (b == 3) ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
        if (k == O + 1) x[IX(i, j, O + 1)] = (b == 3) ? -x[IX(i, j, O)] : x[IX(i, j, O)];
    }

    // Bordas em i = 0 e i = M+1
    if (j >= 1 && j <= N && k >= 1 && k <= O) {
        if (i == 0) x[IX(0, j, k)] = (b == 1) ? -x[IX(1, j, k)] : x[IX(1, j, k)];
        if (i == M + 1) x[IX(M + 1, j, k)] = (b == 1) ? -x[IX(M, j, k)] : x[IX(M, j, k)];
    }

    // Bordas em j = 0 e j = N+1
    if (i >= 1 && i <= M && k >= 1 && k <= O) {
        if (j == 0) x[IX(i, 0, k)] = (b == 2) ? -x[IX(i, 1, k)] : x[IX(i, 1, k)];
        if (j == N + 1) x[IX(i, N + 1, k)] = (b == 2) ? -x[IX(i, N, k)] : x[IX(i, N, k)];
    }

    // Cálculo explícito dos cantos
    if (i == 0 && j == 0 && k == 0) 
        x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
    if (i == M + 1 && j == 0 && k == 0) 
        x[IX(M + 1, 0, 0)] = 0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
    if (i == 0 && j == N + 1 && k == 0) 
        x[IX(0, N + 1, 0)] = 0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
    if (i == M + 1 && j == N + 1 && k == 0) 
        x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] + x[IX(M + 1, N + 1, 1)]);
}

void set_bnd(int M, int N, int O, int b, float *x) {
    // Configuração dos kernels
    int size = compute_size(M, N, O) * sizeof(float);
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);
    
    float* new_x;
    cudaMalloc((void**)&new_x, size);
    cudaMemcpy(new_x, x, size, cudaMemcpyHostToDevice);

    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, new_x);
    cudaDeviceSynchronize();

    cudaMemcpy(x, new_x, size, cudaMemcpyDeviceToHost);

    cudaFree(new_x);
    /*int i, j;

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
    x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] + x[IX(M + 1, N + 1, 1)]); */
}

__device__ __forceinline__ float atomicMaxFloat (float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

__global__ void lin_solve_red_kernel(int M, int N, int O, int b, float* x, const float* x0, float a, float inv_c, float* max_change) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {
        if ((i + j + k) % 2 == 1) {  // Red
            int index = IX(i, j, k);
            float old_x = x[index];
            x[index] = (x0[index] +
                        a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                             x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                             x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) * inv_c;
            float change = fabsf(x[index] - old_x);
            //atomicMaxFloat(max_change, change);
            if (change > *max_change) *max_change = change;
        }
    }
}

__global__ void lin_solve_black_kernel(int M, int N, int O, int b, float* x, const float* x0, float a, float inv_c, float* max_change) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {
        if ((i + j + k) % 2 == 0) {  // Black
            int index = IX(i, j, k);
            float old_x = x[index];
            x[index] = (x0[index] +
                        a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                             x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                             x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) * inv_c;
            float change = fabsf(x[index] - old_x);
            //atomicMaxFloat(max_change, change);
            if (change > *max_change) *max_change = change;
        }
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

    cudaMemcpy(d_max_change, &max_change, sizeof(float), cudaMemcpyHostToDevice);

    // Configuração dos kernels
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

    float inv_c = 1.0f / c;
    int iterations = 0;

    // Iterar até atingir a tolerância
    do {
        max_change = 0.0f;
        cudaMemset(d_max_change, 0, sizeof(float));

        // Fase Red
        lin_solve_red_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d_x, d_x0, a, inv_c, d_max_change);
        cudaDeviceSynchronize();

        // Fase Black
        lin_solve_black_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d_x, d_x0, a, inv_c, d_max_change);
        cudaDeviceSynchronize();

        // Copiar `max_change` de volta para o host
        cudaMemcpy(&max_change, d_max_change, sizeof(float), cudaMemcpyDeviceToHost);

        // Aplicar condições de contorno (não é preciso chamar o setup porque o array já está na GPU)
        set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d_x);
        cudaDeviceSynchronize();

    } while (max_change > tol && ++iterations < 20);

    // Copiar resultados de volta para o host
    cudaMemcpy(x, d_x, size, cudaMemcpyDeviceToHost);

    // Libertar memória na GPU
    cudaFree(d_x);
    cudaFree(d_x0);
    cudaFree(d_max_change);
}


void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff, float dt) {
    int max = MAX(M, MAX(N, O));
    float a = dt * diff * max * max;
    lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

__global__ void advect_kernel(int M, int N, int O, int b, float* d, const float* d0, const float* u, const float* v, const float* w, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1; // +1 para evitar bordas
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i > M || j > N || k > O) return;

    int index = IX(i, j, k);
    float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

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

void advect(int M, int N, int O, int b, float* h_d, float* h_d0, float* h_u, float* h_v, float* h_w, float dt) {
    float *d_d = nullptr, *d_d0 = nullptr, *d_u = nullptr, *d_v = nullptr, *d_w = nullptr;
    int size = compute_size(M, N, O) * sizeof(float);

    // Alocar memória na GPU
    cudaMalloc((void**)&d_d, size);
    cudaMalloc((void**)&d_d0, size);
    cudaMalloc((void**)&d_u, size);
    cudaMalloc((void**)&d_v, size);
    cudaMalloc((void**)&d_w, size);

    // Copiar dados do host para a GPU
    cudaMemcpy(d_d, h_d, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d0, h_d0, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, size, cudaMemcpyHostToDevice);

    // Configuração de dimensões dos blocos e grades
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // Lançar o kernel
    advect_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d_d, d_d0, d_u, d_v, d_w, dt);
    cudaDeviceSynchronize();

    // Aplicar condições de contorno (não é preciso chamar o setup porque o array já está na GPU)
    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, d_d);
    cudaDeviceSynchronize();

    // Copiar resultados de volta para o host
    cudaMemcpy(h_d, d_d, size, cudaMemcpyDeviceToHost);

    // Liberar memória na GPU
    cudaFree(d_d);
    cudaFree(d_d0);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
}

__global__ void project_divergence_kernel(int M, int N, int O, float* u, float* v, float* w, float* p, float* div, float inv_max_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {
        int index = IX(i, j, k);
        div[index] = -0.5f * (
            u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)] +
            v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)] +
            w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]
        ) * inv_max_dim;

        p[index] = 0.0f;
    }
}

__global__ void project_update_velocity_kernel(int M, int N, int O, float* u, float* v, float* w, float* p) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (i <= M && j <= N && k <= O) {
        int index = IX(i, j, k);
        u[index] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
        v[index] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
        w[index] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
    }
}

// Projection step to ensure incompressibility (make the velocity field
// divergence-free)
void project(int M, int N, int O, float* h_u, float* h_v, float* h_w, float* h_p, float* h_div) {
    float *d_u = nullptr, *d_v = nullptr, *d_w = nullptr, *d_p = nullptr, *d_div = nullptr;
    int size = compute_size(M, N, O) * sizeof(float);

    // Alocação de memória na GPU
    cudaMalloc((void**)&d_u, size);
    cudaMalloc((void**)&d_v, size);
    cudaMalloc((void**)&d_w, size);
    cudaMalloc((void**)&d_p, size);
    cudaMalloc((void**)&d_div, size);

    // Copiar dados do host para a GPU
    cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, h_p, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_div, h_div, size, cudaMemcpyHostToDevice);

    // Configuração de dimensões dos blocos e grades
    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

    float inv_max_dim = 1.0f / max(M, max(N, O));

    // Calcular divergência e inicializar pressão
    project_divergence_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, d_u, d_v, d_w, d_p, d_div, inv_max_dim);
    cudaDeviceSynchronize();

    // Aplicar condições de contorno (não é preciso chamar o setup porque o array já está na GPU)
    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 0, d_div);
    cudaDeviceSynchronize();

    // Aplicar condições de contorno (não é preciso chamar o setup porque o array já está na GPU)
    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 0, d_p);
    cudaDeviceSynchronize();

    // Resolver equação linear para pressão
    lin_solve(M, N, O, 0, d_p, d_div, 1, 6);

    // Atualizar campos de velocidade
    project_update_velocity_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, d_u, d_v, d_w, d_p);
    cudaDeviceSynchronize();

    // Ajustar bordas para os campos de velocidade
    // Aplicar condições de contorno (não é preciso chamar o setup porque o array já está na GPU)
    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 1, d_u);
    cudaDeviceSynchronize();

    // Aplicar condições de contorno (não é preciso chamar o setup porque o array já está na GPU)
    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 2, d_v);
    cudaDeviceSynchronize();

    // Aplicar condições de contorno (não é preciso chamar o setup porque o array já está na GPU)
    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 3, d_w);
    cudaDeviceSynchronize();

    // Copiar resultados de volta para o host
    cudaMemcpy(h_u, d_u, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_v, d_v, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w, d_w, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_p, d_p, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_div, d_div, size, cudaMemcpyDeviceToHost);

    // Liberar memória na GPU
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_p);
    cudaFree(d_div);
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