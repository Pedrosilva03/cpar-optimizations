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

static int global_size;

// Fluid simulation arrays
float *d_u, *d_v, *d_w, *d_u_prev, *d_v_prev, *d_w_prev;
float *d_dens, *d_dens_prev;

// Mallocs constantes dos kernels
void initCudaMalloc(int M, int N, int O){
    global_size = compute_size(M, N, O);
    int size = global_size * sizeof(float);

    if (cudaMallocManaged((void**)&d_u, size) != cudaSuccess) {
        printf("Erro ao alocar memória para d_u: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
    if (cudaMallocManaged((void**)&d_v, size) != cudaSuccess) {
        printf("Erro ao alocar memória para d_v: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
    if (cudaMallocManaged((void**)&d_w, size) != cudaSuccess) {
        printf("Erro ao alocar memória para d_w: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
    if (cudaMallocManaged((void**)&d_u_prev, size) != cudaSuccess) {
        printf("Erro ao alocar memória para d_u_prev: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
    if (cudaMallocManaged((void**)&d_v_prev, size) != cudaSuccess) {
        printf("Erro ao alocar memória para d_v_prev: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
    if (cudaMallocManaged((void**)&d_w_prev, size) != cudaSuccess) {
        printf("Erro ao alocar memória para d_w_prev: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
    if (cudaMallocManaged((void**)&d_dens, size) != cudaSuccess) {
        printf("Erro ao alocar memória para d_dens: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
    if (cudaMallocManaged((void**)&d_dens_prev, size) != cudaSuccess) {
        printf("Erro ao alocar memória para d_dens_prev: %s\n", cudaGetErrorString(cudaGetLastError()));
    }
}

void cudaHostToDevice(float* u, float* v, float* w, float* u_prev, float* v_prev, float* w_prev, float* dens, float* dens_prev){
    int size = global_size * sizeof(float);
    if(u != nullptr) cudaMemcpy(d_u, u, size, cudaMemcpyHostToDevice);
    if(v != nullptr) cudaMemcpy(d_v, v, size, cudaMemcpyHostToDevice);
    if(w != nullptr) cudaMemcpy(d_w, w, size, cudaMemcpyHostToDevice);
    if(u_prev != nullptr) cudaMemcpy(d_u_prev, u_prev, size, cudaMemcpyHostToDevice);
    if(v_prev != nullptr) cudaMemcpy(d_v_prev, v_prev, size, cudaMemcpyHostToDevice);
    if(w_prev != nullptr) cudaMemcpy(d_w_prev, w_prev, size, cudaMemcpyHostToDevice);

    if(dens != nullptr) cudaMemcpy(d_dens, dens, size, cudaMemcpyHostToDevice);
    if(dens_prev != nullptr) cudaMemcpy(d_dens_prev, dens_prev, size, cudaMemcpyHostToDevice);
}

void cudaDeviceToHost(float* u, float* v, float* w, float* u_prev, float* v_prev, float* w_prev, float* dens, float* dens_prev){
    int size = global_size * sizeof(float);
    if(u != nullptr) cudaMemcpy(u, d_u, size, cudaMemcpyDeviceToHost);
    if(v != nullptr) cudaMemcpy(v, d_v, size, cudaMemcpyDeviceToHost);
    if(w != nullptr) cudaMemcpy(w, d_w, size, cudaMemcpyDeviceToHost);
    if(u_prev != nullptr) cudaMemcpy(u_prev, d_u_prev, size, cudaMemcpyDeviceToHost);
    if(v_prev != nullptr) cudaMemcpy(v_prev, d_v_prev, size, cudaMemcpyDeviceToHost);
    if(w_prev != nullptr) cudaMemcpy(w_prev, d_w_prev, size, cudaMemcpyDeviceToHost);

    if(dens != nullptr) cudaMemcpy(dens, d_dens, size, cudaMemcpyDeviceToHost);
    if(dens_prev != nullptr) cudaMemcpy(dens_prev, d_dens_prev, size, cudaMemcpyDeviceToHost);
}

// Liberta os mallocs constantes
void freeCudaMalloc(){
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_w);
    cudaFree(d_u_prev);
    cudaFree(d_v_prev);
    cudaFree(d_w_prev);

    cudaFree(d_dens);
    cudaFree(d_dens_prev);
}

__global__ void add_source_kernel(int M, int N, int O, float *x, float *s, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = (M + 2) * (N + 2) * (O + 2);

    if (idx < size) {
        x[idx] += dt * s[idx];
    }
}

// Para já mais rapida que o kernel então é a utilizada
void add_source(int M, int N, int O, float *x, float *s, float dt) {
    int threadsPerBlock = 256;
    int numBlocks = (global_size + threadsPerBlock - 1) / threadsPerBlock;

    add_source_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, x, s, dt);
    cudaDeviceSynchronize();
    /*int size = (M + 2) * (N + 2) * (O + 2);
    for (int i = 0; i < size; i++) {
        x[i] += dt * s[i];
    }*/
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

// Acho que não é utilizada neste momento. Todas as funções chamam diretamente o kernel
void set_bnd(int M, int N, int O, int b, float *x) {
    // Configuração dos kernels
    int size = global_size * sizeof(float);
    dim3 threadsPerBlock(64, 8, 2);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);
    
    //cudaMemcpy(new_x, x, size, cudaMemcpyHostToDevice);

    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, x);
    cudaDeviceSynchronize();

    //cudaMemcpy(x, new_x, size, cudaMemcpyDeviceToHost);

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
    float max_change;
    float* d_max_change;

    cudaMallocManaged((void**)&d_max_change, sizeof(float));

    // Configuração dos kernels
    dim3 threadsPerBlock(64, 8, 2);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

    float inv_c = 1.0f / c;
    int iterations = 0;

    // Iterar até atingir a tolerância
    do {
        *d_max_change = 0.0f;

        // Fase Red
        lin_solve_red_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, x, x0, a, inv_c, d_max_change);
        cudaDeviceSynchronize();

        // Fase Black
        lin_solve_black_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, x, x0, a, inv_c, d_max_change);
        cudaDeviceSynchronize();

        // Aplicar condições de contorno (não é preciso chamar o setup porque o array já está na GPU)
        set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, x);
        cudaDeviceSynchronize();

    } while (*d_max_change > tol && ++iterations < 20);
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
    // Configuração de dimensões dos blocos e grades
    dim3 threadsPerBlock(64, 8, 2);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // Lançar o kernel
    advect_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, h_d, h_d0, h_u, h_v, h_w, dt);
    cudaDeviceSynchronize();

    // Aplicar condições de contorno (não é preciso chamar o setup porque o array já está na GPU)
    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, b, h_d);
    cudaDeviceSynchronize();
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
    // Configuração de dimensões dos blocos e grades
    dim3 threadsPerBlock(64, 8, 2);
    dim3 numBlocks((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (O + threadsPerBlock.z - 1) / threadsPerBlock.z);

    float inv_max_dim = 1.0f / max(M, max(N, O));

    // Calcular divergência e inicializar pressão
    project_divergence_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, h_u, h_v, h_w, h_p, h_div, inv_max_dim);
    cudaDeviceSynchronize();

    // Aplicar condições de contorno (não é preciso chamar o setup porque o array já está na GPU)
    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 0, h_div);
    cudaDeviceSynchronize();

    // Aplicar condições de contorno (não é preciso chamar o setup porque o array já está na GPU)
    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 0, h_p);
    cudaDeviceSynchronize();

    // Resolver equação linear para pressão
    lin_solve(M, N, O, 0, h_p, h_div, 1, 6);

    // Atualizar campos de velocidade
    project_update_velocity_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, h_u, h_v, h_w, h_p);
    cudaDeviceSynchronize();

    // Ajustar bordas para os campos de velocidade
    // Aplicar condições de contorno (não é preciso chamar o setup porque o array já está na GPU)
    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 1, h_u);
    cudaDeviceSynchronize();

    // Aplicar condições de contorno (não é preciso chamar o setup porque o array já está na GPU)
    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 2, h_v);
    cudaDeviceSynchronize();

    // Aplicar condições de contorno (não é preciso chamar o setup porque o array já está na GPU)
    set_bnd_kernel<<<numBlocks, threadsPerBlock>>>(M, N, O, 3, h_w);
    cudaDeviceSynchronize();
}

// Step function for density
void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v,
               float *w, float diff, float dt) {
  add_source(M, N, O, d_dens, d_dens_prev, dt);
  SWAP(d_dens_prev, d_dens);
  diffuse(M, N, O, 0, d_dens, d_dens_prev, diff, dt);
  SWAP(d_dens_prev, d_dens);
  advect(M, N, O, 0, d_dens, d_dens_prev, d_u, d_v, d_w, dt);
}

// Step function for velocity
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0,
              float *v0, float *w0, float visc, float dt) {
  add_source(M, N, O, d_u, d_u_prev, dt);
  add_source(M, N, O, d_v, d_v_prev, dt);
  add_source(M, N, O, d_w, d_w_prev, dt);
  SWAP(d_u_prev, d_u);
  diffuse(M, N, O, 1, d_u, d_u_prev, visc, dt);
  SWAP(d_v_prev, d_v);
  diffuse(M, N, O, 2, d_v, d_v_prev, visc, dt);
  SWAP(d_w_prev, d_w);
  diffuse(M, N, O, 3, d_w, d_w_prev, visc, dt);
  project(M, N, O, d_u, d_v, d_w, d_u_prev, d_v_prev);
  SWAP(d_u_prev, d_u);
  SWAP(d_v_prev, d_v);
  SWAP(d_w_prev, d_w);
  advect(M, N, O, 1, d_u, d_u_prev, d_u_prev, d_v_prev, d_w_prev, dt);
  advect(M, N, O, 2, d_v, d_v_prev, d_u_prev, d_v_prev, d_w_prev, dt);
  advect(M, N, O, 3, d_w, d_w_prev, d_u_prev, d_v_prev, d_w_prev, dt);
  project(M, N, O, d_u, d_v, d_w, d_u_prev, d_v_prev);
}