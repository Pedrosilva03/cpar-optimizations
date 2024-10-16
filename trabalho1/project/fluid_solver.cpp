#include "fluid_solver.h"
#include <cmath>

#define IX(i, j, k) ((i) + (M + 2) * (j) + (M + 2) * (N + 2) * (k))
#define SWAP(x0, x)                                                            \
  {                                                                            \
    float *tmp = x0;                                                           \
    x0 = x;                                                                    \
    x = tmp;                                                                   \
  }
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define LINEARSOLVERTIMES 20

// Add sources (density or velocity)
void add_source(int M, int N, int O, float *x, float *s, float dt) {
  int size = (M + 2) * (N + 2) * (O + 2);
  for (int i = 0; i < size; i++) {
    x[i] += dt * s[i];
  }
}

// Set boundary conditions
void set_bnd(int M, int N, int O, int b, float *x) {
  int i, j;

  // Set boundary on faces
  for (i = 1; i <= M; i++) {
    for (j = 1; j <= N; j++) {
      x[IX(i, j, 0)] = b == 3 ? -x[IX(i, j, 1)] : x[IX(i, j, 1)];
      x[IX(i, j, O + 1)] = b == 3 ? -x[IX(i, j, O)] : x[IX(i, j, O)];
    }
  }
  for (i = 1; i <= N; i++) {
    for (j = 1; j <= O; j++) {
      x[IX(0, i, j)] = b == 1 ? -x[IX(1, i, j)] : x[IX(1, i, j)];
      x[IX(M + 1, i, j)] = b == 1 ? -x[IX(M, i, j)] : x[IX(M, i, j)];
    }
  }
  for (i = 1; i <= M; i++) {
    for (j = 1; j <= O; j++) {
      x[IX(i, 0, j)] = b == 2 ? -x[IX(i, 1, j)] : x[IX(i, 1, j)];
      x[IX(i, N + 1, j)] = b == 2 ? -x[IX(i, N, j)] : x[IX(i, N, j)];
    }
  }

  // Set corners
  x[IX(0, 0, 0)] = 0.33f * (x[IX(1, 0, 0)] + x[IX(0, 1, 0)] + x[IX(0, 0, 1)]);
  x[IX(M + 1, 0, 0)] =
      0.33f * (x[IX(M, 0, 0)] + x[IX(M + 1, 1, 0)] + x[IX(M + 1, 0, 1)]);
  x[IX(0, N + 1, 0)] =
      0.33f * (x[IX(1, N + 1, 0)] + x[IX(0, N, 0)] + x[IX(0, N + 1, 1)]);
  x[IX(M + 1, N + 1, 0)] = 0.33f * (x[IX(M, N + 1, 0)] + x[IX(M + 1, N, 0)] +
                                    x[IX(M + 1, N + 1, 1)]);
}

// Linear solve for implicit methods (diffusion)
void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
    // Blocking parameters (can be tuned)
    int blockSize = 8; // Define the size of the blocks

    for (int l = 0; l < LINEARSOLVERTIMES; l++) {
        // Process blocks
        for (int k = 1; k <= O; k += blockSize) {
            for (int j = 1; j <= N; j += blockSize) {
                for (int i = 1; i <= M; i += blockSize) {
                    // Handle the block of size blockSize
                    for (int bk = k; bk < std::min(k + blockSize, O + 1); ++bk) {
                        for (int bj = j; bj < std::min(j + blockSize, N + 1); ++bj) {
                            for (int bi = i; bi < std::min(i + blockSize, M + 1); ++bi) {
                                // Precompute index for the current position
                                int idx = IX(bi, bj, bk);
                                int idx_left   = IX(bi - 1, bj, bk);
                                int idx_right  = IX(bi + 1, bj, bk);
                                int idx_down   = IX(bi, bj - 1, bk);
                                int idx_up     = IX(bi, bj + 1, bk);
                                int idx_back   = IX(bi, bj, bk - 1);
                                int idx_front  = IX(bi, bj, bk + 1);
                                
                                // Use local variables to reduce cache misses
                                float current_x0 = x0[idx];
                                float left_val = x[idx_left];
                                float right_val = x[idx_right];
                                float down_val = x[idx_down];
                                float up_val = x[idx_up];
                                float back_val = x[idx_back];
                                float front_val = x[idx_front];

                                // Calculate the new value using local variables
                                x[idx] = (current_x0 + a * (left_val + right_val + 
                                                              down_val + up_val + 
                                                              back_val + front_val)) / c;
                            }
                        }
                    }
                }
            }
        }
        set_bnd(M, N, O, b, x);
    }
}



// Diffusion step (uses implicit method)
void diffuse(int M, int N, int O, int b, float *x, float *x0, float diff,
             float dt) {
  int max = MAX(MAX(M, N), O);
  float a = dt * diff * max * max;
  lin_solve(M, N, O, b, x, x0, a, 1 + 6 * a);
}

// Advection step (uses velocity field to move quantities)
void advect(int M, int N, int O, int b, float *d, float *d0, float *u, float *v,
            float *w, float dt) {
  float dtX = dt * M, dtY = dt * N, dtZ = dt * O;

  for (int i = 1; i <= M; i++) {
    for (int j = 1; j <= N; j++) {
      for (int k = 1; k <= O; k++) {
        float x = i - dtX * u[IX(i, j, k)];
        float y = j - dtY * v[IX(i, j, k)];
        float z = k - dtZ * w[IX(i, j, k)];

        // Clamp to grid boundaries
        if (x < 0.5f)
          x = 0.5f;
        if (x > M + 0.5f)
          x = M + 0.5f;
        if (y < 0.5f)
          y = 0.5f;
        if (y > N + 0.5f)
          y = N + 0.5f;
        if (z < 0.5f)
          z = 0.5f;
        if (z > O + 0.5f)
          z = O + 0.5f;

        int i0 = (int)x, i1 = i0 + 1;
        int j0 = (int)y, j1 = j0 + 1;
        int k0 = (int)z, k1 = k0 + 1;

        float s1 = x - i0, s0 = 1 - s1;
        float t1 = y - j0, t0 = 1 - t1;
        float u1 = z - k0, u0 = 1 - u1;

        d[IX(i, j, k)] =
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
    // Calculate the divisor once to avoid recalculating
    float maxSize = MAX(M, MAX(N, O));

    // Precompute indices arrays to reduce calculations and improve data locality
    int idx_up, idx_down, idx_left, idx_right, idx_front, idx_back;

    // Compute divergence and initialize pressure
    for (int k = 1; k <= O; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= M; i++) {
                // Precompute indices for current grid point
                int idx = IX(i, j, k);
                idx_up = IX(i + 1, j, k);
                idx_down = IX(i - 1, j, k);
                idx_left = IX(i, j + 1, k);
                idx_right = IX(i, j - 1, k);
                idx_front = IX(i, j, k + 1);
                idx_back = IX(i, j, k - 1);

                // Use local variables to enhance locality
                float u_left   = u[idx_down];  // u[i - 1]
                float u_right  = u[idx_up];     // u[i + 1]
                float v_down   = v[idx_right];  // v[j + 1]
                float v_up     = v[idx_left];   // v[j - 1]
                float w_back   = w[idx_back];    // w[k - 1]
                float w_front  = w[idx_front];   // w[k + 1]

                // Calculate divergence
                div[idx] = -0.5f * (u_right - u_left + v_up - v_down + w_front - w_back) / maxSize;
                p[idx] = 0;  // Initialize pressure
            }
        }
    }

    // Set boundary conditions for divergence and pressure
    set_bnd(M, N, O, 0, div);
    set_bnd(M, N, O, 0, p);
    
    // Solve for pressure using the linear solver
    lin_solve(M, N, O, 0, p, div, 1, 6);

    // Update velocity components based on pressure gradients
    for (int k = 1; k <= O; k++) {
        for (int j = 1; j <= N; j++) {
            for (int i = 1; i <= M; i++) {
                int idx = IX(i, j, k);
                u[idx] -= 0.5f * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
                v[idx] -= 0.5f * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
                w[idx] -= 0.5f * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
            }
        }
    }

    // Set boundary conditions for velocity components
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
