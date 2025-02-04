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
      // CHANGES BEGIN //
      int IX1 = IX(i, j, 0);
      int IX2 = IX(i, j, O + 1);
      int IX3 = IX(i, j, 1);
      int IX4 = IX(i, j, O);

      x[IX1] = b == 3 ? -x[IX3] : x[IX3];
      x[IX2] = b == 3 ? -x[IX4] : x[IX4];
      // CHANGES END //
    }
  }
  for (i = 1; i <= N; i++) {
    for (j = 1; j <= O; j++) {
      // CHANGES BEGIN //
      int IX1 = IX(0, i, j);
      int IX2 = IX(M + 1, i, j);
      int IX3 = IX(1, i, j);
      int IX4 = IX(M, i, j);

      x[IX1] = b == 1 ? -x[IX3] : x[IX3];
      x[IX2] = b == 1 ? -x[IX4] : x[IX4];
      // CHANGES END //
    }
  }
  for (i = 1; i <= M; i++) {
    for (j = 1; j <= O; j++) {
      // CHANGES BEGIN //
      int IX1 = IX(i, 0, j);
      int IX2 = IX(i, N + 1, j);
      int IX3 = IX(i, 1, j);
      int IX4 = IX(i, N, j);

      x[IX1] = b == 2 ? -x[IX3] : x[IX3];
      x[IX2] = b == 2 ? -x[IX4] : x[IX4];
      // CHANGES END //
    }
  }

  //CHANGES BEGIN//
  int IX1 = IX(0, 0, 0);
  int IX2 = IX(M + 1, 0, 0);
  int IX3 = IX(0, N + 1, 0);
  int IX4 = IX(M + 1, N + 1, 0);

  int IX5 = IX(1, 0, 0);
  int IX6 = IX(0, 1, 0); 
  int IX7 = IX(0, 0, 1);

  int IX9 = IX(M, 0, 0);
  int IX10 = IX(M + 1, 1, 0);
  int IX11 = IX(M + 1, 0, 1);

  int IX13 = IX(1, N + 1, 0);
  int IX14 = IX(0, N, 0);
  int IX15 = IX(0, N + 1, 1);

  int IX17 = IX(M, N + 1, 0);
  int IX18 = IX(M + 1, N, 0);
  int IX19 = IX(M + 1, N + 1, 1);

  // Set corners
  x[IX1] = 0.33f * (x[IX5] + x[IX6] + x[IX7]);
  x[IX2] = 0.33f * (x[IX9] + x[IX10] + x[IX11]);
  x[IX3] = 0.33f * (x[IX13] + x[IX14] + x[IX15]);
  x[IX4] = 0.33f * (x[IX17] + x[IX18] + x[IX19]);
}

// red-black solver with convergence check
void lin_solve(int M, int N, int O, int b, float *x, float *x0, float a, float c) {
    float tol = 1e-7, max_c, old_x, change;
    int l = 0;
    
    do {
        max_c = 0.0f;
        for (int i = 1; i <= M; i++) {
            for (int j = 1; j <= N; j++) {
                 for (int k = 1 + (i+j)%2; k <= O; k+=2) {
                    old_x = x[IX(i, j, k)];
                    x[IX(i, j, k)] = (x0[IX(i, j, k)] +
                                      a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                           x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                           x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) /c;
                    change = fabs(x[IX(i, j, k)] - old_x);
                    if(change > max_c) max_c = change;
                }
            }
        }
        
        for (int i = 1; i <= M; i++) {
            for (int j = 1; j <= N; j++) {
                for (int k = 1 + (i+j+1)%2; k <= O; k+=2) {
                    old_x = x[IX(i, j, k)];
                    x[IX(i, j, k)] = (x0[IX(i, j, k)] +
                                      a * (x[IX(i - 1, j, k)] + x[IX(i + 1, j, k)] +
                                           x[IX(i, j - 1, k)] + x[IX(i, j + 1, k)] +
                                           x[IX(i, j, k - 1)] + x[IX(i, j, k + 1)])) /c;
                    change = fabs(x[IX(i, j, k)] - old_x);
                    if(change > max_c) max_c = change;
                }
            }
        }
        set_bnd(M, N, O, b, x);
    } while (max_c > tol && ++l < 20);
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
        // CHANGES BEGIN //
        int IX1 = IX(i, j, k);

        float x = i - (dtX * u[IX1]);
        float y = j - (dtY * v[IX1]);
        float z = k - (dtZ * w[IX1]);

        // Clamp to grid boundaries
        float if1 = M + 0.5f;
        float if2 = N + 0.5f;
        float if3 = O + 0.5f;

        if (x < 0.5f)
          x = 0.5f;
        if (x > if1)
          x = if1;
        if (y < 0.5f)
          y = 0.5f;
        if (y > if2)
          y = if2;
        if (z < 0.5f)
          z = 0.5f;
        if (z > if3)
          z = if3;

        int i0 = (int)x, i1 = i0 + 1;
        int j0 = (int)y, j1 = j0 + 1;
        int k0 = (int)z, k1 = k0 + 1;

        float s1 = x - i0, s0 = 1 - s1;
        float t1 = y - j0, t0 = 1 - t1;
        float u1 = z - k0, u0 = 1 - u1;

        int IX2 = IX(i0, j0, k0);
        int IX3 = IX(i0, j0, k1);
        int IX4 = IX(i0, j1, k0);
        int IX5 = IX(i0, j1, k1);
        int IX6 = IX(i1, j0, k0);
        int IX7 = IX(i1, j0, k1);
        int IX8 = IX(i1, j1, k0);
        int IX9 = IX(i1, j1, k1);

        d[IX1] =
            s0 * (t0 * (u0 * d0[IX2] + u1 * d0[IX3]) +
                  t1 * (u0 * d0[IX4] + u1 * d0[IX5])) +
            s1 * (t0 * (u0 * d0[IX6] + u1 * d0[IX7]) +
                  t1 * (u0 * d0[IX8] + u1 * d0[IX9]));
        // CHANGES END //
      }
    }
  }
  set_bnd(M, N, O, b, d);
}

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
                // CHANGES BEGIN //
                int IX1 = IX(i + 1, j, k);
                int IX2 = IX(i - 1, j, k);
                int IX3 = IX(i, j + 1, k);
                int IX4 = IX(i, j - 1, k);
                int IX5 = IX(i, j, k + 1);
                int IX6 = IX(i, j, k - 1);

                u[idx] -= 0.5f * (p[IX1] - p[IX2]);
                v[idx] -= 0.5f * (p[IX3] - p[IX4]);
                w[idx] -= 0.5f * (p[IX5] - p[IX6]);
                // CHANGES END //
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
