#ifndef FLUID_SOLVER_H
#define FLUID_SOLVER_H

// Fluid simulation arrays
extern float *d_u, *d_v, *d_w, *d_u_prev, *d_v_prev, *d_w_prev;
extern float *d_dens, *d_dens_prev;

void initCudaMalloc(int M, int N, int O);
void cudaHostToDevice(float* u, float* v, float* w, float* u_prev, float* v_prev, float* w_prev, float* dens, float* dens_prev);
void cudaDeviceToHost(float* u, float* v, float* w, float* u_prev, float* v_prev, float* w_prev, float* dens, float* dens_prev);
void freeCudaMalloc();
void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v,
               float *w, float diff, float dt);
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0,
              float *v0, float *w0, float visc, float dt);

#endif // FLUID_SOLVER_H
