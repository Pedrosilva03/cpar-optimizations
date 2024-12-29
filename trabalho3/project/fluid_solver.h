#ifndef FLUID_SOLVER_H
#define FLUID_SOLVER_H

void initCudaMalloc(int M, int N, int O);
void cudaHostToDevice(float* u, float* v, float* w, float* u_prev, float* v_prev, float* w_prev, float* dens, float* dens_prev);
void cudaDeviceToHost(float* u, float* v, float* w, float* u_prev, float* v_prev, float* w_prev, float* dens, float* dens_prev);
void freeCudaMalloc();
void dens_step(int M, int N, int O, float *x, float *x0, float *u, float *v,
               float *w, float diff, float dt);
void vel_step(int M, int N, int O, float *u, float *v, float *w, float *u0,
              float *v0, float *w0, float visc, float dt);

#endif // FLUID_SOLVER_H
