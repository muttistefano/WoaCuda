#include <stdio.h>
#include <string>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits>
#include <fstream>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <unistd.h>
#include <ctime>
#include <cstdlib>
#include <stdint.h>
#include <cstdio>
#include <sys/mman.h>
#include <cooperative_groups.h>
#define PI_F 3.141592654f


struct joints{
    float jointsval[6];
    bool ch = false;
    float ph = 0.0;
};

struct boundaries{
    float joint1b[2];
    float joint2b[2];
    float joint3b[2];
    float joint4b[2];
    float joint5b[2];
    float joint6b[2];
};

__device__ __forceinline__ float atomicMul(float* address, float val)
{
  int32_t* address_as_int = reinterpret_cast<int32_t*>(address);
  int32_t old = *address_as_int, assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_int, assumed, __float_as_int(val * __int_as_float(assumed)));
  } while (assumed != old);
  return __int_as_float(old);
}

__device__ static float fatomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,__float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ static float fatomicMax(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,__float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

////////////// CLASS DEFINITION
class WoaCuda
{
  int n_whales;
  int n_cycles;
  int n_joints;
  float factor;
  
  float *deviceBestscore;
  
  joints *hostjointbest = static_cast<joints*>(malloc(sizeof(joints)));
  joints *devicejointbest;

  boundaries jointlimits;
  
  size_t shrbytes;
  
  public:

    float *hostBestscore  = static_cast<float *>(malloc(sizeof(float)));
    
    WoaCuda(int nwhales,int ncyc,boundaries limits,float factor);
    void RunCycle();
    void Copytohost();
};