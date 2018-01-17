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


//////////DEVICE FUNCTIONS 

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

__device__ void fobj(joints currpos,float* tmpscore)
{
  
}

__global__ void WaoCycle(boundaries limit,int n_cycles)
{
  float rnd_sel;
  joints jointval;
  joints Leader_pos;
  float tmpscore;
  bool chkl1,chkl2,chkl3,chkl4,chkl5,chkl6,chku1,chku2,chku3,chku4,chku5,chku6;

  curandState_t state;
  extern __shared__ int shmem[];
  curand_init(clock64() ,threadIdx.x, 0, &state);
  
  jointval.jointsval[0]=curand_uniform(&state)*(limit.joint1b[1]-limit.joint1b[0])+limit.joint1b[0];
  jointval.jointsval[1]=curand_uniform(&state)*(limit.joint2b[1]-limit.joint2b[0])+limit.joint2b[0];
  jointval.jointsval[2]=curand_uniform(&state)*(limit.joint3b[1]-limit.joint3b[0])+limit.joint3b[0];
  jointval.jointsval[3]=curand_uniform(&state)*(limit.joint4b[1]-limit.joint4b[0])+limit.joint4b[0];
  jointval.jointsval[4]=curand_uniform(&state)*(limit.joint5b[1]-limit.joint5b[0])+limit.joint5b[0];
  jointval.jointsval[5]=curand_uniform(&state)*(limit.joint6b[1]-limit.joint6b[0])+limit.joint6b[0];
  
  for(int cyc=0;cyc<n_cycles;cyc++)
    {
      chkl1 = (jointval.jointsval[0]<limit.joint1b[0]);
      chkl2 = (jointval.jointsval[1]<limit.joint2b[0]);
      chkl3 = (jointval.jointsval[2]<limit.joint3b[0]);
      chkl4 = (jointval.jointsval[3]<limit.joint4b[0]);
      chkl5 = (jointval.jointsval[4]<limit.joint5b[0]);
      chkl6 = (jointval.jointsval[5]<limit.joint6b[0]);
      chku1 = (jointval.jointsval[0]>limit.joint1b[1]);
      chku2 = (jointval.jointsval[1]>limit.joint2b[1]);
      chku3 = (jointval.jointsval[2]>limit.joint3b[1]);
      chku4 = (jointval.jointsval[3]>limit.joint4b[1]);
      chku5 = (jointval.jointsval[4]>limit.joint5b[1]);
      chku6 = (jointval.jointsval[5]>limit.joint6b[1]);
      jointval.jointsval[0]*(!(chkl1+chku1))+(chkl1*limit.joint1b[0])+(chku1*limit.joint1b[1]);
      jointval.jointsval[1]*(!(chkl2+chku2))+(chkl2*limit.joint2b[0])+(chku2*limit.joint2b[1]);
      jointval.jointsval[2]*(!(chkl3+chku3))+(chkl3*limit.joint3b[0])+(chku3*limit.joint3b[1]);
      jointval.jointsval[3]*(!(chkl4+chku4))+(chkl4*limit.joint4b[0])+(chku4*limit.joint4b[1]);
      jointval.jointsval[4]*(!(chkl5+chku5))+(chkl5*limit.joint5b[0])+(chku5*limit.joint5b[1]);
      jointval.jointsval[5]*(!(chkl6+chku6))+(chkl6*limit.joint6b[0])+(chku6*limit.joint6b[1]);
      
      fobj(jointval,&tmpscore);
      
//       if (tmpscore<Leader_score)
// 	  Leader_score=tmpscore
// 	  Leader_pos=Positions(i,:);
//       end
    }
}


///////////CLASS


class WaoCuda
{
  int n_whales;
  int n_cycles;
  float factor;
  float Leader_score;
  float *hostBest       = static_cast<float *>(malloc(sizeof(float)));
  float *deviceBest;
  joints *hostjointbest = static_cast<joints*>(malloc(sizeof(joints)));
  joints *devicejointbest;

  boundaries jointlimits;
  public:

    WaoCuda(int nwhales,int ncyc,boundaries limits);
    void RunCycle();
};

///////////CLASS METHODS
WaoCuda::WaoCuda(int nwhales,int ncyc,boundaries limits)
{
  n_whales=nwhales;
  n_cycles=ncyc;
  jointlimits=limits;
  
  Leader_score=std::numeric_limits<float>::infinity();
  
  cudaMalloc(static_cast<float**>(&deviceBest),sizeof(float));
  memset(hostBest,0,sizeof(float));
  cudaMemcpy(deviceBest,hostBest,sizeof(float),cudaMemcpyHostToDevice);
  cudaMalloc(static_cast<joints**>(&devicejointbest),sizeof(joints));
  memset(hostjointbest,0,sizeof(joints));
  cudaMemcpy(devicejointbest,hostjointbest,sizeof(joints),cudaMemcpyHostToDevice);
//   cudaMemcpy(hostArray,deviceArray,bytes,cudaMemcpyDeviceToHost);
}


void WaoCuda::RunCycle() //launch cuda kernel 
{
  WaoCycle<<<1,1>>>(jointlimits,n_cycles);//<<<blocks,thread>>>
  if (cudaSuccess != cudaDeviceSynchronize()) {
    printf("ERROR in WaoCycle\n");
    exit(-2);
  }
}

///////////MAIN

int main(int argc, char *argv[]){
   
  return 0;
}


