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

__device__ void fobj(joints currpos,float* tmpscore)
{
  *tmpscore=sqrt(currpos.jointsval[0]*currpos.jointsval[0]+currpos.jointsval[1]*currpos.jointsval[1]+currpos.jointsval[2]*currpos.jointsval[2]+currpos.jointsval[3]*currpos.jointsval[3]+currpos.jointsval[4]*currpos.jointsval[4]+currpos.jointsval[5]*currpos.jointsval[5]);
}

__global__ void WaoCycle(boundaries limit,int n_cycles,float* bestscore,joints* bestjoint,int n_joints)//rifare in funz of n_joints
{
  float rnd_sel;
  joints jointval;
  joints Leader_pos;
  float tmpscore;
  float a,a2,A,C,b,l,p,D_X_rand,X_rand;
  bool chkl1,chkl2,chkl3,chkl4,chkl5,chkl6,chku1,chku2,chku3,chku4,chku5,chku6,chmin;
  int* tmppnt;
  int rand_leader_index[n_joints];
  
  extern __shared__ joints shmem[];
  joints * jointshar  = (joints *)&shmem;
  
  curandState_t state;
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
    jointval.jointsval[0] = jointval.jointsval[0]*(!(chkl1+chku1))+(chkl1*limit.joint1b[0])+(chku1*limit.joint1b[1]);
    jointval.jointsval[1] = jointval.jointsval[1]*(!(chkl2+chku2))+(chkl2*limit.joint2b[0])+(chku2*limit.joint2b[1]);
    jointval.jointsval[2] = jointval.jointsval[2]*(!(chkl3+chku3))+(chkl3*limit.joint3b[0])+(chku3*limit.joint3b[1]);
    jointval.jointsval[3] = jointval.jointsval[3]*(!(chkl4+chku4))+(chkl4*limit.joint4b[0])+(chku4*limit.joint4b[1]);
    jointval.jointsval[4] = jointval.jointsval[4]*(!(chkl5+chku5))+(chkl5*limit.joint5b[0])+(chku5*limit.joint5b[1]);
    jointval.jointsval[5] = jointval.jointsval[5]*(!(chkl6+chku6))+(chkl6*limit.joint6b[0])+(chku6*limit.joint6b[1]);

    fobj(jointval,&tmpscore);
    printf("thr %d tmpscore %f \n",threadIdx.x,tmpscore);
    __syncthreads();

    fatomicMin(bestscore,tmpscore);
    __syncthreads();

    if(*bestscore==tmpscore){
      printf("thr %d tmpscore %f \n",threadIdx.x,*bestscore);
      *bestjoint=jointval;
    }

    a  =  2-cyc*((2) /n_cycles);
    a2 = -1+cyc*((-1)/n_cycles);
    
    A=2*a*curand_uniform(&state)-a;
    C=2*curand_uniform(&state);
  
    b=1;
    l=(a2-1)*curand_uniform(&state)+1;
    p = curand_uniform(&state);
    
    #pragma unroll
    for(int j=0;j<n_joints;j++){
      if(curand_uniform(&state)<0.5)
      {
        if fabsf(A)>=1{
          X_rand = jointshar[floor(n_whales*curand_uniform(&state)+1)].jointsval[j];
          D_X_rand=abs(C*X_rand-jointshar[threadIdx.x].jointsval[j]);
          jointshar[threadIdx.x].jointsval[j]=X_rand-A*D_X_rand;
        }
        else
        {
          jointshar[threadIdx.x].jointsval[j]=Leader_pos-A*abs(C*Leader_pos-jointshar[threadIdx.x].jointsval[j]);
        }
      }
      else
      {  
        distance2Leader=abs(Leader_pos(j)-Positions(i,j));
        Positions(i,j)=distance2Leader*exp(b.*l).*cos(l.*2*pi)+Leader_pos(j);  
      }
    }
  }
}


///////////CLASS


class WaoCuda
{
  int n_whales;
  int n_cycles;
  int n_joints;
  float factor;
  float *hostBestscore  = static_cast<float *>(malloc(sizeof(float)));
  float *deviceBestscore;
  joints *hostjointbest = static_cast<joints*>(malloc(sizeof(joints)));
  joints *devicejointbest;

  boundaries jointlimits;
  public:

    WaoCuda(int nwhales,int ncyc,boundaries limits,int njnts);
    void RunCycle();
};

///////////CLASS METHODS
WaoCuda::WaoCuda(int nwhales,int ncyc,boundaries limits,int njnts)
{
  n_whales=nwhales;
  n_cycles=ncyc;
  jointlimits=limits;
  n_joints=njnts;
  
  cudaMalloc(static_cast<float**>(&deviceBestscore),sizeof(float));
  memset(hostBestscore,0,sizeof(float));
  *hostBestscore=std::numeric_limits<float>::infinity();
  cudaMemcpy(deviceBestscore,hostBestscore,sizeof(float),cudaMemcpyHostToDevice);
  cudaMalloc(static_cast<joints**>(&devicejointbest),sizeof(joints));
  memset(hostjointbest,0,sizeof(joints));
  cudaMemcpy(devicejointbest,hostjointbest,sizeof(joints),cudaMemcpyHostToDevice);
//   cudaMemcpy(hostArray,deviceArray,bytes,cudaMemcpyDeviceToHost);
  
  
}

void WaoCuda::RunCycle() //launch cuda kernel 
{
  WaoCycle<<<1,10>>>(jointlimits,n_cycles,deviceBestscore,devicejointbest,n_joints);//<<<blocks,thread>>>
  if (cudaSuccess != cudaDeviceSynchronize()) {
    printf("ERROR in WaoCycle\n");
    exit(-2);
  }
}

///////////MAIN

int main(int argc, char *argv[]){
  boundaries limit;
  limit.joint1b[0]=-30;
  limit.joint1b[1]= 30;
  limit.joint2b[0]=-40;
  limit.joint2b[1]= 40;
  limit.joint3b[0]=-40;
  limit.joint3b[1]= 40;
  limit.joint4b[0]=-360;
  limit.joint4b[1]= 360;
  limit.joint5b[0]=-360;
  limit.joint5b[1]= 360;
  limit.joint6b[0]=-360;
  limit.joint6b[1]= 360;
  
  WaoCuda testwao(10,1,limit);
  testwao.RunCycle();
  return 0;
}


