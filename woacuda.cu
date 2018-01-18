#include <woacuda.h>


//////////// CUDA FUNCTIONS
__device__ void fobj(joints currpos,float* tmpscore)
{
  *tmpscore=sqrt(currpos.jointsval[0]*currpos.jointsval[0]+currpos.jointsval[1]*currpos.jointsval[1]+currpos.jointsval[2]*currpos.jointsval[2]+currpos.jointsval[3]*currpos.jointsval[3]+currpos.jointsval[4]*currpos.jointsval[4]+currpos.jointsval[5]*currpos.jointsval[5]);
}

__global__ void WaoCycle(boundaries limit,int n_cycles,float* bestscore,joints* bestjoint,float factor)//estrai best ogni ciclo in host
{
  joints Leader_pos;
  float tmpscore;
  float a,a2,A,C,b,l,D_X_rand,X_rand;
  bool chkl1,chkl2,chkl3,chkl4,chkl5,chkl6,chku1,chku2,chku3,chku4,chku5,chku6;
  
  extern __shared__ joints shmem[];
  joints * jointshar  = (joints *)&shmem;
  
  curandState_t state;
  curand_init(clock64() ,threadIdx.x, 0, &state);
  
  jointshar[threadIdx.x].jointsval[0]=curand_uniform(&state)*(limit.joint1b[1]-limit.joint1b[0])+limit.joint1b[0];
  jointshar[threadIdx.x].jointsval[1]=curand_uniform(&state)*(limit.joint2b[1]-limit.joint2b[0])+limit.joint2b[0];
  jointshar[threadIdx.x].jointsval[2]=curand_uniform(&state)*(limit.joint3b[1]-limit.joint3b[0])+limit.joint3b[0];
  jointshar[threadIdx.x].jointsval[3]=curand_uniform(&state)*(limit.joint4b[1]-limit.joint4b[0])+limit.joint4b[0];
  jointshar[threadIdx.x].jointsval[4]=curand_uniform(&state)*(limit.joint5b[1]-limit.joint5b[0])+limit.joint5b[0];
  jointshar[threadIdx.x].jointsval[5]=curand_uniform(&state)*(limit.joint6b[1]-limit.joint6b[0])+limit.joint6b[0];
  for(int cyc=0;cyc<n_cycles;cyc++)
  {
    chkl1 = (jointshar[threadIdx.x].jointsval[0]<limit.joint1b[0]);
    chkl2 = (jointshar[threadIdx.x].jointsval[1]<limit.joint2b[0]);
    chkl3 = (jointshar[threadIdx.x].jointsval[2]<limit.joint3b[0]);
    chkl4 = (jointshar[threadIdx.x].jointsval[3]<limit.joint4b[0]);
    chkl5 = (jointshar[threadIdx.x].jointsval[4]<limit.joint5b[0]);
    chkl6 = (jointshar[threadIdx.x].jointsval[5]<limit.joint6b[0]);
    chku1 = (jointshar[threadIdx.x].jointsval[0]>limit.joint1b[1]);
    chku2 = (jointshar[threadIdx.x].jointsval[1]>limit.joint2b[1]);
    chku3 = (jointshar[threadIdx.x].jointsval[2]>limit.joint3b[1]);
    chku4 = (jointshar[threadIdx.x].jointsval[3]>limit.joint4b[1]);
    chku5 = (jointshar[threadIdx.x].jointsval[4]>limit.joint5b[1]);
    chku6 = (jointshar[threadIdx.x].jointsval[5]>limit.joint6b[1]);
    jointshar[threadIdx.x].jointsval[0] = jointshar[threadIdx.x].jointsval[0]*(!(chkl1+chku1))+(chkl1*limit.joint1b[0])+(chku1*limit.joint1b[1]);
    jointshar[threadIdx.x].jointsval[1] = jointshar[threadIdx.x].jointsval[1]*(!(chkl2+chku2))+(chkl2*limit.joint2b[0])+(chku2*limit.joint2b[1]);
    jointshar[threadIdx.x].jointsval[2] = jointshar[threadIdx.x].jointsval[2]*(!(chkl3+chku3))+(chkl3*limit.joint3b[0])+(chku3*limit.joint3b[1]);
    jointshar[threadIdx.x].jointsval[3] = jointshar[threadIdx.x].jointsval[3]*(!(chkl4+chku4))+(chkl4*limit.joint4b[0])+(chku4*limit.joint4b[1]);
    jointshar[threadIdx.x].jointsval[4] = jointshar[threadIdx.x].jointsval[4]*(!(chkl5+chku5))+(chkl5*limit.joint5b[0])+(chku5*limit.joint5b[1]);
    jointshar[threadIdx.x].jointsval[5] = jointshar[threadIdx.x].jointsval[5]*(!(chkl6+chku6))+(chkl6*limit.joint6b[0])+(chku6*limit.joint6b[1]);
    
    fobj(jointshar[threadIdx.x],&tmpscore);
//     printf("thr %d tmpscore %f \n",threadIdx.x,tmpscore);
    __syncthreads();

    fatomicMin(bestscore,tmpscore);
    __syncthreads();

    if(*bestscore==tmpscore){
      *bestjoint=jointshar[threadIdx.x];
    }

    a  =  2-cyc*((2) /n_cycles);
    a2 = -1+cyc*((-1)/n_cycles);
    
    A=2*a*curand_uniform(&state)-a;
    C=2*curand_uniform(&state);
  
    b=1;
    l=(a2-1)*curand_uniform(&state)+1;
    
    #pragma unroll
    for(int j=0;j<6;j++)
    {
      if(curand_uniform(&state)<0.5)
      {
        if(fabsf(A)>=factor)
        {
          X_rand = jointshar[static_cast<int>(floor((blockDim.x)*curand_uniform(&state)))].jointsval[j]; //blockDim.y e z??
          D_X_rand=abs(C*X_rand-jointshar[threadIdx.x].jointsval[j]);
          jointshar[threadIdx.x].jointsval[j]=X_rand-A*D_X_rand;
        }
        else
        {
          jointshar[threadIdx.x].jointsval[j] = Leader_pos.jointsval[j]-A*abs(C*Leader_pos.jointsval[j]-jointshar[threadIdx.x].jointsval[j]);
        }
      }
      else
      {  
        jointshar[threadIdx.x].jointsval[j] = abs(Leader_pos.jointsval[j]-jointshar[threadIdx.x].jointsval[j])*exp(b*l)*cos(l*2*PI_F)+Leader_pos.jointsval[j];  
      }
    }
    if(threadIdx.x==0) printf("best score : %f \n",*bestscore);
  }
}




///////////CLASS METHODS
WoaCuda::WoaCuda(int nwhales,int ncyc,boundaries limits,float factor)
{
  n_whales=nwhales;
  n_cycles=ncyc;
  jointlimits=limits;
  
  shrbytes=n_whales*sizeof(joints);
  
  cudaMalloc(static_cast<float**>(&deviceBestscore),sizeof(float));
  memset(hostBestscore,0,sizeof(float));
  *hostBestscore=std::numeric_limits<float>::infinity();
  cudaMemcpy(deviceBestscore,hostBestscore,sizeof(float),cudaMemcpyHostToDevice);
  cudaMalloc(static_cast<joints**>(&devicejointbest),sizeof(joints));
  memset(hostjointbest,0,sizeof(joints));
  cudaMemcpy(devicejointbest,hostjointbest,sizeof(joints),cudaMemcpyHostToDevice);
  
  printf("whale number: %lu \n",n_whales);
  printf("cycles      : %lu \n",n_cycles);
  printf("shared bytes: %lu \n",shrbytes);
}

void WoaCuda::RunCycle() //launch cuda kernel 
{
  WaoCycle<<<1,n_whales,shrbytes>>>(jointlimits,n_cycles,deviceBestscore,devicejointbest,factor);//<<<blocks,thread>>>
  if (cudaSuccess != cudaDeviceSynchronize()) {
    printf("ERROR in WaoCycle\n");
    exit(-2);
  }
}

void WoaCuda::Copytohost()
{
  cudaMemcpy(hostBestscore,deviceBestscore,sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(hostjointbest,devicejointbest,sizeof(joints),cudaMemcpyDeviceToHost);
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
  
  int whl=atof(argv[1]);
  int ncyc=atof(argv[2]);
  int fact=1;
  
  WoaCuda testwao(whl,ncyc,limit,fact);//nwhales,cycles,limits,joints
  testwao.RunCycle(); 
  testwao.Copytohost();
   printf("best: %f\n\n",*(testwao.hostBestscore));
  return 0;
}


