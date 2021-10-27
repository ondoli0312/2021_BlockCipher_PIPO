#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifndef PIPO256
#define PIPO128
#endif

#ifdef PIPO128
#define ROUND 13
#define KEYLEN 16
#define INT_NUM 2
#define MASTER_KEY_SIZE 2

#elif defined PIPO256
#define ROUND 
#define KEYLEN 32 
#endif

__global__ void _PIPO_u16_ver(uint16_t* pt, uint16_t* roundkey, uint16_t* ct);