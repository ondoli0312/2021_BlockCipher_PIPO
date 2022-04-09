#include "type.h"
#include "PIPO.cuh"
#include <time.h>
#include <malloc.h>
#include <memory.h>

uint16_t CPU_in[32768 * 1024 * 8];
uint16_t buffer[32768 * 1024 * 8];
void random_16bit_gen(uint16_t* state, uint64_t stateLen) {
	for (int i = 0; i < stateLen; i++) {
		state[i] = rand() % 0x10000;
	}
}

void state_Transform(uint16_t* state, uint64_t blocksize, uint64_t threadsize) {

	memcpy(buffer, state, blocksize * threadsize * sizeof(uint16_t) * 8);
	for (uint64_t i = 0; i < blocksize * threadsize; i++) {
		state[i] = buffer[8 * i];
		state[blocksize * threadsize + i] = state[8 * i + 1];
		state[(2 * blocksize * threadsize) + i] = state[8 * i + 2];
		state[(3 * blocksize * threadsize) + i] = state[8 * i + 3];
		state[(4 * blocksize * threadsize) + i] = state[8 * i + 4];
		state[(5 * blocksize * threadsize) + i] = state[8 * i + 5];
		state[(6 * blocksize * threadsize) + i] = state[8 * i + 6];
		state[(7 * blocksize * threadsize) + i] = state[8 * i + 7];
	}
}

void roundkeyGen(uint32_t* roundkey, uint32_t* masterkey) {
	int i = 0;
	int j = 0;
	uint32_t RCON = 0;
	for (i = 0; i < ROUND + 1; i++) {
		for (j = 0; j < INT_NUM; j++)
			roundkey[INT_NUM * i + j] = masterkey[(INT_NUM * i + j) % (MASTER_KEY_SIZE * INT_NUM)];
		roundkey[INT_NUM * i] ^= RCON;
		RCON++;
	}
}

void GPU_PIPO_performance_analysis(uint64_t Blocksize, uint64_t Threadsize)
{
	cudaEvent_t start, stop;
	cudaError_t err;
	float elapsed_time_ms = 0.0f;
	uint32_t masterKey[4] = {0x2E152297, 0x7E1D20AD, 0x779428D2, 0x6DC416DD };
	uint32_t roundKey[(ROUND + 1) * INT_NUM] = { 0, };
	uint16_t round_16key[112];
	uint8_t* rk = NULL;
	uint16_t* GPU_out = NULL;
	uint16_t* GPU_rk = NULL;
	uint16_t* GPU_in = NULL;
	//uint16_t* CPU_in = NULL;

	//CPU_in = (uint16_t*)malloc(Blocksize * Threadsize * 8 * sizeof(uint16_t));
	//if (CPU_in == NULL) {
	//	printf("GPU_PIPO_performance_analysis, CPU_in : Blocksize & Threadsize over the stacks maximum size\n");
	//	return;
	//}
	err = cudaMalloc((void**)&GPU_out, Blocksize * Threadsize * 8 * sizeof(uint16_t));
	if (err != cudaSuccess) {
		printf("GPU_PIPO_performance_analysis, GPU_out : CUDA error : %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc((void**)&GPU_in, Blocksize * Threadsize * 8 * sizeof(uint16_t));
	if (err != cudaSuccess) {
		printf("GPU_PIPO_performance_analysis, GPU_in : CUDA error : %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc((void**)&GPU_rk, 112 * sizeof(uint16_t));
	if (err != cudaSuccess) {
		printf("GPU_PIPO_performance_analysis, GPU_rk : CUDA error : %s\n", cudaGetErrorString(err));
	}
	roundkeyGen(roundKey, masterKey);
	random_16bit_gen(CPU_in, Blocksize * Threadsize * 8);
	rk = (uint8_t*)roundKey;
	for (int i = 0; i < 112; i++) {
		round_16key[i] = (rk[i] << 8) | (rk[i]);
	}
	err = cudaMemcpy(GPU_rk, round_16key, 112 * sizeof(uint16_t), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("GPU_PIPO_performance_analysis, GPU_rk Memcpy : CUDA error : %s\n", cudaGetErrorString(err));
		return;
	}
	err = cudaMemcpy(GPU_in, CPU_in, Blocksize * Threadsize * 8 * sizeof(uint16_t), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("GPU_PIPO_performance_analysis, GPU_in Memcpy : CUDA error : %s\n", cudaGetErrorString(err));
		return;
	}
	//operation start
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for (int i = 0; i < 10000; i++) {
		//state_Transform(CPU_in, Blocksize, Threadsize);
		_PIPO_u16_ver<<<Blocksize, Threadsize>>>(GPU_in, GPU_rk, GPU_out);
	}
	cudaEventRecord(stop, 0);
	cudaDeviceSynchronize();
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	elapsed_time_ms /= 10000;
	elapsed_time_ms = (Blocksize * Threadsize * 8 * sizeof(uint16_t)) / elapsed_time_ms;
	elapsed_time_ms *= 1000;
	elapsed_time_ms /= (1024 * 1024 * 1024);
	printf("File size = %d MB, Grid : %d, Block : %d, Performance : %4.2f GB/s\n", (Blocksize * Threadsize * 8 * 2)/(1024 * 1024), Blocksize, Threadsize, elapsed_time_ms);
	cudaFree(GPU_out);
	cudaFree(GPU_in);
	//free(CPU_in);
}

uint8_t Ref_CPU_in[32768 * 1024 * 8];
uint8_t Ref_buffer[32768 * 1024 * 8];

void random_8bit_gen(uint8_t* state, uint8_t stateLen) {
	for (int i = 0; i < stateLen; i++) {
		state[i] = rand() % 0x100;
	}
}

void GPU_PIPO_reference_performance_analysis(unsigned int Blocksize, unsigned int Threadsize) {
	uint32_t masterKey[4] = { 0x2E152297, 0x7E1D20AD, 0x779428D2, 0x6DC416DD };
	uint32_t roundKey[(ROUND + 1) * INT_NUM] = { 0, };
	uint8_t round_16key[112 * 2];
	cudaEvent_t start, stop;
	cudaError_t err;

	uint8_t* rk = NULL;
	uint8_t* GPU_out = NULL;
	uint8_t* GPU_rk = NULL;
	uint8_t* GPU_in = NULL;

	float elapsed_time_ms = 0.0f;

	roundkeyGen(roundKey, masterKey);
	err = cudaMalloc((void**)&GPU_out, Blocksize * Threadsize * 8 * sizeof(uint8_t));
	if (err != cudaSuccess) {
		printf("GPU_PIPO_performance_analysis, GPU_out : CUDA error : %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc((void**)&GPU_in, Blocksize * Threadsize * 8 * sizeof(uint8_t));
	if (err != cudaSuccess) {
		printf("GPU_PIPO_performance_analysis, GPU_in : CUDA error : %s\n", cudaGetErrorString(err));
	}
	err = cudaMalloc((void**)&GPU_rk, 224 * sizeof(uint8_t));
	if (err != cudaSuccess) {
		printf("GPU_PIPO_performance_analysis, GPU_rk : CUDA error : %s\n", cudaGetErrorString(err));
	}
	rk = (uint8_t*)roundKey;
	random_8bit_gen(Ref_CPU_in, Blocksize * Threadsize * 8);

	err = cudaMemcpy(GPU_rk, rk, 224 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("GPU_PIPO_performance_analysis, GPU_rk Memcpy : CUDA error : %s\n", cudaGetErrorString(err));
		return;
	}
	err = cudaMemcpy(GPU_in, Ref_CPU_in, Blocksize * Threadsize * 8 * sizeof(uint8_t), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) {
		printf("GPU_PIPO_performance_analysis, GPU_in Memcpy : CUDA error : %s\n", cudaGetErrorString(err));
		return;
	}

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	for (int i = 0; i < 10000; i++) {
		//state_Transform(CPU_in, Blocksize, Threadsize);
		PIPO << <Blocksize, Threadsize >> > (GPU_in, GPU_rk, GPU_out);
	}
	cudaEventRecord(stop, 0);
	cudaDeviceSynchronize();
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);
	elapsed_time_ms /= 10000;
	elapsed_time_ms = (Blocksize * Threadsize * 8 * sizeof(uint8_t)) / elapsed_time_ms;
	elapsed_time_ms *= 1000;
	elapsed_time_ms /= (1024 * 1024 * 1024);
	printf("File size = %d MB, Grid : %d, Block : %d, Performance : %4.2f GB/s\n", (Blocksize * Threadsize * 8) / (1024 * 1024), Blocksize, Threadsize, elapsed_time_ms);
	cudaFree(GPU_out);
	cudaFree(GPU_in);


}

int main()
{
	srand(time(NULL));
	GPU_PIPO_performance_analysis(8192, 256);
	GPU_PIPO_performance_analysis(8192, 512);
	GPU_PIPO_performance_analysis(16384, 512);
	GPU_PIPO_performance_analysis(8192, 1024);
	GPU_PIPO_performance_analysis(16384, 1024);
	GPU_PIPO_performance_analysis(32768, 1024);
	//REF
	//GPU_PIPO_reference_performance_analysis(8192, 256);
	//GPU_PIPO_reference_performance_analysis(8192, 512);
	//GPU_PIPO_reference_performance_analysis(16384, 512);
	//GPU_PIPO_reference_performance_analysis(8192, 1024);
	//GPU_PIPO_reference_performance_analysis(16384, 1024);
	//GPU_PIPO_reference_performance_analysis(32768, 1024);


}