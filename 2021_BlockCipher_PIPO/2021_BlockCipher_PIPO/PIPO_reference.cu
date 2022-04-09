#include <stdio.h>
#include <stdlib.h>
#include "PIPO.cuh"


__device__ void sbox(unsigned char* X) {
	unsigned char T[3] = { 0, };
	X[5] ^= (X[7] & X[6]);
	X[4] ^= (X[3] & X[5]);
	X[7] ^= X[4];
	X[6] ^= X[3];
	X[3] ^= (X[4] | X[5]);
	X[5] ^= X[7];
	X[4] ^= (X[5] & X[6]);
	//S3
	X[2] ^= X[1] & X[0];
	X[0] ^= X[2] | X[1];
	X[1] ^= X[2] | X[0];
	X[2] = ~ X[2];
	// Extend XOR
	X[7] ^= X[1];	X[3] ^= X[2];	X[4] ^= X[0];
	//S5_2
	T[0] = X[7];	T[1] = X[3];	T[2] = X[4];
	X[6] ^= (T[0] & X[5]);
	T[0] ^= X[6];
	X[6] ^= (T[2] | T[1]);
	T[1] ^= X[5];
	X[5] ^= (X[6] | T[2]);
	T[2] ^= (T[1] & T[0]);
	// Truncate XOR and bit change
	X[2] ^= T[0];	T[0] = X[1] ^ T[2];	X[1] = X[0] ^ T[1];	X[0] = X[7];	X[7] = T[0];
	T[1] = X[3];	X[3] = X[6];	X[6] = T[1];
	T[2] = X[4];	X[4] = X[5];	X[5] = T[2];
}

__device__ void pbox(unsigned char* X) {
	X[1] = ((X[1] << 1)) | ((X[1] >> 7));
	X[2] = ((X[2] << 4)) | ((X[2] >> 4));
	X[3] = ((X[3] << 5)) | ((X[3] >> 3));
	X[4] = ((X[4] << 2)) | ((X[4] >> 6));
	X[5] = ((X[5] << 3)) | ((X[5] >> 5));
	X[6] = ((X[6] << 7)) | ((X[6] >> 1));
	X[7] = ((X[7] << 6)) | ((X[7] >> 2));
}

__device__ void keyadd(unsigned char* val, unsigned char* rk) {
	val[0] ^= rk[0];
	val[1] ^= rk[1];
	val[2] ^= rk[2];
	val[3] ^= rk[3];
	val[4] ^= rk[4];
	val[5] ^= rk[5];
	val[6] ^= rk[6];
	val[7] ^= rk[7];
}

__device__ void PIPO_ENC(unsigned char* PLAIN_TEXT, unsigned char* ROUND_KEY, unsigned char* CIPHER_TEXT) {
	int i = 0;
	unsigned char* P = (unsigned char*)PLAIN_TEXT;
	unsigned char* RK = (unsigned char*)ROUND_KEY;
	keyadd(P, RK);
	for (i = 1; i < ROUND + 1; i++)
	{
		//printf("\n  S Before : %02X %02X %02X %02X, %02X %02X %02X %02X", P[7], P[6], P[5], P[4], P[3], P[2], P[1], P[0]);
		sbox(P);
		pbox(P);
		keyadd(P, RK + (i * 8));
	}
	for (i = 0; i < 8; i++)
		CIPHER_TEXT[i] = P[i];
}

__global__ void PIPO(unsigned char* PLAIN_TEXT, unsigned char* ROUND_KEY, unsigned char* CIPHER_TEXT) {
	unsigned int index = (blockDim.x *  blockIdx.x * 8) + 8 * threadIdx.x;
	unsigned char state[8];
	unsigned char cipher[8];
	state[0] = PLAIN_TEXT[index];
	state[1] = PLAIN_TEXT[index + 1];
	state[2] = PLAIN_TEXT[index + 2];
	state[3] = PLAIN_TEXT[index + 3];
	state[4] = PLAIN_TEXT[index + 4];
	state[5] = PLAIN_TEXT[index + 5];
	state[6] = PLAIN_TEXT[index + 6];
	state[7] = PLAIN_TEXT[index + 7];
	PIPO_ENC(state, ROUND_KEY, cipher);
	CIPHER_TEXT[index] = cipher[0];
	CIPHER_TEXT[index + 1] = cipher[1];
	CIPHER_TEXT[index + 2] = cipher[2];
	CIPHER_TEXT[index + 3] = cipher[3];
	CIPHER_TEXT[index + 4] = cipher[4];
	CIPHER_TEXT[index + 5] = cipher[5];
	CIPHER_TEXT[index + 6] = cipher[6];
	CIPHER_TEXT[index + 7] = cipher[7];
}
