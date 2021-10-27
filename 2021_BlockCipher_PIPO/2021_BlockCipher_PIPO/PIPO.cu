#include "type.h"
#include "PIPO.cuh"

__device__ void _core(uint16_t* state, uint16_t* roundkey) {
	
	for (int i = 1; i < ROUND + 1; i++) {

		asm("{\n\t"
			".reg.b16			temp0;					\n\t"
			".reg.b16			temp1;					\n\t"
			".reg.b16			temp2;					\n\t"
			".reg.b16			temp3;					\n\t"
			".reg.b16			temp4;					\n\t"
			".reg.b16			temp5;					\n\t"
			".reg.b16			temp6;					\n\t"
			".reg.b16			temp7;					\n\t"
			".reg.b16			value0;					\n\t"
			".reg.b16			value1;					\n\t"
			".reg.b16			value2;					\n\t"
			".reg.b16			buf;					\n\t"
			
			//S5_1
			//X[5] ^= (X[7] & X[6]);
			"and.b16			temp5, %15, %14;		\n\t"
			"xor.b16			temp5, temp5, %13;		\n\t"
			//X[4] ^= (X[3] & X[5]);
			"and.b16			temp4, %11, temp5;		\n\t"
			"xor.b16			temp4, temp4, %12;		\n\t"
			//X[7] ^= X[4];
			"xor.b16			temp7, %15, temp4;		\n\t"
			//X[6] ^= X[3];
			"xor.b16			temp6, %14, %11;		\n\t"
			//X[3] ^= (X[4] | X[5]);
			"or.b16				temp3, temp4, temp5;	\n\t"
			"xor.b16			temp3, %11, temp3;		\n\t"
			//X[5] ^= X[7];
			"xor.b16			temp5, temp5, temp7;	\n\t"
			//X[4] ^= (X[5] & X[6]);
			"and.b16			buf, temp5, temp6;		\n\t"
			"xor.b16			temp4, temp4, buf;		\n\t"

			//S3
			//X[2] ^= X[1] & X[0];
			"and.b16			temp2, %8, %9;			\n\t"
			"xor.b16			temp2, temp2, %10;		\n\t"

			//X[0] ^= X[2] | X[1];
			"or.b16				temp0, temp2, %9;		\n\t"
			"xor.b16			temp0, temp0, %8;		\n\t"
			//X[1] ^= X[2] | X[0];
			"or.b16				temp1, temp2, temp0;	\n\t"
			"xor.b16			temp1, temp1, %9;		\n\t"
			//X[2] = ~X[2];
			"not.b16			temp2, temp2;			\n\t"

			//Extend XOR
			//X[7] ^= X[1];
			"xor.b16			temp7, temp7, temp1;		\n\t"
			//X[3] ^= X[2];	
			"xor.b16			temp3, temp3, temp2;		\n\t"
			//X[4] ^= X[0];
			"xor.b16			temp4, temp4, temp0;		\n\t"

			//Setting
			"mov.b16			value0, temp7;				\n\t"
			"mov.b16			value1, temp3;				\n\t"
			"mov.b16			value2, temp4;				\n\t"

			//X[6] ^= (T[0] & X[5]);
			"and.b16			buf, value0, temp5;			\n\t"
			"xor.b16			temp6, temp6, buf;			\n\t"

			//T[0] ^= X[6];
			"xor.b16			value0, value0, temp6;		\n\t"

			//X[6] ^= (T[2] | T[1]);
			"or.b16				buf, value2, value1;		\n\t"
			"xor.b16			temp6, temp6, buf;			\n\t"

			//T[1] ^= X[5];
			"xor.b16			value1, value1, temp5;		\n\t"

			//X[5] ^= (X[6] | T[2]);
			"or.b16				buf, temp6, value2;			\n\t"
			"xor.b16			temp5, temp5, buf;			\n\t"

			//T[2] ^= (T[1] & T[0]);
			"and.b16			buf, value1, value0;		\n\t"
			"xor.b16			value2, value2, buf;		\n\t"


			//Truncate XOR and bit change
		    
			//X[2] ^= T[0];
			"xor.b16			temp2, temp2, value0;		\n\t"
			
			//T[0] = X[1] ^ T[2];	
			"xor.b16			value0, temp1, value2;		\n\t"
			
			//X[1] = X[0] ^ T[1];
			"xor.b16			temp1, temp0, value1;		\n\t"
			//EOF SBOX

			//pbox + keyAddition
			//X[0]
			"xor.b16			%0,	temp7, %16;				\n\t"
			
			//X[1]
			"and.b16			value1,	0x0101,	temp1;		\n\t"
			"and.b16			value2,	0xFEFE,	temp1;		\n\t"
			"shl.b16			value1,	value1, 7;			\n\t"
			"shr.b16			value2,	value2, 1;			\n\t"
			"or.b16				value2,	value2, value1;		\n\t"
			"xor.b16			%1,	value2, %17;			\n\t"
			
			//X[2]
			"and.b16			value1,	0x0F0F, temp2;		\n\t"
			"and.b16			value2,	0xF0F0, temp2;		\n\t"
			"shl.b16			value1,	value1, 4;				\n\t"
			"shr.b16			value2,	value2, 4;			\n\t"
			"or.b16				value2,	value2, value1;			\n\t"
			"xor.b16			%2,		value2, %18;			\n\t"

			//X[3]
			"and.b16			value1,	0x1F1F, temp6;		\n\t"
			"and.b16			value2,	0xE0E0, temp6;		\n\t"
			"shl.b16			value1,	value1, 3;				\n\t"
			"shr.b16			value2,	value2, 5;			\n\t"
			"or.b16				value2,	value2, value1;			\n\t"
			"xor.b16			%3,		value2, %19;			\n\t"

			//X[4]
			"and.b16			value1,	0x0303, temp5;		\n\t"
			"and.b16			value2,	0xFCFC, temp5;		\n\t"
			"shl.b16			value1,	value1, 6;				\n\t"
			"shr.b16			value2,	value2, 2;			\n\t"
			"or.b16				value2,	value2, value1;			\n\t"
			"xor.b16			%4,		value2, %20;			\n\t"

			//X[5]
			"and.b16			value1, 0x0707, temp4;			\n\t"
			"and.b16			value2, 0xF8F8, temp4;		\n\t"
			"shl.b16			value1, value1, 5;				\n\t"
			"shr.b16			value2, value2, 3;				\n\t"
			"or.b16				value2,	value2, value1;			\n\t"
			"xor.b16			%5,		value2, %21;			\n\t"

			//X[6]
			"and.b16			value1, 0x7F7F, temp3;			\n\t"
			"and.b16			value2, 0x8080, temp3;		\n\t"
			"shl.b16			value1,	value1, 1;				\n\t"
			"shr.b16			value2,	value2, 7;			\n\t"
			"or.b16				value2,	value2, value1;			\n\t"
			"xor.b16			%6,		value2, %22;			\n\t"

			//X[7]
			"and.b16			value1, 0x3F3F, value0;		\n\t"
			"and.b16			value2, 0xC0C0, value0;		\n\t"
			"shl.b16			value1, value1, 2;				\n\t"
			"shr.b16			value2, value2, 6;				\n\t"
			"or.b16				value2,	value2, value1;			\n\t"
			"xor.b16			%7,		value2, %23;			}\n\t"

			: "=h"(state[0]), "=h"(state[1]), "=h"(state[2]), "=h"(state[3]), "=h"(state[4]), "=h"(state[5]), "=h"(state[6]), "=h"(state[7])
			: "h"(state[0]), "h"(state[1]), "h"(state[2]), "h"(state[3]), "h"(state[4]), "h"(state[5]), "h"(state[6]), "h"(state[7]),
			"h"(roundkey[8 * i]), "h"(roundkey[8 * i + 1]), "h"(roundkey[8 * i + 2]), "h"(roundkey[8 * i + 3]), "h"(roundkey[8 * i + 4]), "h"(roundkey[8 * i + 5]), "h"(roundkey[8 * i + 6]), "h"(roundkey[8 * i + 7])
		);
	}
}

__global__ void _PIPO_u16_ver(uint16_t* pt, uint16_t* roundkey, uint16_t* ct) {

	uint16_t index0 = (blockDim.x * blockIdx.x) + threadIdx.x;
	uint16_t index1 = gridDim.x * blockDim.x;

	uint16_t state[8];
	state[0] = pt[index0];
	state[1] = pt[index0 + index1];
	state[2] = pt[index0 + 2 * index1];
	state[3] = pt[index0 + 3 * index1];
	state[4] = pt[index0 + 4 * index1];
	state[5] = pt[index0 + 5 * index1];
	state[6] = pt[index0 + 6 * index1];
	state[7] = pt[index0 + 7 * index1];

	//initial
	asm("{\n\t"
		"xor.b16			%0, %8,  %16;		\n\t"
		"xor.b16			%1, %9,	 %17;		\n\t"
		"xor.b16			%2, %10, %18;		\n\t"
		"xor.b16			%3, %11, %19;		\n\t"
		"xor.b16			%4, %12, %20;		\n\t"
		"xor.b16			%5, %13, %21;		\n\t"
		"xor.b16			%6, %14, %22;		\n\t"
		"xor.b16			%7, %15, %23;		}\n\t"
		: "=h"(state[0]), "=h"(state[1]), "=h"(state[2]), "=h"(state[3]), "=h"(state[4]), "=h"(state[5]), "=h"(state[6]), "=h"(state[7])
		: "h"(state[0]), "h"(state[1]), "h"(state[2]), "h"(state[3]), "h"(state[4]), "h"(state[5]), "h"(state[6]), "h"(state[7]),
		"h"(roundkey[0]), "h"(roundkey[1]), "h"(roundkey[2]), "h"(roundkey[3]), "h"(roundkey[4]), "h"(roundkey[5]), "h"(roundkey[6]), "h"(roundkey[7])
	);

	_core(state, roundkey);

	ct[index0] = state[0];
	ct[(index1) + index0] = state[1];
	ct[2 * (index1) + index0] = state[2];
	ct[3 * (index1) + index0] = state[3];
	ct[4 * (index1) + index0] = state[4];
	ct[5 * (index1) + index0] = state[5];
	ct[6 * (index1) + index0] = state[6];
	ct[7 * (index1) + index0] = state[7];
}


