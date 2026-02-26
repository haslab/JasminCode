#include "ds_benchmark.h"
#include "encode.h"
#include "kem.h"
#include "random.h"
#include "param.h"
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#define KEM_TEST_ITERATIONS 100
#define KEM_BENCH_SECONDS 1
#if (scloudplus_l == 128)
#define SYSTEM_NAME "scloud plus 128"
#elif (scloudplus_l == 192)
#define SYSTEM_NAME "scloud plus 192"
#elif (scloudplus_l == 256)
#define SYSTEM_NAME "scloud plus 256"
#endif


extern void j_keygen(uint8_t *, uint8_t *, uint8_t *);
extern void j_enc_derand(uint8_t *, uint8_t *, uint8_t *, uint8_t *);
extern void j_dec(uint8_t *, uint8_t *, uint8_t *);


extern void j_F(uint8_t *, uint8_t *);
extern void j_samplepsi(uint16_t *, uint8_t *);
extern void j_sampleeta1(uint16_t *, uint8_t *);
extern void j_genMat(uint16_t *, uint8_t *);
extern void j_AxStxE(uint16_t *, uint16_t *, uint16_t *, uint16_t *);
extern void j_packpk(uint8_t *, uint16_t *);
extern void j_packsk(uint8_t *, uint16_t *);

extern void j_unpackpk(uint16_t *, uint8_t *);
extern void j_msgencode(uint16_t *, uint8_t *);
extern void j_samplephi(uint16_t *, uint8_t *);
extern void j_sampleeta2(uint16_t *, uint16_t *, uint8_t *);
extern void j_SxAxE(uint16_t *, uint16_t *, uint16_t *, uint16_t *);
extern void j_SxBxExM(uint16_t *, uint16_t *, uint16_t *, uint16_t *, uint16_t *);
extern void j_compressc1(uint16_t *);
extern void j_compressc2(uint16_t *);
extern void j_packc1(uint8_t *, uint16_t *);
extern void j_packc2(uint8_t *, uint16_t *);

extern void j_unpacksk(uint16_t *, uint8_t *);
extern void j_unpackc1(uint16_t *, uint8_t *);
extern void j_unpackc2(uint16_t *, uint8_t *);

extern void j_keygen_plain(uint16_t *, uint8_t *, uint16_t *, uint8_t *);
extern void j_enc_derand_plain(uint16_t *, uint16_t *, uint16_t *, uint8_t *, uint16_t *, uint8_t *);
extern void j_dec_plain(uint8_t *, uint16_t *, uint16_t *, uint16_t *);

extern void j_keygen(uint8_t *, uint8_t *, uint8_t *);
extern void j_enc_derand(uint8_t *, uint8_t *, uint8_t *, uint8_t *);
extern void j_dec(uint8_t *, uint8_t *, uint8_t *);

extern void j_kem_keygen(uint8_t *, uint8_t *);
extern void j_kem_encaps(uint8_t *, uint8_t *, uint8_t *);
extern void j_kem_decaps(uint8_t *, uint8_t *, uint8_t *);

static int kem_test(const char *named_parameters, int iterations)
{
	uint8_t pk[scloudplus_pk];
	uint8_t sk[scloudplus_kem_sk];
	uint8_t ctx[scloudplus_ctx];
	uint8_t ssa[scloudplus_ss];
	uint8_t ssb[scloudplus_ss];
	j_kem_keygen(pk, sk);
	j_kem_encaps(ctx, ssa, pk);
	j_kem_decaps(ssb, sk, ctx);

	printf("====================================================================="
		   "========================================================\n");
	printf("Testing correctness of key encapsulation mechanism (KEM),system %s,"
		   "tests for %d iterations\n",
		   named_parameters, iterations);
	printf("====================================================================="
		   "========================================================\n");

	for (int i = 0; i < KEM_TEST_ITERATIONS; i++)
	{

		j_kem_keygen(pk, sk);
		j_kem_encaps(ctx, ssa, pk);
		j_kem_decaps(ssb, sk, ctx);
		if (memcmp(ssa, ssb, scloudplus_ss) != 0)
		{
			printf("\n");
			for (int i = 0; i < scloudplus_ss; i++)
			{
				printf("%d ", ssa[i]);
			}
			printf("\n");
			for (int i = 0; i < scloudplus_ss; i++)
			{
				printf("%d ", ssb[i]);
			}
			printf("wrong idx is %d", i);
			printf("\n");
			return false;
		}
	}
	printf("Tests PASSED. All session keys matched.\n");

	return true;
}

static void kem_bench(const int seconds)
{
	uint8_t pk[scloudplus_pk];
	uint8_t sk[scloudplus_kem_sk];
	uint8_t ctx[scloudplus_ctx];
	uint8_t ssa[scloudplus_ss];
	uint8_t ssb[scloudplus_ss];

	TIME_OPERATION_SECONDS({ j_kem_keygen(pk, sk); }, "Jasmin Key generation", seconds);

	j_kem_keygen(pk, sk);
	TIME_OPERATION_SECONDS({ j_kem_encaps(ctx, ssa, pk); }, "Jasmin KEM encapsulate", seconds);

	j_kem_encaps(ctx, ssa, pk);
	TIME_OPERATION_SECONDS({ j_kem_decaps(ssb, sk, ctx); }, "Jasmin KEM decapsulate", seconds);

	TIME_OPERATION_SECONDS(
		{
			j_kem_encaps(ctx, ssa, pk);
			j_kem_decaps(ssb, sk, ctx);
		},
		"Jasmin KEM enc and decapsulate", seconds);
}

int main()
{
	int OK = true;

	OK = kem_test(SYSTEM_NAME, KEM_TEST_ITERATIONS);
	if (OK != true)
	{
		goto exit;
	}

	PRINT_TIMER_HEADER
	kem_bench(KEM_BENCH_SECONDS);

exit:
	return (OK == true) ? EXIT_SUCCESS : EXIT_FAILURE;
}
