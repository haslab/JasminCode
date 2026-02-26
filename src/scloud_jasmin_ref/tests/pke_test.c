#include "ds_benchmark.h"
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

static int pke_test(const char *named_parameters, int iterations)
{
	uint8_t pk[scloudplus_pk];
	uint8_t sk[scloudplus_pke_sk];
	uint8_t ctx[scloudplus_ctx];
	uint8_t alpha[32];
	uint8_t coins[32];
	uint8_t m[scloudplus_ss];
	uint8_t mm[scloudplus_ss];

	printf("====================================================================="
		   "========================================================\n");
	printf("Testing correctness of PKE system %s,"
		   "tests for %d iterations\n",
		   named_parameters, iterations);
	printf("====================================================================="
		   "========================================================\n");

	for (int i = 0; i < KEM_TEST_ITERATIONS; i++)
	{
	  randombytes(alpha, 32);
	  randombytes(coins, 32);
	  randombytes(m, scloudplus_ss);
	  j_keygen(pk, sk, alpha);
	  j_enc_derand(ctx, pk, m, coins);
	  j_dec(mm, sk, ctx);
	  if (memcmp(m, mm, scloudplus_ss) != 0)
	  {
	    printf("\n");
	    for (int i = 0; i < scloudplus_ss; i++)
	    {
	      printf("%d ", m[i]);
	    }
	    printf("\n");
	    for (int i = 0; i < scloudplus_ss; i++)
	    {
	      printf("%d ", mm[i]);
	    }
	    printf("\n");
	    return false;
	  }
	}
	printf("Tests PASSED. All messages matched.\n");

	return true;
}

static void pke_bench(const int seconds)
{
	uint8_t pk[scloudplus_pk];
	uint8_t sk[scloudplus_pke_sk];
	uint8_t m[scloudplus_ss];
	uint8_t mm[scloudplus_ss];
	uint8_t ctx[scloudplus_ctx];
	uint8_t alpha[32];
	uint8_t coins[32];

	randombytes(alpha, 32);
	randombytes(coins, 32);

	TIME_OPERATION_SECONDS({ j_keygen(pk, sk, alpha); }, "Key generation", seconds);

	j_keygen(pk, sk, alpha);
	randombytes(m, scloudplus_ss);
	TIME_OPERATION_SECONDS({ j_enc_derand(ctx, pk, m, coins); }, "PKE encryption", seconds);

	j_enc_derand(ctx, pk, m, coins);
	TIME_OPERATION_SECONDS({ j_dec(mm, sk, ctx); }, "PKE decryption", seconds);

	TIME_OPERATION_SECONDS(
		{
			j_enc_derand(ctx, pk, m, coins);
			j_dec(mm, sk, ctx);
		},
		"PKE encryption and decryption", seconds);
}

int main()
{
	int OK = true;

	OK = pke_test(SYSTEM_NAME, KEM_TEST_ITERATIONS);
	if (OK != true)
	{
		goto exit;
	}

	PRINT_TIMER_HEADER
	pke_bench(KEM_BENCH_SECONDS);

exit:
	return (OK == true) ? EXIT_SUCCESS : EXIT_FAILURE;
}
