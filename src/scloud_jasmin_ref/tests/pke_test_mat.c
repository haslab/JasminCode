#include "ds_benchmark.h"
#include "encode.h"
#include "sample.h"
#include "matrix.h"
#include "random.h"
#include "param.h"
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#define KEM_TEST_ITERATIONS 100
#define KEM_BENCH_SECONDS 1
#if (scloudplus_l == 128)
#define SYSTEM_NAME "scloud plus 128 (mat)"
#elif (scloudplus_l == 192)
#define SYSTEM_NAME "scloud plus 192 (mat)"
#elif (scloudplus_l == 256)
#define SYSTEM_NAME "scloud plus 256 (mat)"
#endif



extern void j_F(uint8_t *, uint8_t *);
extern void j_samplepsi(uint16_t *, uint8_t *);
extern void j_sampleeta1(uint16_t *, uint8_t *);
extern void j_packpk(uint8_t *, uint16_t *);
extern void j_packsk(uint8_t *, uint16_t *);
extern void j_unpackpk(uint16_t *, uint8_t *);
extern void j_unpacksk(uint16_t *, uint8_t *);
extern void j_samplephi(uint16_t *, uint8_t *);
extern void j_sampleeta2(uint16_t *, uint16_t *, uint8_t *);
extern void j_msgencode(uint16_t *, uint8_t *);
extern void j_msgdecode(uint8_t *, uint16_t *);
extern void j_compressc1(uint16_t *);
extern void j_decompressc1(uint16_t *);
extern void j_compressc2(uint16_t *);
extern void j_decompressc2(uint16_t *);
extern void j_packc1(uint8_t *, uint16_t *);
extern void j_unpackc1(uint16_t *, uint8_t *);
extern void j_packc2(uint8_t *, uint16_t *);
extern void j_unpackc2(uint16_t *, uint8_t *);


void j_pkekeygen(uint8_t *pk, uint8_t *sk)
{
	uint16_t *S =
		(uint16_t *)malloc(sizeof(uint16_t) * scloudplus_n * scloudplus_nbar);
	uint16_t *E =
		(uint16_t *)malloc(sizeof(uint16_t) * scloudplus_m * scloudplus_nbar);
	uint16_t *B =
		(uint16_t *)malloc(sizeof(uint16_t) * scloudplus_m * scloudplus_nbar);
	uint8_t alpha[32], seed[80];
	uint8_t *seedA = seed;
	uint8_t *r1 = seed + 16;
	uint8_t *r2 = seed + 48;
	randombytes(alpha, 32);
	j_F(seed, alpha);
	j_samplepsi(S, r1);
	j_sampleeta1(E, r2);
	scloudplus_mul_add_as_e(seedA, S, E, B);
	j_packpk(pk, B);
	memcpy(pk + scloudplus_pk - 16, seedA, 16);
	j_packsk(sk, S);
	free(S);
	free(E);
	free(B);
}

void j_pkeenc(uint8_t *pk, uint8_t *m, uint8_t *r, uint8_t *ctx)
{
	uint16_t *S1 =
		(uint16_t *)malloc(sizeof(uint16_t) * scloudplus_mbar * scloudplus_m);
	uint16_t *E1 =
		(uint16_t *)malloc(sizeof(uint16_t) * scloudplus_mbar * scloudplus_n);
	uint16_t *E2 =
		(uint16_t *)malloc(sizeof(uint16_t) * scloudplus_mbar * scloudplus_nbar);
	uint16_t *mu0 =
		(uint16_t *)malloc(sizeof(uint16_t) * scloudplus_mbar * scloudplus_nbar);
	uint16_t *C1 =
		(uint16_t *)malloc(sizeof(uint16_t) * scloudplus_mbar * scloudplus_n);
	uint16_t *C2 =
		(uint16_t *)malloc(sizeof(uint16_t) * scloudplus_mbar * scloudplus_nbar);
	uint16_t *B =
		(uint16_t *)malloc(sizeof(uint16_t) * scloudplus_m * scloudplus_nbar);
	uint8_t seed[80];
	uint8_t *seedA = pk + scloudplus_pk - 16;
	uint8_t *r1 = seed;
	uint8_t *r2 = seed + 32;
	j_F(seed, r);
	j_samplephi(S1, r1);
	j_sampleeta2(E1, E2, r2);
	j_msgencode(mu0, m);
	j_unpackpk(B, pk);
	scloudplus_mul_add_sa_e(seedA, S1, E1, C1);
	scloudplus_mul_add_sb_e(S1, B, E2, C2);
	scloudplus_add(C2, mu0, scloudplus_mbar * scloudplus_nbar, C2);
	j_compressc1(C1);
	j_compressc2(C2);
	j_packc1(ctx, C1);
	j_packc2(ctx + scloudplus_c1, C2);
	free(S1);
	free(E1);
	free(E2);
	free(mu0);
	free(C1);
	free(C2);
	free(B);
}

void j_pkedec(uint8_t *sk, uint8_t *ctx, uint8_t *m)
{
	uint16_t *S =
		(uint16_t *)malloc(sizeof(uint16_t) * scloudplus_n * scloudplus_nbar);
	uint16_t *C1 =
		(uint16_t *)malloc(sizeof(uint16_t) * scloudplus_mbar * scloudplus_n);
	uint16_t *C2 =
		(uint16_t *)malloc(sizeof(uint16_t) * scloudplus_mbar * scloudplus_nbar);
	uint16_t *D =
		(uint16_t *)malloc(sizeof(uint16_t) * scloudplus_mbar * scloudplus_nbar);
	j_unpacksk(S, sk);
	j_unpackc1(C1, ctx);
	j_unpackc2(C2, ctx + scloudplus_c1);
	j_decompressc1(C1);
	j_decompressc2(C2);
	scloudplus_mul_cs(C1, S, D);
	scloudplus_sub(C2, D, scloudplus_mbar * scloudplus_nbar, D);
	j_msgdecode(m, D);
	free(S);
	free(C1);
	free(C2);
	free(D);
}

static int pke_test(const char *named_parameters, int iterations)
{
  uint8_t pk[scloudplus_pk];
  uint8_t sk[scloudplus_pke_sk];
  uint8_t ctx[scloudplus_ctx];
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
      randombytes(coins, 32);
      randombytes(m, scloudplus_ss);
      j_pkekeygen(pk, sk);
      j_pkeenc(pk, m, coins, ctx);
      j_pkedec(sk, ctx, mm);
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
  uint8_t ctx[scloudplus_ctx];
  uint8_t coins[32];
  uint8_t m[scloudplus_ss];
  uint8_t mm[scloudplus_ss];

  randombytes(coins, 32);

  TIME_OPERATION_SECONDS({ j_pkekeygen(pk, sk); }, "Key (mat) generation", seconds);

  j_pkekeygen(pk, sk);
  TIME_OPERATION_SECONDS({ j_pkeenc(pk, m, coins, ctx); }, "PKE (mat) encryption", seconds);
  
  j_pkeenc(pk, m, coins, ctx);
  TIME_OPERATION_SECONDS({ j_pkedec(sk, ctx, mm); }, "PKE (mat) decryption", seconds);
  
  TIME_OPERATION_SECONDS(
			 {
			   j_pkeenc(pk, m, coins, ctx);
			   j_pkedec(sk, ctx, mm);
			 },
			 "PKE (mat) encryption and decryption", seconds);
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
