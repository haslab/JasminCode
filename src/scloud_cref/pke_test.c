#include "ds_benchmark.h"
#include "encode.h"
#include "pke.h"
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
      scloudplus_pkekeygen(pk, sk);
      scloudplus_pkeenc(pk, m, coins, ctx);
      scloudplus_pkedec(sk, ctx, mm);
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

  TIME_OPERATION_SECONDS({ scloudplus_pkekeygen(pk, sk); }, "Key generation", seconds);

  scloudplus_pkekeygen(pk, sk);
  TIME_OPERATION_SECONDS({ scloudplus_pkeenc(pk, m, coins, ctx); }, "PKE encryptio", seconds);
  
  scloudplus_pkeenc(pk, m, coins, ctx);
  TIME_OPERATION_SECONDS({ scloudplus_pkedec(sk, ctx, mm); }, "PKE decryption", seconds);
  
  TIME_OPERATION_SECONDS(
			 {
			   scloudplus_pkeenc(pk, m, coins, ctx);
			   scloudplus_pkedec(sk, ctx, mm);
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
