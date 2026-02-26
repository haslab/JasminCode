#include "ds_benchmark.h"
#include "random.h"
#include "param.h"
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#define KEM_TEST_ITERATIONS 1
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
extern void j_decompressc1(uint16_t *);
extern void j_decompressc2(uint16_t *);
extern void j_msgdecode(uint8_t *, uint16_t *);
extern void j_C2xC1xS(uint16_t *, uint16_t *, uint16_t *, uint16_t *);

extern void j_keygen_plain(uint16_t *, uint8_t *, uint16_t *, uint8_t *);
extern void j_enc_derand_plain(uint16_t *, uint16_t *, uint16_t *, uint8_t *, uint16_t *, uint8_t *);
extern void j_dec_plain(uint8_t *, uint16_t *, uint16_t *, uint16_t *);

extern void j_keygen(uint8_t *, uint8_t *, uint8_t *);
extern void j_enc_derand(uint8_t *, uint8_t *, uint8_t *, uint8_t *);
extern void j_dec(uint8_t *, uint8_t *, uint8_t *);

extern void j_kem_keygen(uint8_t *, uint8_t *);
extern void j_kem_encaps(uint8_t *, uint8_t *, uint8_t *);
extern void j_kem_decaps(uint8_t *, uint8_t *, uint8_t *);

int main()
{
  uint8_t *seedA = malloc(sizeof(uint8_t) * 16);
  uint8_t *r1 = malloc(sizeof(uint8_t) * 32);
  uint8_t *r2 = malloc(sizeof(uint8_t) * 32);

  uint8_t *hbuf = malloc(sizeof(uint8_t) * 80);

  uint16_t *S1 = malloc(sizeof(uint16_t) * scloudplus_nbar * scloudplus_n); //11*1120
  uint16_t *A1 = malloc(sizeof(uint16_t) * scloudplus_m * scloudplus_n); //11*1120
  uint16_t *E1 = malloc(sizeof(uint16_t) * scloudplus_m * scloudplus_nbar); //1136*11
  uint16_t *B1 = malloc(sizeof(uint16_t) * scloudplus_m * scloudplus_nbar); //1136*11
  uint8_t  *pk_= malloc(sizeof(uint8_t) * (scloudplus_pk-16));
  uint8_t  *sk = malloc(sizeof(uint8_t) * scloudplus_pke_sk);

  uint8_t  *ss2= malloc(sizeof(uint8_t) * scloudplus_ss);
  uint16_t *M2 = malloc(sizeof(uint16_t) * scloudplus_mbar * scloudplus_nbar);
  uint16_t *S2 = malloc(sizeof(uint16_t) * scloudplus_mbar * scloudplus_m); //12*1136
  uint16_t *E21= malloc(sizeof(uint16_t) * scloudplus_mbar * scloudplus_n);
  uint16_t *E22= malloc(sizeof(uint16_t) * scloudplus_mbar * scloudplus_nbar);
  uint16_t *C21= malloc(sizeof(uint16_t) * scloudplus_mbar * scloudplus_n);
  uint16_t *C22= malloc(sizeof(uint16_t) * scloudplus_mbar * scloudplus_nbar);
  uint8_t  *c1 = malloc(sizeof(uint8_t) * scloudplus_c1);
  uint8_t  *c2 = malloc(sizeof(uint8_t) * scloudplus_c2);
  uint8_t  *ctx = malloc(sizeof(uint8_t) * (scloudplus_c1 + scloudplus_c2));
  uint8_t  *pk = malloc(sizeof(uint8_t) * (scloudplus_pk));
  uint8_t  *kem_sk = malloc(sizeof(uint8_t) * (scloudplus_kem_sk));


  randombytes(seedA, 16);
  randombytes(r1, 32);
  randombytes(r2, 32);
  randombytes(ss2, scloudplus_ss);

  j_F(hbuf, r1);
  j_genMat(A1, seedA);
  j_samplepsi(S1, r1);
  j_sampleeta1(E1, r2);
  j_AxStxE(B1, A1, S1, E1);
  j_packpk(pk_, B1);
  j_packsk(sk, S1);

  j_unpackpk(B1, pk_);
  j_msgencode(M2, ss2);
  j_samplephi(S2, r1);
  j_sampleeta2(E21, E22, r2);
  j_SxAxE(C21, S2, A1, E21);
  j_compressc1(C21);
  j_compressc2(C22);
  j_SxBxExM(C22, S2, B1, E22, M2);

  j_packc1(c1, C21);
  j_packc2(c2, C22);

  j_unpacksk(S1, sk);
  j_unpackc1(C21, c1);
  j_unpackc2(C22, c2);
  j_decompressc1(C21);
  j_decompressc2(C22);
  j_msgdecode(ss2, M2);
  j_C2xC1xS(M2, C21, C21, S1);
  j_dec_plain(ss2, S1, C21, C22);

  j_keygen_plain(B1, seedA, S1, r1);
  j_enc_derand_plain(C21, C22, B1, seedA, M2, r2);
  j_dec_plain(ss2, S1, C21, C22);

  j_keygen(pk, sk, r1);
  j_enc_derand(ctx, pk, ss2, r2);
  j_dec(ss2, sk, ctx);

  j_kem_keygen(pk, kem_sk);
  j_kem_encaps(ctx, ss2, pk);
  j_kem_decaps(ss2, kem_sk, ctx);

  free(seedA);
  free(r1);
  free(r2);
  free(hbuf);
  free(S1);
  free(A1);
  free(E1);
  free(B1);
  free(pk_);
  free(pk);
  free(sk);
  free(kem_sk);
  free(ss2);
  free(M2);
  free(S2);
  free(E21);
  free(E22);
  free(C21);
  free(C22);
  free(c1);
  free(c2);

  return 0;
}

