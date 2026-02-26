// jdouble.h
// API for Jasmin implementation of double (IEEE-754) handling.

#ifndef JASMIN_DOUBLE_H
#define JASMIN_DOUBLE_H

#include <stdint.h>

// OPS
extern double _double_add(double x, double y);
extern double _double_sub(double x, double y);
extern double _double_mul(double x, double y);
// ROUNDINGS/CONVERSIONs
extern double _double_round(double x);
extern double _double_floor(double x);
extern double _double_ceil(double x);
extern double _double_trunc(double x);
extern double _double_myround(double x);
extern uint64_t _double_trunc_u64(double x);
extern uint64_t _double_round_u64(double x);
extern double _double_from_u64(uint64_t x);
// COMPARISONS
extern double _double_eq(double x, double y);
extern double _double_lt(double x, double y);
extern double _double_le(double x, double y);
extern double _double_unord(double x, double y);
extern double _double_neq(double x, double y);
extern double _double_ge(double x, double y);
extern double _double_gt(double x, double y);
extern double _double_ord(double x, double y);
extern uint64_t _double_cmpmask(double d);

#endif // JASMIN_DOUBLE_H