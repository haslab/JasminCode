PWD = $(shell pwd)

OPENSSLDIR ?= $(HOME)/.local
JASMINCDIR ?= $(PWD)/../../../submodules/jasmin/compiler
SCLOUDDIR  ?= $(PWD)/../../../submodules/scloudplus/src

CC         ?= arch -x86_64 gcc
CFLAGS     ?= -g -O3 -maes -I$(OPENSSLDIR)/include -I $(SCLOUDDIR)
LFLAGS	   ?= -L $(OPENSSLDIR)/lib -lcrypto -lm
CLIBS      ?= -lcrypto -lm
LIBFLAGS   ?= -shared
SYSCALL    ?= $(JASMINCDIR)/syscall/jasmin_syscall.c

VALGRIND   ?= valgrind --error-exitcode=1 --max-stackframe=4096000

MEM_TESTS := mem_test_128 mem_test_192 mem_test_256
PKE_TESTS := pke_test_128 pke_test_192 pke_test_256
PKEMAT_TESTS := pke_test_mat_128 pke_test_mat_192 pke_test_mat_256

SCLOUDFILES := encode.c sample.c matrix.c random.c pke.c kem.c aes_ni.c fips202.c util.c
SCLOUDFILES := $(addprefix $(SCLOUDDIR)/, $(SCLOUDFILES))

.DEFAULT_GOAL := all

all: tests

run-mem-tests: $(MEM_TESTS)
	@for t in $(MEM_TESTS); do \
		echo "Running $$t..."; \
		$(VALGRIND) ./$$t; \
	done

run-pke-tests: $(PKE_TESTS)
	@for t in $(PKE_TESTS); do \
		echo "Running $$t..."; \
		./$$t; \
	done

run-pke-mat-tests: $(PKEMAT_TESTS)
	@for t in $(PKEMAT_TESTS); do \
		echo "Running $$t..."; \
		./$$t; \
	done

run-kem-tests: kem_test_128 kem_test_192 kem_test_256
	@for t in kem_test_128 kem_test_192 kem_test_256; do \
		echo "Running $$t..."; \
		./$$t; \
	done
	
tests: run-kem-tests

clean:
	rm -rf $(TESTS) *.dSYM *~

mem_test_128: mem_test.c ../jscloud128.s ${SCLOUDFILES} $(SYSCALL)
	$(CC) $(CFLAGS) -Dscloudplus_l=128 -I ../../submodules/scloudplus/src $^ -o $@ $(LFLAGS)

mem_test_192: mem_test.c ../jscloud192.s ${SCLOUDFILES} $(SYSCALL)
	$(CC) $(CFLAGS) -Dscloudplus_l=192 -I ../../submodules/scloudplus/src $^ -o $@ $(LFLAGS)

mem_test_256: mem_test.c ../jscloud256.s ${SCLOUDFILES} $(SYSCALL)
	$(CC) $(CFLAGS) -Dscloudplus_l=256 -I ../../submodules/scloudplus/src $^ -o $@ $(LFLAGS)

pke_tests_mat: pke_test_mat_128 pke_test_mat_192 pke_test_mat_256

pke_test_128: pke_test.c ../jscloud128.s $(SCLOUDFILES) $(SYSCALL)
	$(CC) $(CFLAGS) -Dscloudplus_l=128 -I ../../submodules/scloudplus/src $^ -o $@ $(LFLAGS)

pke_test_192: pke_test.c ../jscloud192.s ${SCLOUDFILES} $(SYSCALL)
	$(CC) $(CFLAGS) -Dscloudplus_l=192 -I ../../submodules/scloudplus/src $^ -o $@ $(LFLAGS)

pke_test_256: pke_test.c ../jscloud256.s ${SCLOUDFILES} $(SYSCALL)
	$(CC) $(CFLAGS) -Dscloudplus_l=256 -I ../../submodules/scloudplus/src $^ -o $@ $(LFLAGS)

pke_test_mat_128: pke_test_mat.c ../jscloud128.s ${SCLOUDFILES} $(SYSCALL)
	$(CC) $(CFLAGS) -Dscloudplus_l=128 -I ../../submodules/scloudplus/src $^ -o $@ $(LFLAGS)

pke_test_mat_192: pke_test_mat.c ../jscloud192.s ${SCLOUDFILES} $(SYSCALL)
	$(CC) $(CFLAGS) -Dscloudplus_l=192 -I ../../submodules/scloudplus/src $^ -o $@ $(LFLAGS)

pke_test_mat_256: pke_test_mat.c ../jscloud256.s ${SCLOUDFILES} $(SYSCALL)
	$(CC) $(CFLAGS) -Dscloudplus_l=256 -I ../../submodules/scloudplus/src $^ -o $@ $(LFLAGS)

kem_test_128: kem_test.c ../jscloud128.s ${SCLOUDFILES} $(SYSCALL)
	$(CC) $(CFLAGS) -Dscloudplus_l=128 -I ../../submodules/scloudplus/src $^ -o $@ $(LFLAGS)
kem_test_192: kem_test.c ../jscloud192.s ${SCLOUDFILES} $(SYSCALL)
	$(CC) $(CFLAGS) -Dscloudplus_l=192 -I ../../submodules/scloudplus/src $^ -o $@ $(LFLAGS)
kem_test_256: kem_test.c ../jscloud256.s ${SCLOUDFILES} $(SYSCALL)
	$(CC) $(CFLAGS) -Dscloudplus_l=256 -I ../../submodules/scloudplus/src $^ -o $@ $(LFLAGS)

.PHONY: print-makefile-vars
print-makefile-vars:
	@$(foreach v,$(.VARIABLES),\
	  $(if $(filter file,$(origin $v)),$(info $v=$($v))))
