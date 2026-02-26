COMMON_OPENSSL ?= 0

CFLAGS = -O3 -march=native -maes -g
LDLIBS = -lm -lcrypto

ifeq ($(COMMON_OPENSSL),0)
  CC ?= arch -x86_64 gcc
  CFLAGS  += -I$(HOME)/.local/include
  LDLIBS  += -L$(HOME)/.local/lib
endif

ifeq "$(CC)" "gcc"
	COMPILER = gcc
else ifeq "$(CC)" "clang"
	COMPILER = clang
endif

SOURCES := aes_ni.c encode.c random.c fips202.c sample.c pke.c matrix.c util.c kem.c
HEADERS := aes.h encode.h random.h fips202.h sample.h pke.h matrix.h util.h kem.h ds_benchmark.h config.h

SOURCES := $(addprefix ../../submodules/scloudplus/src/,$(SOURCES))
HEADERS := $(addprefix ../../submodules/scloudplus/src/,$(HEADERS))

ifeq ($(OS),Windows_NT)
	UNAME_P := $(PROCESSOR_ARCHITECTURE)
	RM = cmd /C del
	RMDIR = cmd /C rmdir /S /Q
	TARGET := $(TARGET).exe

else
	UNAME_P := $(shell uname -p)
	RM = rm -f
	RMDIR = rm -rf
endif

ifeq ($(UNAME_P),AMD64)
	CFLAGS += -DAMD
belse ifeq ($(UNAME_P),arm64)
	CFLAGS += -DARM
else ifeq ($(UNAME_P),x86_64)
	CFLAGS += -DAMD
else ifeq ($(UNAME_P),arm)
	CFLAGS += -DARM
endif

.PHONY: all scloudplus128lib.so scloudplus192lib.so scloudplus256lib.so clean

all: scloudplus

scloudplus:\
	scloudplus128lib.so\
	scloudplus192lib.so\
	scloudplus256lib.so\
	pke_tests\
	kem_tests

scloudplus128lib.so: $(SOURCES)
	$(CC) -shared $(CFLAGS) -Dscloudplus_l=128  $^ -o $@ $(LDLIBS)

scloudplus192lib.so: $(SOURCES)
	$(CC) -shared $(CFLAGS) -Dscloudplus_l=192  $^ -o $@ $(LDLIBS)

scloudplus256lib.so: $(SOURCES)
	$(CC) -shared $(CFLAGS) -Dscloudplus_l=256  $^ -o $@ $(LDLIBS)

pke_test: pke_test.c $(SOURCES)
	$(CC) $^ -o $@

kem_tests: kem_test_128 kem_test_192 kem_test_256

kem_test_128: kem_test.c $(SOURCES)
	$(CC) -D'scloudplus_l=128' $(CFLAGS) -I../../submodules/scloudplus/src $^ -o $@ $(LDLIBS)

kem_test_192: kem_test.c $(SOURCES)
	$(CC) -D'scloudplus_l=192' $(CFLAGS) -I../../submodules/scloudplus/src $^ -o $@ $(LDLIBS)

kem_test_256: kem_test.c $(SOURCES)
	$(CC) -D'scloudplus_l=256' $(CFLAGS) -I../../submodules/scloudplus/src $^ -o $@ $(LDLIBS)

pke_tests: pke_test_128 pke_test_192 pke_test_256

pke_test_128: pke_test.c $(SOURCES)
	$(CC) -D'scloudplus_l=128' $(CFLAGS) -I../../submodules/scloudplus/src $^ -o $@ $(LDLIBS)

pke_test_192: pke_test.c $(SOURCES)
	$(CC) -D'scloudplus_l=192' $(CFLAGS) -I../../submodules/scloudplus/src $^ -o $@ $(LDLIBS)

pke_test_256: pke_test.c $(SOURCES)
	$(CC) -D'scloudplus_l=256' $(CFLAGS) -I../../submodules/scloudplus/src $^ -o $@ $(LDLIBS)

clean:
	-$(RM) -rf *.so kem_test_??? pke_test_??? *.dSYM/

