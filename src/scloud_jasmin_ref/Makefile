PWD = $(shell pwd)

OPENSSLDIR ?= $(HOME)/.local
JASMINCDIR ?= $(PWD)/../../submodules/jasmin/compiler
KECCAKDIR  ?= $(PWD)/../../submodules/formosa-keccak/src/amd64

JASMINC    ?= $(JASMINCDIR)/jasminc
JFLAGS     ?= -I Keccak=$(KECCAKDIR)

CC         ?= arch -x86_64 gcc
LIBFLAGS   ?= -shared $(JASMINCDIR)/syscall/jasmin_syscall.c


JLIBS := libjcomplex.so libjscloud128.so libjscloud192.so libjscloud256.so

JASMIN_FILES := aes.jinc double.jinc complex.jinc
JASMIN_P_FILES := bdd_BW.jinc scloudplus_PARAMS.jazz scloudplus_PARAMS.jinc scloudplus_keccak_PARAMS.jinc encode_PARAMS.jinc sample_PARAMS.jinc matrix_PARAMS.jinc pke_PARAMS.jinc kem_PARAMS.jinc

.DEFAULT_GOAL := all

all: $(JLIBS)

$(JLIBS): lib%.so: %.s
	$(CC) $(CFLAGS) $(LIBFLAGS) $^ -o $@

clean:
	rm -f $(JLIBS) *.s *~
	make -C tests clean

tests: $(JLIBS)
	make -C tests

jcomplex.s: complex.jazz double.jinc complex.jinc
	$(JASMINC) $(JFLAGS) -o $@ $<

jscloud128.s: scloudplus128.jazz scloudplus128_params.jinc $(JASMIN_FILES) $(JASMIN_P_FILES)
	$(JASMINC) $(JFLAGS) -o $@ $<

jscloud192.s: scloudplus192.jazz scloudplus192_params.jinc $(JASMIN_FILES) $(JASMIN_P_FILES)
	$(JASMINC) $(JFLAGS) -o $@ $<

jscloud256.s: scloudplus256.jazz scloudplus256_params.jinc $(JASMIN_FILES) $(JASMIN_P_FILES)
	$(JASMINC) $(JFLAGS) -o $@ $<

.PHONY: print-makefile-vars
print-makefile-vars:
	@$(foreach v,$(.VARIABLES),\
	  $(if $(filter file,$(origin $v)),$(info $v=$($v))))
