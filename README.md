# An implementation of the SCloud+ KEM in *Jasmin*

This repository contains an implementation in *Jasmin* of the SCloud+ KEM (https://eprint.iacr.org/2024/1306). The code is based on the *C* implementation available from https://github.com/scloudplus/scloudplus.

## Some remarks on this implementation

### Differences wrt the *C* implementation

We diverge from the C implementation in a couple of implementation choices. Specifically, 
1. serialisation strategy of the ciphertext;
2. the sampling of ternary secrets ($\Phi$ and $\Psi$ algorithms).

### Jasmin compiler version

To compile the Jasmin code, a development version of the Jasmin compiler (`jasminc`) with limited support for floating-point instructions is required. For convenience, the required version is provided as a submodule (`submodules/jasmin`). Instructions for building the toolset are available at https://jasmin-lang.readthedocs.io/en/stable/misc/installation_guide.html#install-the-jasmin-tools.

### Keccak library

We rely on the Jasmin library of the SHA3 family of hash functions available from git@github.com:formosa-crypto/formosa-keccak.git. Currently, we are using the `AVX2` variant of the library.

### Build instructions

To build the library and run functional tests, please execute the following commands (assuming the Jasmin compiler was previously built):

```bash
$ cd src/scloud_jasmin_ref
$ make tests
```

## Repository structure

- `src`
    - `scloud_python` — executable specification in Python of the SCloud+ KEM
    - `scloud_jasmin_ref` — Jasmin implementation
    - `scloud_cref` — shared library build of the SCloud+ C implementation
- `submodules`
    - `jasmin` — development branch of the Jasmin compiler with floating-point support
    - `formosa-keccack` — third-party Jasmin implementation of SHA3
    - `scloudplus` — a fork of the the SCloud+ C implementation repository (https://github.com/scloudplus/scloudplus)
- `env` — support files (`Dockerfile` to build an environment with all required dependencies; etc.)

<!-- ------------------------------------------------------------------------->
## Building instructions:

### Build the assembly files

Run the following commands (requires `docker` installed on your machine; in case of doubt, run `docker run hello-world` to check that everything is working):

```
$ docker pull tfaoliveira/jc
$ docker run --name jasmincode -it tfaoliveira/jc bash

$ git clone https://github.com/haslab/JasminCode.git

$ cd JasminCode/
$ git submodule init
$ git submodule update

$ cd src/scloud_jasmin_ref/
$ JASMINC=jasminc make
$ cd ../../../
$ exit # optional as you may want to explore the contents of the repository
```

Inside the docker container (that is now in `Exited` status as you may observe by running `docker ps -a`), in directory `/workspace/JasminCode/src/scloud_jasmin_ref` there are 3 assembly files that correspond to the result of compiling the Jasmin implementations into assembly using the Jasmin compiler:
  - `jscloud128.s`
  - `jscloud192.s`
  - `jscloud256.s`

To copy them to the host machine, run, on the host machine, the following command:
```
$ docker cp jasmincode:/workspace/JasminCode/src/scloud_jasmin_ref/jscloud128.s .
$ docker cp jasmincode:/workspace/JasminCode/src/scloud_jasmin_ref/jscloud192.s .
$ docker cp jasmincode:/workspace/JasminCode/src/scloud_jasmin_ref/jscloud256.s .
```

### Run the tests

This step assumes that you have run `exit` in the previous section and that the `jasmincode` container is in `Exited` status.

To run the tests run:
```
$ docker start jasmincode 
$ docker exec -it jasmincode bash
$ cd JasminCode/src/scloud_cref
$ make
$ cd ../../../
$ cd JasminCode/src/scloud_python/
$ make
```

### Stop and remove the container

To stop and remove the container run the following commands:
```
$ docker stop jasmincode
$ docker rm jasmincode
```

To remove the docker image run the following command:
```
$ docker image rm tfaoliveira/jc
```
