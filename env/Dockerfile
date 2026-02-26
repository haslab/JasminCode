FROM nixos/nix:2.24.15 AS builder1

LABEL maintainer="Tiago"

###########################################################
# jasmin

WORKDIR /workspace
COPY submodules/jasmin/ /workspace/jasmin
WORKDIR /workspace/jasmin/
ENV NIX_PATH="nixpkgs=channel:nixpkgs-unstable"
RUN nix-shell --arg inCI true --arg coqDeps true --run 'make -C compiler CIL'

WORKDIR /workspace/jasmin/compiler
RUN out=$(nix-build . --no-out-link) && \
    mkdir -p /rootfs/usr/local/bin/ && \
    bins="jasminc jasmin-ct" && \
    deps="" && \
    for b in $bins; do \
        deps="$deps $(nix-store -qR "$out/bin/$b")"; \
    done && \
    unique_deps=$(printf "%s\n" $deps | sort -u) && \
    cp -Rv --parents $unique_deps /rootfs && \
    for b in $bins; do \
        cp -v "$out/bin/$b" /rootfs/usr/local/bin/; \
    done

############################################################

FROM debian:trixie-slim AS builder2

RUN apt update \
 && apt install -y --no-install-recommends curl ca-certificates procps build-essential perl perl-modules \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1001 jc
USER jc
WORKDIR /workspace

RUN set -eux; \
    curl -L https://github.com/openssl/openssl/releases/download/openssl-3.6.0/openssl-3.6.0.tar.gz | tar xz; \
    cd openssl-3.6.0; \
    ./Configure linux-x86_64 no-shared -fPIC --prefix=/home/jc/.local \
      --openssldir=/home/jc/.local/ssl; \
    make -j"$(nproc)"; \
    make install_sw; \
    cd /workspace; \
    rm -rf openssl-3.6.0

############################################################
FROM debian:trixie-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl ca-certificates bzip2 build-essential libssl-dev valgrind git \
 && rm -rf /var/lib/apt/lists/* \
 && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba \
 && chmod +x /usr/local/bin/micromamba

RUN useradd -m -u 1001 jc

COPY --from=builder1 /rootfs /

RUN mkdir -p /workspace /workspace/wheels \
 && chown -R jc:jc /workspace

USER jc
WORKDIR /workspace
SHELL ["/bin/bash", "-c"]

COPY --from=builder2 --chown=jc:jc /home/jc/.local /home/jc/.local

COPY --chown=jc:jc env/requirements-pinned.txt /workspace/requirements-pinned.txt

RUN micromamba create -n jcenv python \
 && micromamba clean --all --yes \
 \
 && OPENSSL_DIR="/home/jc/.local" \
    micromamba run -n jcenv pip wheel \
      --no-cache-dir \
      -r /workspace/requirements-pinned.txt \
      -w /workspace/wheels \
 \
 && micromamba run -n jcenv pip install \
      --no-index --find-links=/workspace/wheels \
      --upgrade --force-reinstall \
      -r /workspace/requirements-pinned.txt \
 \
 && micromamba run -n jcenv python -c "import sys; print(sys.version)" \
 && micromamba run -n jcenv pip list \
 && rm -fr /workspace/*

RUN micromamba shell init --shell bash --root-prefix=~/.local/share/mamba

RUN cat >> /home/jc/.bashrc <<'EOF'
export MAMBA_ROOT_PREFIX="$HOME/.local/share/mamba"
eval "$(micromamba shell hook --shell bash)"
micromamba activate jcenv >/dev/null 2>&1 || true
EOF

CMD ["bash"]

