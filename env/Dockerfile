FROM nixos/nix:latest AS builder

LABEL maintainer="Tiago"
WORKDIR /src

###########################################################
# easycrypt
# wip

###########################################################
# jasmin

COPY jasmin/ /src/jasmin
WORKDIR /src/jasmin/compiler

RUN nix-channel --update && \
    out=$(nix-build . --no-out-link) && \
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

###########################################################
# final assembly

FROM debian:trixie-slim

COPY --from=builder /rootfs /

RUN useradd -m -u 1001 jc
USER jc
WORKDIR /workspace
ENV PATH="/usr/local/bin:${PATH}"

