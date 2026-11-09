# proteus — multi-stage build; final image is a small Debian runtime with the single binary.
FROM rust:1-bookworm AS build
WORKDIR /src
COPY . .
RUN cargo build --release --locked -p proteus-bio && scripts/licenses.sh /licenses

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/proteus /usr/local/bin/proteus
# Our licences and the notices of the data and fonts compiled into the binary.
COPY --from=build /licenses /usr/share/doc/proteus
ENV PROTEUS_DATA_DIR=/data
VOLUME ["/data"]
EXPOSE 8080
ENTRYPOINT ["proteus"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
