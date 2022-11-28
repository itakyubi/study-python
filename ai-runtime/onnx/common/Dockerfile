FROM --platform=$TARGETPLATFORM ubuntu:18.04 as devel

# Install python
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends python3-pip python3-dev binutils && \
    cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    pip3 install --upgrade pip

# Install python packages
RUN pip install onnxruntime flask pillow requests numpy pyinstaller

# build exe
COPY runtime.py /home/
RUN pyinstaller -F /home/runtime.py

# package
FROM --platform=$TARGETPLATFORM ubuntu:18.04
COPY --from=devel /dist/runtime /home/baetyl/ai-runtime
COPY libonnxruntime_providers_shared.so /home/baetyl
ENV LD_LIBRARY_PATH /home/baetyl
WORKDIR /home/baetyl
ENTRYPOINT ["./ai-runtime"]


