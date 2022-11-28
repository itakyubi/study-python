FROM ubuntu:18.04
COPY libonnxruntime_providers_shared.so /home/baetyl/lib/
COPY dist /home/baetyl
ENV LD_LIBRARY_PATH /home/baetyl/lib
#ENTRYPOINT["./onnx_runtime"]