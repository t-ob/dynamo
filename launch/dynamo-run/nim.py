import asyncio
import ctypes

import numpy as np
import tensorrt as trt
import uvloop
from cuda import cuda, cudart

from dynamo.llm import ModelType, register_llm
from dynamo.runtime import DistributedRuntime, dynamo_worker


def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))


def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res


def _do_inference_base(inputs, outputs, stream, execute_async_func):
    # Transfer input data to the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
    [
        cuda_call(
            cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, kind, stream)
        )
        for inp in inputs
    ]
    # Run inference.
    execute_async_func()
    # Transfer predictions back from the GPU.
    kind = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
    [
        cuda_call(
            cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, kind, stream)
        )
        for out in outputs
    ]
    # Synchronize the stream
    cuda_call(cudart.cudaStreamSynchronize(stream))
    # Return only the host outputs.
    return [out.host for out in outputs]


# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, engine, bindings, inputs, outputs, stream):
    def execute_async_func():
        context.execute_async_v3(stream_handle=stream)

    # Setup context tensor address.
    num_io = engine.num_io_tensors
    for i in range(num_io):
        context.set_tensor_address(engine.get_tensor_name(i), bindings[i])
    return _do_inference_base(inputs, outputs, stream, execute_async_func)


class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""

    def __init__(self, size: int, dtype: np.dtype | None = None):
        dtype = dtype or np.dtype(np.uint8)
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))

        self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))
        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes

    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, data: np.ndarray | bytes):
        if isinstance(data, np.ndarray):
            if data.size > self.host.size:
                raise ValueError(
                    f"Tried to fit an array of size {data.size} into host memory of size {self.host.size}"
                )
            np.copyto(self.host[: data.size], data.flat, casting="safe")
        else:
            assert self.host.dtype == np.uint8
            self.host[: self.nbytes] = np.frombuffer(data, dtype=np.uint8)

    @property
    def device(self) -> int:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))


def _select_profile(
    profile_shapes: list[list[list[tuple[int, int, int]]]],
    input_idx: int,
    shape: tuple[int, int],
) -> int:
    for i, profile in enumerate(profile_shapes):
        min_shape, opt_shape, max_shape = profile[input_idx]
        if (
            min_shape[0] <= shape[0] <= max_shape[0]
            and min_shape[1] <= shape[1] <= max_shape[1]
        ):
            return i
    raise Exception("No profile found for shape", shape)


def _allocate_buffers(
    engine: trt.ICudaEngine,
    input_tensor_names: list[str],
    output_tensor_names: list[str],
    profile_idx: int | None = None,
):
    input_shapes: list[tuple[int, ...]] = []
    output_shapes: list[tuple[int, ...]] = []
    input_dtypes: list[np.dtype] = []
    output_dtypes: list[np.dtype] = []

    inputs: list[HostDeviceMem] = []
    outputs: list[HostDeviceMem] = []
    bindings: list[int] = []

    def _shape(tensor_name: str) -> tuple[int, ...]:
        shape = (
            engine.get_tensor_shape(tensor_name)
            if profile_idx is None
            else engine.get_tensor_profile_shape(tensor_name, profile_idx)[-1]
        )
        shape_valid = np.all([s >= 0 for s in shape])
        if not shape_valid and profile_idx is None:
            raise ValueError(
                f"Binding {tensor_name} has dynamic shape {shape}, "
                + "but no profile was specified."
            )
        return shape

    for tensor_name in input_tensor_names:
        shape = _shape(tensor_name)
        trt_type = engine.get_tensor_dtype(tensor_name)
        input_shapes.append(shape)
        input_dtypes.append(np.dtype(trt.nptype(trt_type)))

    for tensor_name in output_tensor_names:
        shape = tuple(input_shapes[0][:-1] + (2048,))  # TODO: hardcoded dimension
        # shape = _shape(tensor_name)
        trt_type = engine.get_tensor_dtype(tensor_name)
        output_shapes.append(shape)
        output_dtypes.append(np.dtype(trt.nptype(trt_type)))

    for shape, dtype in zip(input_shapes, input_dtypes):
        size = trt.volume(shape)
        bindingMemory = HostDeviceMem(size, dtype)
        inputs.append(bindingMemory)
        bindings.append(int(bindingMemory.device))

    for shape, dtype in zip(output_shapes, output_dtypes):
        size = trt.volume(shape)
        bindingMemory = HostDeviceMem(size, dtype)
        outputs.append(bindingMemory)
        bindings.append(int(bindingMemory.device))

    return inputs, outputs, input_shapes, output_shapes, bindings


def _init_engine():
    trt_logger = trt.Logger(trt.Logger.INFO)

    runtime = trt.Runtime(trt_logger)
    runtime.engine_host_code_allowed = True

    with open("/tmp/model.plan", "rb") as f:  # TODO: hardcoded path
        engine_bytes = f.read()

    engine = runtime.deserialize_cuda_engine(engine_bytes)

    input_tensor_names = []
    output_tensor_names = []
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            input_tensor_names.append(tensor_name)
        else:
            output_tensor_names.append(tensor_name)

    profile_shapes = []
    buffers = []
    for i in range(engine.num_optimization_profiles):
        shapes = []
        for input_name in input_tensor_names:
            shapes.append(engine.get_tensor_profile_shape(input_name, i))
        profile_shapes.append(shapes)
        buffers.append(
            _allocate_buffers(engine, input_tensor_names, output_tensor_names, i)
        )

    return (
        runtime,
        engine,
        input_tensor_names,
        output_tensor_names,
        profile_shapes,
        buffers,
    )


@dynamo_worker(static=False)
async def worker(runtime: DistributedRuntime):
    component = runtime.namespace("namespace").component("component")
    await component.create_service()
    model_path = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"  # or "/data/models/Qwen3-0.6B"
    model_type = ModelType.Embedding
    endpoint = component.endpoint("endpoint")
    # Optional last param to register_llm is model_name. If not present derives it from model_path
    await register_llm(model_type, endpoint, model_path)

    # Initialize your engine here
    (
        runtime,
        engine,
        input_tensor_names,
        output_tensor_names,
        profile_shapes,
        buffers,
    ) = _init_engine()

    # 3. Attach request handler
    #
    await endpoint.serve_endpoint(
        RequestHandler(
            runtime,
            engine,
            input_tensor_names,
            output_tensor_names,
            profile_shapes,
            buffers,
        ).generate
    )


class RequestHandler:
    def __init__(
        self,
        runtime: trt.Runtime,
        engine: trt.ICudaEngine,
        input_tensor_names: list[str],
        output_tensor_names: list[str],
        profile_shapes: list[list[list[tuple[int, int, int]]]],
        buffers: list[
            tuple[
                list[HostDeviceMem],
                list[HostDeviceMem],
                list[tuple[int, ...]],
                list[tuple[int, ...]],
                list[int],
            ]
        ],
    ):
        self.runtime = runtime
        self.engine = engine
        self.input_tensor_names = input_tensor_names
        self.output_tensor_names = output_tensor_names
        self.profile_shapes = profile_shapes
        self.buffers = buffers

    async def generate(self, request):
        # Call the engine
        # yield result dict
        token_ids = request["token_ids"]
        dimension = request.get("dimension", 2048)

        # Create padded tokens array and attention mask
        if not token_ids:
            raise ValueError("token_ids cannot be empty")

        # Find the maximum length among all sequences
        max_length = max(len(seq) for seq in token_ids)
        batch_size = len(token_ids)

        # Initialize arrays with padding value (0)
        tokens = np.zeros((batch_size, max_length), dtype=np.int32)
        attention_mask = np.zeros((batch_size, max_length), dtype=np.int32)
        dimension = np.full(batch_size, dimension, dtype=np.int64)

        # Fill the arrays
        for i, seq in enumerate(token_ids):
            seq_len = len(seq)
            tokens[i, :seq_len] = seq
            attention_mask[i, :seq_len] = 1

        profile_idx = _select_profile(self.profile_shapes, 0, (batch_size, max_length))
        inputs, outputs, input_shapes, output_shapes, bindings = self.buffers[
            profile_idx
        ]

        inputs[0].host = tokens
        inputs[1].host = attention_mask
        inputs[2].host = dimension

        context = self.engine.create_execution_context()
        stream = cuda_call(cudart.cudaStreamCreate())
        context.set_optimization_profile_async(profile_idx, stream)

        for input_tensor_name, input_tensor in zip(
            self.input_tensor_names, [tokens, attention_mask, dimension]
        ):
            context.set_input_shape(input_tensor_name, input_tensor.shape)

        [output] = do_inference(context, self.engine, bindings, inputs, outputs, stream)
        [output_shape] = output_shapes

        output = output.reshape(output_shape)[:batch_size, :]

        total_tokens = int(attention_mask.sum())
        out = {
            "embeddings": output.tolist(),
            "prompt_tokens": total_tokens,
            "total_tokens": total_tokens,
        }

        yield out


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
