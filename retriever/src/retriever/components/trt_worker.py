import bisect
import ctypes
import logging

import numpy as np
import tensorrt as trt
from cuda import cuda, cudart
from pydantic import BaseModel

# from retriever.common.base_engine import BaseTensorrtLLMEngine
# from retriever.common.parser import parse_tensorrt_llm_args
from retriever.common.protocol import TrtWorkerEmbeddingRequest

from dynamo.sdk import async_on_start, dynamo_endpoint, service
from dynamo.sdk.lib.config import ServiceConfig

# from retriever.common.utils import ServerType
# from retriever.components.prefill_worker import TensorRTLLMPrefillWorker


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


logger = logging.getLogger(__name__)


class TrtWorkerEmbeddingConfig(BaseModel):
    model: str
    tokenizer: str
    tokenizer_max_seq_len: int
    embedding_dim: int


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"gpu": 1, "cpu": "10", "memory": "20Gi"},
    workers=1,
)
class TrtWorkerEmbedding:
    # prefill_worker = depends(TensorRTLLMPrefillWorker)

    def __init__(self):
        logger.info("Initializing TensorRT-LLM Worker")
        # class_name = self.__class__.__name__
        config = ServiceConfig.get_instance()
        trt_worker_config = TrtWorkerEmbeddingConfig.model_validate(
            config.get(self.__class__.__name__, {})
        )
        self._config = trt_worker_config
        self.engine: trt.ICudaEngine | None = None
        self.input_tensor_names: list[str] | None = None
        self.output_tensor_names: list[str] | None = None
        # config_args = config.as_args(class_name, prefix="")
        # args, engine_config = parse_tensorrt_llm_args(config_args)
        # worker_id = dynamo_context["endpoints"][0].lease_id()
        # self._min_prefill_workers = args.min_prefill_workers
        # super().__init__(
        #     namespace_str="dynamo",
        #     component_str=class_name,
        #     worker_id=worker_id,
        #     engine_config=engine_config,
        #     remote_prefill=args.remote_prefill,
        #     min_workers=args.min_workers,
        #     disagg_config_file=args.llmapi_disaggregated_config,
        #     block_size=args.block_size,
        #     router=args.router,
        #     server_type=ServerType.GEN,
        # )
        self.profile_shapes = []
        self.profiles = []

    @async_on_start
    async def async_init(self):
        logger.info(f"Loading TensorRT engine from {self._config.model}")

        # Create TensorRT logger
        trt_logger = trt.Logger(trt.Logger.INFO)

        # Create runtime and deserialize engine
        runtime = trt.Runtime(trt_logger)

        # Load engine from file
        with open(f"{self._config.model}/model.plan", "rb") as f:
            engine_bytes = f.read()

        # Deserialize the engine
        self.engine: trt.ICudaEngine = runtime.deserialize_cuda_engine(engine_bytes)
        print("engine type", type(self.engine))

        if not self.engine:
            raise RuntimeError(
                f"Failed to load TensorRT engine from {self._config.model}"
            )

        tensor_names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        input_tensor_names = [
            tensor_name
            for tensor_name in tensor_names
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT
        ]
        output_tensor_names = [
            tensor_name
            for tensor_name in tensor_names
            if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT
        ]

        self.input_tensor_names = input_tensor_names
        self.output_tensor_names = output_tensor_names

        # Create execution context
        # self.context = self.engine.create_execution_context()

        logger.info("TensorRT engine loaded successfully")
        logger.info(self.engine.num_optimization_profiles)

        for i in range(self.engine.num_optimization_profiles):
            profile_shapes = []
            for input_name in self.input_tensor_names:
                profile_shapes.append(
                    tuple(self.engine.get_tensor_profile_shape(input_name, i)[-1])
                )
            self.profile_shapes.append(profile_shapes)
            self.profiles.append(self.allocate_buffers(profile_idx=i))

        # etcd call to register model is live (possibly automatic)

    def allocate_buffers(self, profile_idx: int | None = None):
        input_shapes = []
        output_shapes = []
        input_dtypes = []
        output_dtypes = []

        inputs = []
        outputs = []
        bindings = []
        stream = cuda_call(cudart.cudaStreamCreate())

        def _shape(binding):
            shape = (
                self.engine.get_tensor_shape(binding)
                if profile_idx is None
                else self.engine.get_tensor_profile_shape(binding, profile_idx)[-1]
            )
            shape_valid = np.all([s >= 0 for s in shape])
            if not shape_valid and profile_idx is None:
                raise ValueError(
                    f"Binding {binding} has dynamic shape {shape}, "
                    + "but no profile was specified."
                )
            return shape

        for binding in self.input_tensor_names:
            shape = _shape(binding)
            trt_type = self.engine.get_tensor_dtype(binding)
            input_shapes.append(shape)
            input_dtypes.append(np.dtype(trt.nptype(trt_type)))

        for binding in self.output_tensor_names:
            shape = tuple(input_shapes[0][:-1] + (self._config.embedding_dim,))
            trt_type = self.engine.get_tensor_dtype(binding)
            output_shapes.append(shape)
            output_dtypes.append(np.dtype(trt.nptype(trt_type)))

        for binding, shape, dtype in zip(
            self.input_tensor_names, input_shapes, input_dtypes
        ):
            size = trt.volume(shape)
            bindingMemory = HostDeviceMem(size, dtype)
            inputs.append(bindingMemory)
            bindings.append(int(bindingMemory.device))

        for binding, shape, dtype in zip(
            self.output_tensor_names, output_shapes, output_dtypes
        ):
            size = trt.volume(shape)
            bindingMemory = HostDeviceMem(size, dtype)
            outputs.append(bindingMemory)
            bindings.append(int(bindingMemory.device))

        return inputs, outputs, input_shapes, output_shapes, bindings, stream

    @dynamo_endpoint()
    async def generate(self, request: TrtWorkerEmbeddingRequest):
        if self.engine is None:
            raise RuntimeError("Engine not initialized")

        # Create input and output buffers
        input_ids = request.input_ids
        attention_mask = request.attention_mask
        token_type_ids = request.token_type_ids

        input_ids_data = np.zeros(
            (len(input_ids), self._config.tokenizer_max_seq_len), dtype=np.int32
        )
        attention_mask_data = np.zeros(
            (len(attention_mask), self._config.tokenizer_max_seq_len), dtype=np.int32
        )
        token_type_ids_data = np.zeros(
            (len(token_type_ids), self._config.tokenizer_max_seq_len), dtype=np.int32
        )

        for idx, (input_id, attention_mask, token_type_id) in enumerate(
            zip(input_ids, attention_mask, token_type_ids)
        ):
            input_ids_data[idx, : len(input_id)] = np.array(input_id, dtype=np.int32)
            attention_mask_data[idx, : len(attention_mask)] = np.array(
                attention_mask, dtype=np.int32
            )
            token_type_ids_data[idx, : len(token_type_id)] = np.array(
                token_type_id, dtype=np.int32
            )

        profile_idx = bisect.bisect_left(
            self.profile_shapes, input_ids_data.shape, key=lambda x: x[0]
        )

        (
            inputs,
            outputs,
            input_shapes,
            output_shapes,
            bindings,
            stream,
        ) = self.profiles[
            profile_idx
        ]  # self.allocate_buffers(profile_idx=1)

        inputs[0].host = input_ids_data
        inputs[1].host = attention_mask_data
        inputs[2].host = token_type_ids_data

        context = self.engine.create_execution_context()
        context.set_optimization_profile_async(profile_idx, stream)

        context.set_input_shape(self.engine.get_tensor_name(0), input_ids_data.shape)
        context.set_input_shape(
            self.engine.get_tensor_name(1), attention_mask_data.shape
        )
        context.set_input_shape(
            self.engine.get_tensor_name(2), token_type_ids_data.shape
        )

        [output] = do_inference(context, self.engine, bindings, inputs, outputs, stream)
        [output_shape] = output_shapes

        output = output.reshape(output_shape)[: input_ids_data.shape[0], :]

        # Convert numpy array to list of lists for JSON serialization
        output_as_lists = output.tolist()

        yield {"data": output_as_lists}
