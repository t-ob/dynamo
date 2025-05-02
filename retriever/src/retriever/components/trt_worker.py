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
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)

        if not self.engine:
            raise RuntimeError(
                f"Failed to load TensorRT engine from {self._config.model}"
            )

        # Create execution context
        self.context = self.engine.create_execution_context()

        logger.info("TensorRT engine loaded successfully")
        pass
        # self._init_engine()

        # if self._remote_prefill:
        #     runtime = dynamo_context["runtime"]
        #     comp_ns, comp_name = TensorRTLLMPrefillWorker.dynamo_address()  # type: ignore
        #     self._prefill_client = (
        #         await runtime.namespace(comp_ns)
        #         .component(comp_name)
        #         .endpoint("generate")
        #         .client()
        #     )
        #     while len(self._prefill_client.endpoint_ids()) < self._min_prefill_workers:
        #         logger.info(
        #             f"Waiting for prefill workers to be ready.\n"
        #             f" Current: {len(self._prefill_client.endpoint_ids())},"
        #             f" Required: {self._min_prefill_workers}"
        #         )
        #         await asyncio.sleep(30)

        # if self._kv_metrics_publisher is not None:
        #     task = asyncio.create_task(self.create_metrics_publisher_endpoint())
        #     task.add_done_callback(
        #         lambda _: logger.info("metrics publisher endpoint created")
        #     )

    # async def create_metrics_publisher_endpoint(self):
    #     component = dynamo_context["component"]
    #     await self._kv_metrics_publisher.create_endpoint(component)

    # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
    # If engine uses dynamic shapes, specify a profile to find the maximum input & output size.
    def allocate_buffers(self, profile_idx: int | None = None):
        input_shapes = []
        output_shapes = []
        input_dtypes = []
        output_dtypes = []

        inputs = []
        outputs = []
        bindings = []
        stream = cuda_call(cudart.cudaStreamCreate())
        tensor_names = [
            self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)
        ]
        print("hihi", tensor_names)

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

        for binding in input_tensor_names:
            # get_tensor_profile_shape returns (min_shape, optimal_shape, max_shape)
            # Pick out the max shape to allocate enough memory for the binding.
            shape = _shape(binding)
            trt_type = self.engine.get_tensor_dtype(binding)
            input_shapes.append(shape)
            input_dtypes.append(np.dtype(trt.nptype(trt_type)))
            # print("debug", binding, shape, trt_type, self.engine.get_tensor_profile_shape(binding, profile_idx))

            # # Allocate host and device buffers
            # if trt.nptype(trt_type):
            #     dtype = np.dtype(trt.nptype(trt_type))
            #     bindingMemory = HostDeviceMem(size, dtype)
            # else: # no numpy support: create a byte array instead (BF16, FP8, INT4)
            #     size = int(size * trt_type.itemsize)
            #     bindingMemory = HostDeviceMem(size)

            # # Append the device buffer to device bindings.
            # bindings.append(int(bindingMemory.device))

            # # Append to the appropriate list.
            # if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
            #     inputs.append(bindingMemory)
            # else:
            #     outputs.append(bindingMemory)

        for binding in output_tensor_names:
            shape = tuple(input_shapes[0][:-1] + (self._config.embedding_dim,))
            trt_type = self.engine.get_tensor_dtype(binding)
            output_shapes.append(shape)
            output_dtypes.append(np.dtype(trt.nptype(trt_type)))

        for binding, shape, dtype in zip(
            input_tensor_names, input_shapes, input_dtypes
        ):
            size = trt.volume(shape)
            bindingMemory = HostDeviceMem(size, dtype)
            inputs.append(bindingMemory)
            bindings.append(int(bindingMemory.device))

        for binding, shape, dtype in zip(
            output_tensor_names, output_shapes, output_dtypes
        ):
            size = trt.volume(shape)
            bindingMemory = HostDeviceMem(size, dtype)
            outputs.append(bindingMemory)
            bindings.append(int(bindingMemory.device))

        return inputs, outputs, input_shapes, output_shapes, bindings, stream

    @dynamo_endpoint()
    async def generate(self, request: TrtWorkerEmbeddingRequest):
        # Create input and output buffers
        input_ids = request.input_ids
        attention_mask = request.attention_mask
        token_type_ids = request.token_type_ids

        input_ids_data = np.zeros((len(input_ids), 512), dtype=np.int32)
        attention_mask_data = np.zeros((len(attention_mask), 512), dtype=np.int32)
        token_type_ids_data = np.zeros((len(token_type_ids), 512), dtype=np.int32)

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

        # Prepare input data from list[list[int]] to np.ndarray

        (
            inputs,
            outputs,
            input_shapes,
            output_shapes,
            bindings,
            stream,
        ) = self.allocate_buffers(profile_idx=1)

        inputs[0].host = input_ids_data
        inputs[1].host = attention_mask_data
        inputs[2].host = token_type_ids_data

        print(len(inputs), len(outputs), len(bindings), stream)

        context = self.engine.create_execution_context()
        context.set_optimization_profile_async(1, stream)

        context.set_input_shape(self.engine.get_tensor_name(0), input_ids_data.shape)
        context.set_input_shape(
            self.engine.get_tensor_name(1), attention_mask_data.shape
        )
        context.set_input_shape(
            self.engine.get_tensor_name(2), token_type_ids_data.shape
        )
        # self.context.set_binding_shape(3, (len(input_ids), 512))

        # self.context.set_output_shape(self.engine.get_tensor_name(3), (len(input_ids), 512))

        [output] = do_inference(context, self.engine, bindings, inputs, outputs, stream)
        [output_shape] = output_shapes

        output = output.reshape(output_shape)[: input_ids_data.shape[0], :]

        # Convert numpy array to list of lists for JSON serialization
        output_as_lists = output.tolist()
        print(
            "Converted output shape:",
            len(output_as_lists),
            "x",
            len(output_as_lists[0]) if output_as_lists else 0,
        )

        yield {"data": output_as_lists}
