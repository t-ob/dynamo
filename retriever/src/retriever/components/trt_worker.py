import logging

# from retriever.common.base_engine import BaseTensorrtLLMEngine
# from retriever.common.parser import parse_tensorrt_llm_args
from retriever.common.protocol import TrtWorkerEmbeddingRequest

from dynamo.sdk import async_on_start, dynamo_endpoint, service

# from retriever.common.utils import ServerType
# from retriever.components.prefill_worker import TensorRTLLMPrefillWorker


logger = logging.getLogger(__name__)


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
        # config = ServiceConfig.get_instance()
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

        logger.info("TensorRT-LLM Worker initialized")

    # async def create_metrics_publisher_endpoint(self):
    #     component = dynamo_context["component"]
    #     await self._kv_metrics_publisher.create_endpoint(component)

    @dynamo_endpoint()
    async def generate(self, request: TrtWorkerEmbeddingRequest):
        yield {"foo": "bar"}
