# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Literal

from pydantic import BaseModel
from retriever.common.chat_processor import ChatProcessorMixin
from retriever.common.protocol import (
    DynamoEmbedding,
    DynamoEmbeddingRequest,
    DynamoEmbeddingResponse,
    DynamoEmbeddingUsage,
)

# from retriever.common.protocol import DynamoTRTLLMChatCompletionRequest
from retriever.components.trt_worker import TrtWorkerEmbedding
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

from dynamo.runtime import Client, DistributedRuntime
from dynamo.sdk import async_on_start, depends, dynamo_context, dynamo_endpoint, service
from dynamo.sdk.lib.config import ServiceConfig

# from retriever.components.kv_router import Router
# from retriever.components.worker import TensorRTLLMWorker


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO"))


class ProcessorConfig(BaseModel):
    tokenizer: str
    model: str
    served_model_name: str
    router_mode: Literal["random", "round-robin"] = "random"
    min_workers: int = 1


@service(
    dynamo={
        "enabled": True,
        "namespace": "dynamo",
    },
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
)
class Processor(ChatProcessorMixin):
    worker = depends(TrtWorkerEmbedding)
    # router = depends(Router)

    def __init__(
        self,
    ):
        class_name = self.__class__.__name__
        config = ServiceConfig.get_instance()
        processor_config = ProcessorConfig.model_validate(config.get("Processor", {}))
        # config_args = config.as_args(class_name, prefix="")
        # args, engine_config = parse_tensorrt_llm_args(config_args)
        # self.remote_prefill = args.remote_prefill
        self.config = processor_config
        self.router_mode = processor_config.router_mode
        self.min_workers = processor_config.min_workers
        # self.args = args

        self._tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = (
            AutoTokenizer.from_pretrained(processor_config.tokenizer)
        )
        self._worker_client: Client | None = None
        # super().__init__(engine_config)

    @async_on_start
    async def async_init(self):
        runtime: DistributedRuntime = dynamo_context["runtime"]
        comp_ns, comp_name = TrtWorkerEmbedding.dynamo_address()  # type: ignore
        print("!!!!!!", comp_ns, comp_name, runtime, type(runtime))
        self._worker_client = (
            await runtime.namespace(comp_ns)
            .component(comp_name)
            .endpoint("generate")
            .client()
        )

        print(
            "!!!!!!",
            self._worker_client,
            type(self._worker_client),
            self._worker_client.endpoint_ids(),
        )

        # while len(self.worker_client.endpoint_ids()) < self.min_workers:
        #     logger.info(
        #         f"Waiting for workers to be ready.\n"
        #         f" Current: {len(self.worker_client.endpoint_ids())},"
        #         f" Required: {self.min_workers}"
        #     )
        #     await asyncio.sleep(30)

    async def _generate(self, raw_request: DynamoEmbeddingRequest):
        # raw_request.skip_special_tokens = False
        # raw_request.add_special_tokens = False
        # raw_request.spaces_between_special_tokens = False
        logger.debug(f"[preprocessor] Received request: {raw_request}")

        # Tokenize the input texts
        tokens = self._tokenizer(raw_request.input)

        # Calculate total number of tokens
        total_tokens = sum(len(ids) for ids in tokens.input_ids)
        logger.debug(f"Total tokens processed: {total_tokens}")

        if self.router_mode == "random":
            send_request = self._worker_client.random
        elif self.router_mode == "round-robin":
            send_request = self._worker_client.round_robin
        else:
            raise ValueError(f"Invalid router mode: {self.router_mode}")

        print(tokens)

        engine_generator = await send_request(tokens)

        async for raw_response in engine_generator:
            response = raw_response.data()
            embedding_data = [
                DynamoEmbedding(index=i, object="embedding", embedding=embedding[:4])
                for i, embedding in enumerate(response["data"])
            ]
            yield DynamoEmbeddingResponse(
                object="list",
                model=self.config.served_model_name,
                data=embedding_data,
                usage=DynamoEmbeddingUsage(
                    prompt_tokens=total_tokens, total_tokens=total_tokens
                ),
            )

        """
                async for raw_response in engine_generator:
            response = TRTLLMWorkerResponse.model_validate_json(raw_response.data())
            response.outputs = [TRTLLMWorkerResponseOutput(**response.outputs[0])]

            response_data = self.create_completion_stream_response(
                request,
                response,
            )
            logger.debug(f"[postprocessor] Response: {response_data}")
            yield response_data

        """

        """
                else:
            engine_generator = await self.worker_client.direct(
                preprocessed_request.model_dump_json(), int(worker_id)
            )

        if request_type == RequestType.CHAT:
            async for response in self.chat_processor.postprocess(
                engine_generator,
                raw_request,
                preprocessed_request.conversation,
            ):
                logger.debug(f"[preprocessor] Response: {response}")
                yield json.loads(response)
        else:
            async for response in self.completions_processor.postprocess(
                engine_generator, raw_request
            ):
                logger.debug(f"[preprocessor] Response: {response}")
                yield json.loads(response)
        """

        # if request_type == RequestType.CHAT:
        #     preprocessed_request = await self.chat_processor.preprocess(raw_request)
        # else:
        #     preprocessed_request = await self.completions_processor.preprocess(
        #         raw_request
        #     )

        # worker_id = ""
        # if self.router_mode == "kv":
        #     router_generator = await self.router_client.generate(
        #         preprocessed_request.tokens.model_dump_json()
        #     )
        #     decision = await router_generator.__anext__()
        #     decision = decision.data()
        #     worker_id, prefix_hit_rate = decision.split("_")
        #     prefix_hit_rate = float(prefix_hit_rate)
        #     logger.info(
        #         f"Worker ID: {worker_id} with estimated prefix hit rate: {prefix_hit_rate}"
        #     )

        # if worker_id == "":
        #     if self.router_mode == "round-robin":
        #         self._send_request = self.worker_client.round_robin
        #     else:
        #         # fallback to random
        #         self._send_request = self.worker_client.random

        #     engine_generator = await self._send_request(
        #         preprocessed_request.model_dump_json()
        #     )

        # else:
        #     engine_generator = await self.worker_client.direct(
        #         preprocessed_request.model_dump_json(), int(worker_id)
        #     )

    @dynamo_endpoint(name="embed")
    async def embed(self, raw_request: DynamoEmbeddingRequest):
        # max_tokens is deprecated, however if the max_tokens is provided instead
        # of max_completion_tokens, we will use the value as max_completion_tokens.
        async for response in self._generate(raw_request):
            yield response.model_dump()

    # @dynamo_endpoint()
    # async def completions(self, raw_request):
    #     async for response in self._generate(raw_request, RequestType.COMPLETION):
    #         yield response
