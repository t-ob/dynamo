// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use anyhow::Error;
use async_openai::config::OpenAIConfig;
use async_stream::stream;
use dynamo_llm::http::{
    client::{
        GenericBYOTClient, HttpClientConfig, HttpRequestContext, NvCustomClient, PureOpenAIClient,
    },
    service::{
        error::HttpError,
        metrics::{Endpoint, RequestType, Status},
        service_v2::HttpService,
        Metrics,
    },
};
use dynamo_llm::protocols::{
    openai::{
        chat_completions::{NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse},
        completions::{NvCreateCompletionRequest, NvCreateCompletionResponse},
    },
    Annotated,
};
use dynamo_runtime::{
    engine::AsyncEngineContext,
    pipeline::{
        async_trait, AsyncEngine, AsyncEngineContextProvider, ManyOut, ResponseStream, SingleIn,
    },
    CancellationToken,
};
use futures::StreamExt;
use prometheus::{proto::MetricType, Registry};
use reqwest::StatusCode;
use rstest::*;
use std::sync::Arc;

struct CounterEngine {}

#[allow(deprecated)]
#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for CounterEngine
{
    async fn generate(
        &self,
        request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        let (request, context) = request.transfer(());
        let ctx = context.context();

        // ALLOW: max_tokens is deprecated in favor of completion_usage_tokens
        let max_tokens = request.inner.max_tokens.unwrap_or(0) as u64;

        // let generator = NvCreateChatCompletionStreamResponse::generator(request.model.clone());
        let generator = request.response_generator();

        let stream = stream! {
            tokio::time::sleep(std::time::Duration::from_millis(max_tokens)).await;
            for i in 0..10 {
                let inner = generator.create_choice(i,Some(format!("choice {i}")), None, None);

                let output = NvCreateChatCompletionStreamResponse {
                    inner,
                };

                yield Annotated::from_data(output);
            }
        };

        Ok(ResponseStream::new(Box::pin(stream), ctx))
    }
}

struct AlwaysFailEngine {}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateChatCompletionRequest>,
        ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>,
        Error,
    > for AlwaysFailEngine
{
    async fn generate(
        &self,
        _request: SingleIn<NvCreateChatCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>, Error> {
        Err(HttpError {
            code: 403,
            message: "Always fail".to_string(),
        })?
    }
}

#[async_trait]
impl
    AsyncEngine<
        SingleIn<NvCreateCompletionRequest>,
        ManyOut<Annotated<NvCreateCompletionResponse>>,
        Error,
    > for AlwaysFailEngine
{
    async fn generate(
        &self,
        _request: SingleIn<NvCreateCompletionRequest>,
    ) -> Result<ManyOut<Annotated<NvCreateCompletionResponse>>, Error> {
        Err(HttpError {
            code: 401,
            message: "Always fail".to_string(),
        })?
    }
}

fn compare_counter(
    metrics: &Metrics,
    model: &str,
    endpoint: &Endpoint,
    request_type: &RequestType,
    status: &Status,
    expected: u64,
) {
    assert_eq!(
        metrics.get_request_counter(model, endpoint, request_type, status),
        expected,
        "model: {}, endpoint: {:?}, request_type: {:?}, status: {:?}",
        model,
        endpoint.as_str(),
        request_type.as_str(),
        status.as_str()
    );
}

fn compute_index(endpoint: &Endpoint, request_type: &RequestType, status: &Status) -> usize {
    let endpoint = match endpoint {
        Endpoint::Completions => 0,
        Endpoint::ChatCompletions => 1,
        Endpoint::Embeddings => todo!(),
        Endpoint::Responses => todo!(),
    };

    let request_type = match request_type {
        RequestType::Unary => 0,
        RequestType::Stream => 1,
    };

    let status = match status {
        Status::Success => 0,
        Status::Error => 1,
    };

    endpoint * 4 + request_type * 2 + status
}

fn compare_counters(metrics: &Metrics, model: &str, expected: &[u64; 8]) {
    for endpoint in &[Endpoint::Completions, Endpoint::ChatCompletions] {
        for request_type in &[RequestType::Unary, RequestType::Stream] {
            for status in &[Status::Success, Status::Error] {
                let index = compute_index(endpoint, request_type, status);
                compare_counter(
                    metrics,
                    model,
                    endpoint,
                    request_type,
                    status,
                    expected[index],
                );
            }
        }
    }
}

fn inc_counter(
    endpoint: Endpoint,
    request_type: RequestType,
    status: Status,
    expected: &mut [u64; 8],
) {
    let index = compute_index(&endpoint, &request_type, &status);
    expected[index] += 1;
}

#[allow(deprecated)]
#[tokio::test]
async fn test_http_service() {
    let service = HttpService::builder().port(8989).build().unwrap();
    let state = service.state_clone();
    let manager = state.manager();

    let token = CancellationToken::new();
    let cancel_token = token.clone();
    let task = tokio::spawn(async move { service.run(token.clone()).await });

    let registry = Registry::new();

    let counter = Arc::new(CounterEngine {});
    let result = manager.add_chat_completions_model("foo", counter);
    assert!(result.is_ok());

    let failure = Arc::new(AlwaysFailEngine {});
    let result = manager.add_chat_completions_model("bar", failure.clone());
    assert!(result.is_ok());

    let result = manager.add_completions_model("bar", failure);
    assert!(result.is_ok());

    let metrics = state.metrics_clone();
    metrics.register(&registry).unwrap();

    let mut foo_counters = [0u64; 8];
    let mut bar_counters = [0u64; 8];

    compare_counters(&metrics, "foo", &foo_counters);
    compare_counters(&metrics, "bar", &bar_counters);

    let client = reqwest::Client::new();

    let message = async_openai::types::ChatCompletionRequestMessage::User(
        async_openai::types::ChatCompletionRequestUserMessage {
            content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                "hi".to_string(),
            ),
            name: None,
        },
    );

    let mut request = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("foo")
        .messages(vec![message])
        .build()
        .expect("Failed to build request");

    // let mut request = ChatCompletionRequest::builder()
    //     .model("foo")
    //     .add_user_message("hi")
    //     .build()
    //     .unwrap();

    // ==== ChatCompletions / Stream / Success ====
    request.stream = Some(true);

    // ALLOW: max_tokens is deprecated in favor of completion_usage_tokens
    request.max_tokens = Some(3000);

    let response = client
        .post("http://localhost:8989/v1/chat/completions")
        .json(&request)
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success(), "{:?}", response);

    tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
    assert_eq!(metrics.get_inflight_count("foo"), 1);

    // process byte stream
    let _ = response.bytes().await.unwrap();

    inc_counter(
        Endpoint::ChatCompletions,
        RequestType::Stream,
        Status::Success,
        &mut foo_counters,
    );
    compare_counters(&metrics, "foo", &foo_counters);
    compare_counters(&metrics, "bar", &bar_counters);

    // check registry and look or the request duration histogram
    let families = registry.gather();
    let histogram_metric_family = families
        .into_iter()
        .find(|m| m.get_name() == "nv_llm_http_service_request_duration_seconds")
        .expect("Histogram metric not found");

    assert_eq!(
        histogram_metric_family.get_field_type(),
        MetricType::HISTOGRAM
    );

    let histogram_metric = histogram_metric_family.get_metric();

    assert_eq!(histogram_metric.len(), 1); // We have one metric with label model

    let metric = &histogram_metric[0];
    let histogram = metric.get_histogram();

    let buckets = histogram.get_bucket();

    let mut found = false;

    for bucket in buckets {
        let upper_bound = bucket.get_upper_bound();
        let cumulative_count = bucket.get_cumulative_count();

        println!(
            "Bucket upper bound: {}, count: {}",
            upper_bound, cumulative_count
        );

        // Since our observation is 2.5, it should fall into the bucket with upper bound 4.0
        if upper_bound >= 4.0 {
            assert_eq!(
                cumulative_count, 1,
                "Observation should be counted in the 4.0 bucket"
            );
            found = true;
        } else {
            assert_eq!(
                cumulative_count, 0,
                "No observations should be in this bucket"
            );
        }
    }

    assert!(found, "The expected bucket was not found");
    // ==== ChatCompletions / Stream / Success ====

    // ==== ChatCompletions / Unary / Success ====
    request.stream = Some(false);

    // ALLOW: max_tokens is deprecated in favor of completion_usage_tokens
    request.max_tokens = Some(0);

    let future = client
        .post("http://localhost:8989/v1/chat/completions")
        .json(&request)
        .send();

    let response = future.await.unwrap();

    assert!(response.status().is_success(), "{:?}", response);
    inc_counter(
        Endpoint::ChatCompletions,
        RequestType::Unary,
        Status::Success,
        &mut foo_counters,
    );
    compare_counters(&metrics, "foo", &foo_counters);
    compare_counters(&metrics, "bar", &bar_counters);
    // ==== ChatCompletions / Unary / Success ====

    // ==== ChatCompletions / Stream / Error ====
    request.model = "bar".to_string();

    // ALLOW: max_tokens is deprecated in favor of completion_usage_tokens
    request.max_tokens = Some(0);
    request.stream = Some(true);

    let response = client
        .post("http://localhost:8989/v1/chat/completions")
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    inc_counter(
        Endpoint::ChatCompletions,
        RequestType::Stream,
        Status::Error,
        &mut bar_counters,
    );
    compare_counters(&metrics, "foo", &foo_counters);
    compare_counters(&metrics, "bar", &bar_counters);
    // ==== ChatCompletions / Stream / Error ====

    // ==== ChatCompletions / Unary / Error ====
    request.stream = Some(false);

    let response = client
        .post("http://localhost:8989/v1/chat/completions")
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::FORBIDDEN);
    inc_counter(
        Endpoint::ChatCompletions,
        RequestType::Unary,
        Status::Error,
        &mut bar_counters,
    );
    compare_counters(&metrics, "foo", &foo_counters);
    compare_counters(&metrics, "bar", &bar_counters);
    // ==== ChatCompletions / Unary / Error ====

    // ==== Completions / Unary / Error ====
    let mut request = async_openai::types::CreateCompletionRequestArgs::default()
        .model("bar")
        .prompt("hi")
        .build()
        .unwrap();

    let response = client
        .post("http://localhost:8989/v1/completions")
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    inc_counter(
        Endpoint::Completions,
        RequestType::Unary,
        Status::Error,
        &mut bar_counters,
    );
    compare_counters(&metrics, "foo", &foo_counters);
    compare_counters(&metrics, "bar", &bar_counters);
    // ==== Completions / Unary / Error ====

    // ==== Completions / Stream / Error ====
    request.stream = Some(true);

    let response = client
        .post("http://localhost:8989/v1/completions")
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    inc_counter(
        Endpoint::Completions,
        RequestType::Stream,
        Status::Error,
        &mut bar_counters,
    );
    compare_counters(&metrics, "foo", &foo_counters);
    compare_counters(&metrics, "bar", &bar_counters);
    // ==== Completions / Stream / Error ====

    // =========== Test Invalid Request ===========
    // send a completion request to a chat endpoint
    request.stream = Some(false);

    let response = client
        .post("http://localhost:8989/v1/chat/completions")
        .json(&request)
        .send()
        .await
        .unwrap();

    assert_eq!(
        response.status(),
        StatusCode::UNPROCESSABLE_ENTITY,
        "{:?}",
        response
    );

    // =========== Query /metrics endpoint ===========
    let response = client
        .get("http://localhost:8989/metrics")
        .send()
        .await
        .unwrap();

    assert!(response.status().is_success(), "{:?}", response);
    println!("{}", response.text().await.unwrap());

    cancel_token.cancel();
    task.await.unwrap().unwrap();
}

// === HTTP Client Tests ===

/// Wait for the HTTP service to be ready by checking its health endpoint
async fn wait_for_service_ready(port: u16) {
    let start = tokio::time::Instant::now();
    let timeout = tokio::time::Duration::from_secs(5);
    loop {
        match reqwest::get(&format!("http://localhost:{}/health", port)).await {
            Ok(_) => break,
            Err(_) if start.elapsed() < timeout => {
                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            }
            Err(e) => panic!("Service failed to start within timeout: {}", e),
        }
    }
}

#[fixture]
fn service_with_engines(
    #[default(8990)] port: u16,
) -> (HttpService, Arc<CounterEngine>, Arc<AlwaysFailEngine>) {
    let service = HttpService::builder().port(port).build().unwrap();
    let manager = service.model_manager();

    let counter = Arc::new(CounterEngine {});
    let failure = Arc::new(AlwaysFailEngine {});

    manager
        .add_chat_completions_model("foo", counter.clone())
        .unwrap();
    manager
        .add_chat_completions_model("bar", failure.clone())
        .unwrap();
    manager
        .add_completions_model("bar", failure.clone())
        .unwrap();

    (service, counter, failure)
}

#[fixture]
fn pure_openai_client(#[default(8990)] port: u16) -> PureOpenAIClient {
    let config = HttpClientConfig {
        openai_config: OpenAIConfig::new().with_api_base(format!("http://localhost:{}/v1", port)),
        verbose: false,
    };
    PureOpenAIClient::new(config)
}

#[fixture]
fn nv_custom_client(#[default(8991)] port: u16) -> NvCustomClient {
    let config = HttpClientConfig {
        openai_config: OpenAIConfig::new().with_api_base(format!("http://localhost:{}/v1", port)),
        verbose: false,
    };
    NvCustomClient::new(config)
}

#[fixture]
fn generic_byot_client(#[default(8992)] port: u16) -> GenericBYOTClient {
    let config = HttpClientConfig {
        openai_config: OpenAIConfig::new().with_api_base(format!("http://localhost:{}/v1", port)),
        verbose: false,
    };
    GenericBYOTClient::new(config)
}

#[rstest]
#[tokio::test]
async fn test_pure_openai_client(
    #[with(8990)] service_with_engines: (HttpService, Arc<CounterEngine>, Arc<AlwaysFailEngine>),
    #[with(8990)] pure_openai_client: PureOpenAIClient,
) {
    let (service, _counter, _failure) = service_with_engines;
    let token = CancellationToken::new();
    let cancel_token = token.clone();

    // Start the service
    let task = tokio::spawn(async move { service.run(token).await });

    // Wait for service to be ready
    wait_for_service_ready(8990).await;

    // Test successful streaming request
    let request = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("foo")
        .messages(vec![
            async_openai::types::ChatCompletionRequestMessage::User(
                async_openai::types::ChatCompletionRequestUserMessage {
                    content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                        "Hi".to_string(),
                    ),
                    name: None,
                },
            ),
        ])
        .stream(true)
        .max_tokens(50u32)
        .build()
        .unwrap();

    let result = pure_openai_client.chat_stream(request).await;
    assert!(result.is_ok(), "PureOpenAI client should succeed");

    let (mut stream, _context) = result.unwrap().dissolve();
    let mut count = 0;
    while let Some(response) = stream.next().await {
        count += 1;
        assert!(response.is_ok(), "Response should be ok");
        if count >= 3 {
            break; // Don't consume entire stream
        }
    }
    assert!(count > 0, "Should receive at least one response");

    // Test error case with invalid model
    let request = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("bar") // This model will fail
        .messages(vec![
            async_openai::types::ChatCompletionRequestMessage::User(
                async_openai::types::ChatCompletionRequestUserMessage {
                    content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                        "Hi".to_string(),
                    ),
                    name: None,
                },
            ),
        ])
        .stream(true)
        .max_tokens(50u32)
        .build()
        .unwrap();

    let result = pure_openai_client.chat_stream(request).await;
    assert!(
        result.is_ok(),
        "Client should return stream even for failing model"
    );

    let (mut stream, _context) = result.unwrap().dissolve();
    if let Some(response) = stream.next().await {
        assert!(
            response.is_err(),
            "Response should be error for failing model"
        );
    }

    // Test context management
    let ctx = HttpRequestContext::new();
    let request = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("foo")
        .messages(vec![
            async_openai::types::ChatCompletionRequestMessage::User(
                async_openai::types::ChatCompletionRequestUserMessage {
                    content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                        "Hi".to_string(),
                    ),
                    name: None,
                },
            ),
        ])
        .stream(true)
        .max_tokens(50u32)
        .build()
        .unwrap();

    let result = pure_openai_client
        .chat_stream_with_context(request, ctx.clone())
        .await;
    assert!(result.is_ok(), "Context-based request should succeed");

    let (_stream, context) = result.unwrap().dissolve();
    assert_eq!(context.id(), ctx.id(), "Context ID should match");

    cancel_token.cancel();
    task.await.unwrap().unwrap();
}

#[rstest]
#[tokio::test]
async fn test_nv_custom_client(
    #[with(8991)] service_with_engines: (HttpService, Arc<CounterEngine>, Arc<AlwaysFailEngine>),
    #[with(8991)] nv_custom_client: NvCustomClient,
) {
    let (service, _counter, _failure) = service_with_engines;
    let token = CancellationToken::new();
    let cancel_token = token.clone();

    // Start the service
    let task = tokio::spawn(async move { service.run(token).await });

    // Wait for service to be ready
    wait_for_service_ready(8991).await;

    // Test successful streaming request
    let inner_request = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("foo")
        .messages(vec![
            async_openai::types::ChatCompletionRequestMessage::User(
                async_openai::types::ChatCompletionRequestUserMessage {
                    content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                        "Hi".to_string(),
                    ),
                    name: None,
                },
            ),
        ])
        .stream(true)
        .max_tokens(50u32)
        .build()
        .unwrap();

    let request = NvCreateChatCompletionRequest {
        inner: inner_request,
        nvext: None,
    };

    let result = nv_custom_client.chat_stream(request).await;
    assert!(result.is_ok(), "NvCustom client should succeed");

    let (mut stream, _context) = result.unwrap().dissolve();
    let mut count = 0;
    while let Some(response) = stream.next().await {
        count += 1;
        assert!(response.is_ok(), "Response should be ok");
        if count >= 3 {
            break; // Don't consume entire stream
        }
    }
    assert!(count > 0, "Should receive at least one response");

    // Test error case with invalid model
    let inner_request = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("bar") // This model will fail
        .messages(vec![
            async_openai::types::ChatCompletionRequestMessage::User(
                async_openai::types::ChatCompletionRequestUserMessage {
                    content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                        "Hi".to_string(),
                    ),
                    name: None,
                },
            ),
        ])
        .stream(true)
        .max_tokens(50u32)
        .build()
        .unwrap();

    let request = NvCreateChatCompletionRequest {
        inner: inner_request,
        nvext: None,
    };

    let result = nv_custom_client.chat_stream(request).await;
    assert!(
        result.is_ok(),
        "Client should return stream even for failing model"
    );

    let (mut stream, _context) = result.unwrap().dissolve();
    if let Some(response) = stream.next().await {
        assert!(
            response.is_err(),
            "Response should be error for failing model"
        );
    }

    // Test context management
    let ctx = HttpRequestContext::new();
    let inner_request = async_openai::types::CreateChatCompletionRequestArgs::default()
        .model("foo")
        .messages(vec![
            async_openai::types::ChatCompletionRequestMessage::User(
                async_openai::types::ChatCompletionRequestUserMessage {
                    content: async_openai::types::ChatCompletionRequestUserMessageContent::Text(
                        "Hi".to_string(),
                    ),
                    name: None,
                },
            ),
        ])
        .stream(true)
        .max_tokens(50u32)
        .build()
        .unwrap();

    let request = NvCreateChatCompletionRequest {
        inner: inner_request,
        nvext: None,
    };

    let result = nv_custom_client
        .chat_stream_with_context(request, ctx.clone())
        .await;
    assert!(result.is_ok(), "Context-based request should succeed");

    let (_stream, context) = result.unwrap().dissolve();
    assert_eq!(context.id(), ctx.id(), "Context ID should match");

    cancel_token.cancel();
    task.await.unwrap().unwrap();
}

#[rstest]
#[tokio::test]
async fn test_generic_byot_client(
    #[with(8992)] service_with_engines: (HttpService, Arc<CounterEngine>, Arc<AlwaysFailEngine>),
    #[with(8992)] generic_byot_client: GenericBYOTClient,
) {
    let (service, _counter, _failure) = service_with_engines;
    let token = CancellationToken::new();
    let cancel_token = token.clone();

    // Start the service
    let task = tokio::spawn(async move { service.run(token).await });

    // Wait for service to be ready
    wait_for_service_ready(8992).await;

    // Test successful streaming request
    let request = serde_json::json!({
        "model": "foo",
        "messages": [
            {
                "role": "user",
                "content": "Hi"
            }
        ],
        "stream": true,
        "max_tokens": 50
    });

    let result = generic_byot_client.chat_stream(request).await;
    assert!(result.is_ok(), "GenericBYOT client should succeed");

    let (mut stream, _context) = result.unwrap().dissolve();
    let mut count = 0;
    while let Some(response) = stream.next().await {
        println!("Response: {:?}", response);
        count += 1;
        assert!(response.is_ok(), "Response should be ok");
        if count >= 3 {
            break; // Don't consume entire stream
        }
    }
    assert!(count > 0, "Should receive at least one response");

    // Test error case with invalid model
    let request = serde_json::json!({
        "model": "bar", // This model will fail
        "messages": [
            {
                "role": "user",
                "content": "Hi"
            }
        ],
        "stream": true,
        "max_tokens": 50
    });

    let result = generic_byot_client.chat_stream(request).await;
    assert!(
        result.is_ok(),
        "Client should return stream even for failing model"
    );

    let (mut stream, _context) = result.unwrap().dissolve();
    if let Some(response) = stream.next().await {
        assert!(
            response.is_err(),
            "Response should be error for failing model"
        );
    }

    // Test context management
    let ctx = HttpRequestContext::new();
    let request = serde_json::json!({
        "model": "foo",
        "messages": [
            {
                "role": "user",
                "content": "Hi"
            }
        ],
        "stream": true,
        "max_tokens": 50
    });

    let result = generic_byot_client
        .chat_stream_with_context(request, ctx.clone())
        .await;
    assert!(result.is_ok(), "Context-based request should succeed");

    let (_stream, context) = result.unwrap().dissolve();
    assert_eq!(context.id(), ctx.id(), "Context ID should match");

    cancel_token.cancel();
    task.await.unwrap().unwrap();
}
