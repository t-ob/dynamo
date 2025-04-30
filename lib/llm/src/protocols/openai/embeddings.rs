use std::collections::HashMap;

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use validator::Validate;

mod aggregator;
mod delta;
mod nvext;

// pub use aggregator::DeltaAggregator;
// pub use delta::DeltaGenerator;
pub use nvext::{NvExt, NvExtProvider};

use super::{
    common::{self, SamplingOptionsProvider, StopConditionsProvider},
    // nvext::{NvExt, NvExtProvider},
    CompletionUsage, ContentProvider, OpenAISamplingOptionsProvider, OpenAIStopConditionsProvider,
};

use dynamo_runtime::protocols::annotated::AnnotationsProvider;

// #[derive(Serialize, Deserialize, Validate, Debug, Clone)]
// pub struct NvCreateEmbeddingRequest {
//     #[serde(flatten)]
//     pub inner: async_openai::types::CreateEmbeddingRequest
// }

#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateEmbeddingRequest {
    #[serde(flatten)]
    pub inner: async_openai::types::CreateEmbeddingRequest,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub nvext: Option<NvExt>,
}

/// A response structure for unary chat completion responses, embedding OpenAI's
/// `CreateChatCompletionResponse`.
///
/// # Fields
/// - `inner`: The base OpenAI unary chat completion response, embedded
///   using `serde(flatten)`.
#[derive(Serialize, Deserialize, Validate, Debug, Clone)]
pub struct NvCreateEmbeddingResponse {
    #[serde(flatten)]
    pub inner: async_openai::types::CreateEmbeddingResponse,
}

/// Implements `NvExtProvider` for `NvCr    eateEmbeddingRequest`,
/// providing access to NVIDIA-specific extensions.
impl NvExtProvider for NvCreateEmbeddingRequest {
    /// Returns a reference to the optional `NvExt` extension, if available.
    fn nvext(&self) -> Option<&NvExt> {
        self.nvext.as_ref()
    }

    /// Returns `None`, as raw prompt extraction is not implemented.
    fn raw_prompt(&self) -> Option<String> {
        None
    }
}

/// Implements `AnnotationsProvider` for `NvCreateEmbeddingRequest`,
/// enabling retrieval and management of request annotations.
impl AnnotationsProvider for NvCreateEmbeddingRequest {
    /// Retrieves the list of annotations from `NvExt`, if present.
    fn annotations(&self) -> Option<Vec<String>> {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.clone())
    }

    /// Checks whether a specific annotation exists in the request.
    ///
    /// # Arguments
    /// * `annotation` - A string slice representing the annotation to check.
    ///
    /// # Returns
    /// `true` if the annotation exists, `false` otherwise.
    fn has_annotation(&self, annotation: &str) -> bool {
        self.nvext
            .as_ref()
            .and_then(|nvext| nvext.annotations.as_ref())
            .map(|annotations| annotations.contains(&annotation.to_string()))
            .unwrap_or(false)
    }
}
