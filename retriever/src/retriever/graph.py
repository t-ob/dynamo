from retriever.components.frontend import Frontend
from retriever.components.processor import Processor
from retriever.components.trt_worker import TrtWorker

Frontend.link(Processor).link(TrtWorker)
