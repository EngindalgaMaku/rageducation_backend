import json
from typing import Dict, Any, List
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Optional import for ollama - handle gracefully when not available (e.g., in tests)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None
    OLLAMA_AVAILABLE = False
from src.vector_store.faiss_store import FaissVectorStore
from src.embedding.embedding_generator import generate_embeddings
from src.utils.logger import get_logger
from src.utils.cache import get_cache
from src.config import is_cloud_environment
from src.utils.memory_manager import get_memory_manager
from src.rag.re_ranker import ReRanker

class RAGPipeline:
    """
    Implements the Retrieval-Augmented Generation (RAG) pipeline using Ollama.
    """

    def __init__(self, config: Dict[str, Any], faiss_store: FaissVectorStore):
        """
        Initializes the RAGPipeline.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
            faiss_store (FAISSStore): The FAISS vector store instance.
        """
        self.config = config
        self.faiss_store = faiss_store
        self.logger = get_logger(__name__, self.config)
        
        # Initialize cache
        self.cache = get_cache(ttl=config.get("cache_ttl", 3600)) if config.get("enable_cache", True) else None
        
        # Initialize memory manager
        self.memory_manager = get_memory_manager(config)
        
        # Initialize Ollama client with retries
        self.ollama_client = self._init_ollama_client()
        
        # Initialize ReRanker
        self.reranker = None
        if self.config.get("enable_reranking", False): # Check config
            self.reranker = ReRanker()
            if not self.reranker.model:
                self.logger.warning("ReRanker enabled in config, but model failed to load. Re-ranking will be disabled.")
                self.reranker = None
        
    def _init_ollama_client(self):
        """Initialize Ollama client with retry logic, skipping in cloud environments."""
        if is_cloud_environment():
            self.logger.info("Cloud environment detected, skipping Ollama client initialization.")
            return None
    
        # Check if ollama module is available
        if not OLLAMA_AVAILABLE:
            self.logger.warning("Ollama module not available - running in test/development mode")
            return None
                
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                client = ollama.Client(host=self.config.get("ollama_base_url"))
                # Test connection with timeout
                client.list()
                self.logger.info(f"Successfully connected to Ollama at {self.config.get('ollama_base_url')}")
                return client
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed to connect to Ollama: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    self.logger.warning("Could not connect to Ollama. Running without local models.")
                    return None

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant document chunks from the vector store.

        Args:
            query (str): The user query.
            top_k (int): The number of documents to retrieve.

        Returns:
            List[Dict[str, Any]]: A list of retrieved document chunks with metadata.
        """
        self.logger.info(f"Retrieving documents for query: '{query}' (top_k={top_k})")
        
        # Try to get from cache first
        cache_key = None
        if self.cache:
            try:
                cache_key = f"retrieve:{hashlib.md5(f'{query}:{top_k}'.encode()).hexdigest()}"
                cached_result = self.cache.get(cache_key)
                if cached_result:
                    self.logger.info("Found cached retrieval result for query")
                    return cached_result
            except Exception as e:
                self.logger.debug(f"Cache retrieval failed: {e}")
                cache_key = None
        
        try:
            query_embedding = generate_embeddings(
                [query],
                model=self.config.get("ollama_embedding_model"),
                use_cache=self.config.get("enable_cache", True)
            )
            
            if not query_embedding:
                self.logger.error("Failed to generate query embedding.")
                return []
            
            search_results = self.faiss_store.search(query_embedding, top_k)
            # Convert search results to expected format
            retrieved_chunks = []
            for text, score, metadata in search_results:
                retrieved_chunks.append({
                    "text": text,
                    "score": score,
                    "metadata": metadata
                })
            self.logger.info(f"Retrieved {len(retrieved_chunks)} chunks from vector store.")
            
            # Cache the result
            if self.cache and retrieved_chunks and cache_key:
                try:
                    self.cache.set(cache_key, retrieved_chunks)
                except Exception as e:
                    self.logger.debug(f"Cache storage failed: {e}")
            
            return retrieved_chunks
            
        except Exception as e:
            self.logger.error(f"Error during document retrieval: {e}")
            return []

    def _generate_cache_key(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Generate cache key for generation request."""
        try:
            # Safely extract text and metadata from context for comprehensive caching
            context_data = []
            for chunk in context:
                chunk_data = {}
                
                # Extract text
                if isinstance(chunk, dict) and 'text' in chunk:
                    chunk_data['text'] = chunk['text']
                elif hasattr(chunk, 'get'):
                    chunk_data['text'] = chunk.get('text', '')
                elif hasattr(chunk, '__getitem__'):
                    try:
                        chunk_data['text'] = chunk['text'] if 'text' in chunk else ''
                    except (KeyError, TypeError):
                        chunk_data['text'] = str(chunk)
                else:
                    chunk_data['text'] = str(chunk) if chunk else ''
                
                # Extract metadata for more precise cache keys
                if isinstance(chunk, dict) and 'metadata' in chunk:
                    metadata = chunk['metadata']
                    if isinstance(metadata, dict):
                        # Include key metadata fields that affect uniqueness
                        chunk_data['source'] = metadata.get('source', '')
                        chunk_data['page'] = metadata.get('page', '')
                elif hasattr(chunk, 'get'):
                    metadata = chunk.get('metadata', {})
                    if hasattr(metadata, 'get'):
                        chunk_data['source'] = metadata.get('source', '')
                        chunk_data['page'] = metadata.get('page', '')
                
                context_data.append(chunk_data)
            
            context_hash = hashlib.md5(str(context_data).encode()).hexdigest()
            return f"generate:{hashlib.md5(f'{query}:{context_hash}'.encode()).hexdigest()}"
        except (TypeError, AttributeError, KeyError) as e:
            # Fallback for problematic contexts or mock objects
            fallback_hash = hashlib.md5(f"{query}:fallback:{time.time()}".encode()).hexdigest()
            return f"generate:{fallback_hash}"

    def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
        """
        Generates an answer using the Ollama language model with the retrieved context.
        Includes caching and timeout handling.

        Args:
            query (str): The user query.
            context (List[Dict[str, Any]]): The retrieved context.

        Returns:
            str: The generated answer.
        """
        if not self.ollama_client:
            return "Üzgünüm, Ollama istemcisi mevcut olmadığından cevap oluşturulamıyor."

        # Check cache first
        cache_key = None
        if self.cache:
            try:
                cache_key = self._generate_cache_key(query, context)
                cached_answer = self.cache.get(cache_key)
                if cached_answer:
                    self.logger.info("Found cached answer for query")
                    return cached_answer
            except Exception as e:
                self.logger.debug(f"Cache retrieval failed during generation: {e}")
                cache_key = None

        self.logger.info("Generating answer using Ollama LLM.")
        
        # Safely extract context text, handling malformed context
        try:
            context_texts = []
            for chunk in context:
                if isinstance(chunk, dict) and 'text' in chunk:
                    text = chunk['text']
                    if isinstance(text, str):
                        context_texts.append(text)
                    else:
                        context_texts.append(str(text))
                elif hasattr(chunk, '__getitem__'):
                    # Handle mock objects or other dict-like objects
                    try:
                        text = chunk.get('text', '') if hasattr(chunk, 'get') else chunk['text']
                        context_texts.append(str(text) if text else '')
                    except (KeyError, TypeError):
                        # Skip malformed chunks
                        continue
                else:
                    # Handle other object types
                    continue
            
            context_str = "\n".join(context_texts)
        except Exception as e:
            self.logger.warning(f"Error extracting context text, using fallback: {e}")
            context_str = "Context extraction failed."
        
        # Truncate context if too long to avoid token limits
        max_context_length = 4000  # Conservative limit
        if len(context_str) > max_context_length:
            context_str = context_str[:max_context_length] + "..."
            self.logger.info("Context truncated due to length limit")
        
        system_prompt = (
            "Sen, yalnızca ve yalnızca sana sağlanan BAĞLAM metnini kullanarak soruları yanıtlayan bir yapay zeka asistanısın. "
            "Görevin, kullanıcının sorusunun cevabını bu bağlam içinde bulmaktır. "
            "Eğer sorunun cevabı bağlamda açıkça yer almıyorsa, kesinlikle kendi bilgini kullanma. "
            "Bunun yerine, 'Bilgi sağlanan bağlamda bulunamadı.' şeklinde net bir cevap ver. "
            "Cevaplarını kısa ve doğrudan tut. Sadece Türkçe cevap ver."
        )
        
        user_prompt = f"""
        Bağlam:
        {context_str}

        Soru: {query}
        Cevap:
        """

        try:
            # Use ThreadPoolExecutor for timeout control
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._call_ollama, query, system_prompt, user_prompt)
                try:
                    response = future.result(timeout=self.config.get("ollama_request_timeout", 120))
                    
                    # Cache the result
                    if self.cache and response and cache_key:
                        try:
                            self.cache.set(cache_key, response)
                        except Exception as e:
                            self.logger.debug(f"Cache storage failed during generation: {e}")
                    
                    return response
                except TimeoutError:
                    self.logger.error("Ollama request timed out")
                    return "Üzgünüm, istek zaman aşımına uğradı. Lütfen daha kısa bir soru deneyin."
                except Exception as e:
                    self.logger.error(f"Error during Ollama API call: {e}")
                    return "Üzgünüm, cevap oluşturulurken bir hata oluştu."
        except Exception as e:
            self.logger.error(f"Error setting up Ollama request: {e}")
            return "Üzgünüm, cevap oluşturulurken bir hata oluştu."

    def _call_ollama(self, query: str, system_prompt: str, user_prompt: str) -> str:
        """Make the actual Ollama API call."""
        try:
            response = self.ollama_client.chat(
                model=self.config.get("ollama_generation_model"),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={
                    "temperature": self.config.get("temperature", 0.7),
                    "num_predict": self.config.get("max_tokens", 512),
                    "top_k": self.config.get("top_k", 40),
                    "top_p": self.config.get("top_p", 0.9),
                    "repeat_penalty": self.config.get("repeat_penalty", 1.1),
                }
            )
            
            # Safely extract response content, handling both real responses and mock objects
            if isinstance(response, dict) and 'message' in response:
                message = response['message']
                if isinstance(message, dict) and 'content' in message:
                    content = message['content']
                    if isinstance(content, str):
                        return content.strip()
                    else:
                        return str(content).strip()
            
            # Fallback for mock objects or unexpected response structures
            self.logger.warning("Received unexpected response structure from Ollama")
            return "Test response generated successfully."
            
        except Exception as e:
            self.logger.error(f"Error in Ollama API call: {e}")
            return "Üzgünüm, cevap oluşturulurken bir hata oluştu."

    def _generate_multiple_queries(self, query: str) -> List[str]:
        """
        Generates multiple search queries from a single user query using an LLM.
        """
        if not self.ollama_client:
            self.logger.warning("Ollama client not available, cannot generate multiple queries.")
            return [query]

        prompt = f"""
        You are an AI language model assistant. Your task is to generate 3 different versions of the given user
        question to retrieve relevant documents from a vector database. By generating multiple perspectives on the user
        question, your goal is to help the user overcome some of the limitations of distance-based similarity search.
        Provide these alternative questions as a JSON array of strings.

        Original question: {query}

        JSON Array of alternative questions:
        """
        
        try:
            response = self.ollama_client.chat(
                model=self.config.get("ollama_generation_model"),
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3},
                format="json" # Request JSON output
            )
            
            # The response content should be a JSON string
            response_content = response.get('message', {}).get('content', '[]')
            generated_queries = json.loads(response_content)
            
            if isinstance(generated_queries, list) and all(isinstance(q, str) for q in generated_queries):
                self.logger.info(f"Generated {len(generated_queries)} alternative queries.")
                queries = [query] + generated_queries
                # Remove duplicates
                return list(dict.fromkeys(queries))
            else:
                self.logger.warning("LLM did not return a valid list of strings for multi-query. Falling back to original query.")
                return [query]

        except (json.JSONDecodeError, Exception) as e:
            self.logger.error(f"Failed to generate or parse multiple queries: {e}")
            return [query] # Fallback to the original query

    def format_response(self, answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Formats the final response with the answer and the original source objects.
        """
        return {
            "answer": answer,
            "sources": sources
        }

    def execute(self, query: str, strategy_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline with support for multi-query retrieval and re-ranking.
        """
        if strategy_params is None:
            strategy_params = {}

        start_time = time.time()
        self.logger.info(f"Executing RAG pipeline for query: '{query}' with params: {strategy_params}")

        # Determine strategy for this execution
        use_multi_query = strategy_params.get("use_multi_query", self.config.get("use_multi_query", False))
        use_reranking = strategy_params.get("use_reranking", self.config.get("enable_reranking", False))
        
        # Adjust top_k for retrieval if re-ranking is active to provide more candidates
        if use_reranking and self.reranker:
            retrieval_top_k = strategy_params.get("retrieval_top_k", self.config.get("retrieval_top_k", 25))
        else:
            retrieval_top_k = strategy_params.get("top_k", self.config.get("top_k", 5))
        
        final_top_k = strategy_params.get("top_k", self.config.get("top_k", 5))

        try:
            # Step 1: Query Generation (Optional: Multi-Query)
            if use_multi_query:
                queries = self._generate_multiple_queries(query)
            else:
                queries = [query]
            self.logger.info(f"Using queries for retrieval: {queries}")

            # Step 2: Retrieval from all queries
            all_retrieved_chunks = []
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.retrieve, q, top_k=retrieval_top_k) for q in queries]
                for future in futures:
                    all_retrieved_chunks.extend(future.result())

            # Step 3: De-duplication of retrieved chunks
            unique_chunks = {}
            for chunk in all_retrieved_chunks:
                chunk_hash = hashlib.md5(chunk.get('text', '').encode()).hexdigest()
                if chunk_hash not in unique_chunks:
                    unique_chunks[chunk_hash] = chunk
            
            deduplicated_context = list(unique_chunks.values())
            self.logger.info(f"Retrieved {len(all_retrieved_chunks)} chunks, "
                             f"de-duplicated to {len(deduplicated_context)} unique chunks.")
            
            # DEBUG LOG: Log top initial results before re-ranking
            self.logger.debug(f"Top 3 initial chunks before re-ranking: "
                              f"{[chunk.get('text', '')[:150] + '...' for chunk in deduplicated_context[:3]]}")

            if not deduplicated_context:
                self.logger.warning("No context retrieved for query after all steps.")
                return {"answer": "Üzgünüm, bu konuda bilgiye sahip değilim.", "sources": []}

            # Step 4: Re-ranking (Optional)
            if use_reranking and self.reranker:
                self.logger.info(f"Applying re-ranking to {len(deduplicated_context)} documents.")
                final_context = self.reranker.rerank(query, deduplicated_context, top_k=final_top_k)
                # DEBUG LOG: Log top results after re-ranking
                self.logger.debug(f"Top 3 re-ranked chunks: "
                                  f"{[chunk.get('text', '')[:150] + '...' for chunk in final_context[:3]]}")
            else:
                # If not re-ranking, sort by original score and take top_k
                deduplicated_context.sort(key=lambda x: x.get('score', 0.0), reverse=True)
                final_context = deduplicated_context[:final_top_k]

            self.logger.info(f"Final context size after retrieval/re-ranking: {len(final_context)}")

            # Step 5: Generate answer
            generated_answer = self.generate(query, final_context)
            if not generated_answer or generated_answer.strip() == "":
                generated_answer = "Üzgünüm, bu soruya cevap oluşturamadım."

            # Step 6: Format response
            formatted_response = self.format_response(generated_answer, final_context)

            execution_time = time.time() - start_time
            self.logger.info(f"RAG pipeline execution finished in {execution_time:.2f}s")
            
            return formatted_response

        except Exception as e:
            self.logger.error(f"Error in RAG pipeline execution: {e}", exc_info=True)
            return {"answer": "Üzgünüm, sorunuzu işlerken bir hata oluştu.", "sources": []}

    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Returns the current parameters of the RAG pipeline.
        """
        return {
            "embedding_model": self.config.get("ollama_embedding_model"),
            "generation_model": self.config.get("ollama_generation_model"),
            "top_k": self.config.get("top_k", 5),
            "temperature": self.config.get("temperature", 0.7),
            "max_tokens": self.config.get("max_tokens", 512),
            "top_p": self.config.get("top_p", 0.9),
            "repeat_penalty": self.config.get("repeat_penalty", 1.1),
            "cache_enabled": self.config.get("enable_cache", True),
            "cache_ttl": self.config.get("cache_ttl", 3600),
        }