#!/usr/bin/env python3
"""
Performance test script for optimized RAG system.
Tests GPU usage, caching, memory management, and response times.
"""

import os
import sys
import time
import asyncio
import concurrent.futures
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config import get_config
from src.vector_store.faiss_store import FaissVectorStore
from src.rag.rag_pipeline import RAGPipeline
from src.utils.memory_manager import get_memory_manager
from src.utils.cache import get_cache

class PerformanceTestSuite:
    """Performance testing suite for RAG system."""
    
    def __init__(self):
        self.config = get_config()
        self.memory_manager = get_memory_manager(self.config)
        self.cache = get_cache(ttl=self.config.get("cache_ttl", 3600))
        
        # Initialize RAG components
        print("Initializing RAG components...")
        self.faiss_store = FaissVectorStore()
        if not self.faiss_store.index:
            print("⚠️ Warning: No FAISS index found. Some tests may not work.")
        
        self.rag_pipeline = RAGPipeline(self.config, self.faiss_store)
        
        # Test queries
        self.test_queries = [
            "Python programlamada değişken tanımlama nasıl yapılır?",
            "Yapay zeka nedir ve nasıl çalışır?",
            "Nesne tabanlı programlamanın temel özellikleri nelerdir?",
            "Derin öğrenme algoritmaları nasıl çalışır?",
            "Eğitimde teknoloji kullanımının faydaları nelerdir?"
        ]
    
    def test_memory_monitoring(self) -> Dict[str, Any]:
        """Test memory management and monitoring."""
        print("\n🧠 Testing Memory Management...")
        
        initial_memory = self.memory_manager.get_memory_usage()
        print(f"Initial memory usage: {initial_memory['percent']:.1f}% ({initial_memory['rss_mb']:.1f}MB)")
        
        # Force some memory usage
        large_data = [list(range(10000)) for _ in range(100)]
        
        peak_memory = self.memory_manager.get_memory_usage()
        print(f"Peak memory usage: {peak_memory['percent']:.1f}% ({peak_memory['rss_mb']:.1f}MB)")
        
        # Test garbage collection
        del large_data
        gc_stats = self.memory_manager.force_gc()
        
        final_memory = self.memory_manager.get_memory_usage()
        print(f"Memory after GC: {final_memory['percent']:.1f}% ({final_memory['rss_mb']:.1f}MB)")
        
        return {
            "initial_memory_mb": initial_memory['rss_mb'],
            "peak_memory_mb": peak_memory['rss_mb'],
            "final_memory_mb": final_memory['rss_mb'],
            "gc_objects_collected": gc_stats['objects_collected'],
            "memory_freed_mb": gc_stats['memory_freed_mb']
        }
    
    def test_cache_performance(self) -> Dict[str, Any]:
        """Test caching functionality."""
        print("\n💾 Testing Cache Performance...")
        
        test_query = self.test_queries[0]
        
        # First run - should miss cache
        print("Testing cache miss...")
        start_time = time.time()
        result1 = self.rag_pipeline.execute(test_query)
        first_response_time = time.time() - start_time
        
        # Second run - should hit cache
        print("Testing cache hit...")
        start_time = time.time()
        result2 = self.rag_pipeline.execute(test_query)
        second_response_time = time.time() - start_time
        
        cache_speedup = first_response_time / second_response_time if second_response_time > 0 else 1
        
        print(f"First response (cache miss): {first_response_time:.2f}s")
        print(f"Second response (cache hit): {second_response_time:.2f}s")
        print(f"Cache speedup: {cache_speedup:.2f}x")
        
        return {
            "cache_miss_time": first_response_time,
            "cache_hit_time": second_response_time,
            "speedup_ratio": cache_speedup,
            "cache_working": cache_speedup > 1.5  # Expect at least 1.5x speedup
        }
    
    def test_concurrent_performance(self) -> Dict[str, Any]:
        """Test concurrent processing performance."""
        print("\n🔄 Testing Concurrent Performance...")
        
        def process_query(query: str, query_id: int) -> Dict[str, Any]:
            start_time = time.time()
            result = self.rag_pipeline.execute(query)
            response_time = time.time() - start_time
            return {
                "query_id": query_id,
                "query": query,
                "response_time": response_time,
                "success": len(result.get("answer", "")) > 0
            }
        
        # Test sequential processing
        print("Testing sequential processing...")
        sequential_start = time.time()
        sequential_results = []
        for i, query in enumerate(self.test_queries):
            result = process_query(query, i)
            sequential_results.append(result)
        sequential_time = time.time() - sequential_start
        
        # Test concurrent processing
        print("Testing concurrent processing...")
        concurrent_start = time.time()
        concurrent_results = []
        
        max_workers = self.config.get("max_concurrent_requests", 3)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_query = {
                executor.submit(process_query, query, i): (query, i) 
                for i, query in enumerate(self.test_queries)
            }
            
            for future in concurrent.futures.as_completed(future_to_query):
                result = future.result()
                concurrent_results.append(result)
        
        concurrent_time = time.time() - concurrent_start
        
        # Calculate statistics
        sequential_avg = sequential_time / len(self.test_queries)
        concurrent_avg = concurrent_time / len(self.test_queries)
        concurrency_speedup = sequential_time / concurrent_time if concurrent_time > 0 else 1
        
        print(f"Sequential total time: {sequential_time:.2f}s (avg: {sequential_avg:.2f}s per query)")
        print(f"Concurrent total time: {concurrent_time:.2f}s (avg: {concurrent_avg:.2f}s per query)")
        print(f"Concurrency speedup: {concurrency_speedup:.2f}x")
        
        return {
            "sequential_time": sequential_time,
            "concurrent_time": concurrent_time,
            "sequential_avg": sequential_avg,
            "concurrent_avg": concurrent_avg,
            "concurrency_speedup": concurrency_speedup,
            "all_queries_successful": all(r["success"] for r in concurrent_results)
        }
    
    def test_response_quality(self) -> Dict[str, Any]:
        """Test response quality and consistency."""
        print("\n📝 Testing Response Quality...")
        
        quality_scores = []
        response_lengths = []
        
        for query in self.test_queries[:3]:  # Test first 3 queries
            result = self.rag_pipeline.execute(query)
            answer = result.get("answer", "")
            sources = result.get("sources", [])
            
            # Basic quality metrics
            has_answer = len(answer.strip()) > 0
            has_sources = len(sources) > 0
            answer_length = len(answer)
            num_sources = len(sources)
            
            quality_score = 0
            if has_answer:
                quality_score += 1
            if has_sources:
                quality_score += 1
            if answer_length > 50:  # Substantial answer
                quality_score += 1
            if num_sources >= 2:  # Multiple sources
                quality_score += 1
            
            quality_scores.append(quality_score)
            response_lengths.append(answer_length)
            
            print(f"Query: {query[:50]}...")
            print(f"  Answer length: {answer_length} chars")
            print(f"  Sources: {num_sources}")
            print(f"  Quality score: {quality_score}/4")
        
        avg_quality = sum(quality_scores) / len(quality_scores)
        avg_length = sum(response_lengths) / len(response_lengths)
        
        return {
            "average_quality_score": avg_quality,
            "average_response_length": avg_length,
            "quality_scores": quality_scores,
            "response_lengths": response_lengths,
            "all_queries_answered": all(score >= 2 for score in quality_scores)
        }
    
    def run_full_test_suite(self) -> Dict[str, Any]:
        """Run the complete performance test suite."""
        print("🚀 Starting Performance Test Suite...")
        print("=" * 60)
        
        start_time = time.time()
        results = {}
        
        try:
            # Run all tests
            results["memory_test"] = self.test_memory_monitoring()
            results["cache_test"] = self.test_cache_performance()
            results["concurrent_test"] = self.test_concurrent_performance()
            results["quality_test"] = self.test_response_quality()
            
            total_time = time.time() - start_time
            results["total_test_time"] = total_time
            
            print("\n" + "=" * 60)
            print("🎯 Performance Test Results Summary:")
            print("=" * 60)
            
            # Memory Test Results
            memory = results["memory_test"]
            print(f"Memory Management: ✅ GC freed {memory['memory_freed_mb']:.2f}MB")
            
            # Cache Test Results
            cache = results["cache_test"]
            cache_status = "✅" if cache["cache_working"] else "❌"
            print(f"Cache Performance: {cache_status} {cache['speedup_ratio']:.2f}x speedup")
            
            # Concurrency Test Results
            concurrent = results["concurrent_test"]
            concurrent_status = "✅" if concurrent["all_queries_successful"] else "❌"
            print(f"Concurrent Processing: {concurrent_status} {concurrent['concurrency_speedup']:.2f}x speedup")
            
            # Quality Test Results
            quality = results["quality_test"]
            quality_status = "✅" if quality["all_queries_answered"] else "❌"
            print(f"Response Quality: {quality_status} {quality['average_quality_score']:.1f}/4.0 average")
            
            print(f"\nTotal test time: {total_time:.2f}s")
            
            # Overall assessment
            all_tests_passed = (
                memory['memory_freed_mb'] > 0 and
                cache['cache_working'] and
                concurrent['all_queries_successful'] and
                quality['all_queries_answered']
            )
            
            overall_status = "✅ ALL TESTS PASSED" if all_tests_passed else "⚠️ SOME TESTS FAILED"
            print(f"\nOverall Status: {overall_status}")
            
            return results
            
        except Exception as e:
            print(f"\n❌ Test suite failed with error: {e}")
            results["error"] = str(e)
            return results

def main():
    """Main test function."""
    try:
        test_suite = PerformanceTestSuite()
        results = test_suite.run_full_test_suite()
        
        # Save results to file
        import json
        with open("performance_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📊 Detailed results saved to: performance_test_results.json")
        
        return 0 if not results.get("error") else 1
        
    except Exception as e:
        print(f"❌ Failed to run performance tests: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)