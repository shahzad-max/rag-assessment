"""
RAG system evaluator
"""

from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import time
from pathlib import Path
import json

from src.generation.rag_pipeline import RAGPipeline, RAGResponse
from src.evaluation.metrics import (
    RetrievalMetrics,
    GenerationMetrics,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_mrr,
    calculate_ndcg,
    calculate_map,
    calculate_bleu,
    calculate_rouge,
    calculate_bertscore,
    calculate_citation_metrics,
    calculate_answer_relevance,
    aggregate_metrics
)
from src.evaluation.ground_truth import GroundTruthManager, TestQuery
from src.utils import log


@dataclass
class EvaluationResult:
    """Results from evaluating a single query"""
    query_id: str
    query: str
    query_type: str
    
    # Response
    generated_answer: str
    generation_time: float
    
    # Retrieval metrics
    retrieval_metrics: RetrievalMetrics
    
    # Generation metrics
    generation_metrics: GenerationMetrics
    
    # Raw data
    retrieved_chunk_ids: List[str]
    relevant_chunk_ids: List[str]
    predicted_citations: List[str]
    expected_citations: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'query_id': self.query_id,
            'query': self.query,
            'query_type': self.query_type,
            'generated_answer': self.generated_answer,
            'generation_time': self.generation_time,
            'retrieval_metrics': self.retrieval_metrics.to_dict(),
            'generation_metrics': self.generation_metrics.to_dict(),
            'retrieved_chunk_ids': self.retrieved_chunk_ids,
            'relevant_chunk_ids': self.relevant_chunk_ids,
            'predicted_citations': self.predicted_citations,
            'expected_citations': self.expected_citations
        }


class RAGEvaluator:
    """Evaluates RAG system performance"""
    
    def __init__(
        self,
        pipeline: RAGPipeline,
        ground_truth: GroundTruthManager,
        k_values: Optional[List[int]] = None
    ):
        """
        Initialize evaluator
        
        Args:
            pipeline: RAG pipeline to evaluate
            ground_truth: Ground truth test queries
            k_values: List of k values for metrics (default: [1, 3, 5, 10])
        """
        self.pipeline = pipeline
        self.ground_truth = ground_truth
        self.k_values = k_values or [1, 3, 5, 10]
        
        self.results: List[EvaluationResult] = []
        
        log.info(f"Initialized RAGEvaluator with {len(ground_truth.test_queries)} test queries")
    
    def evaluate_query(
        self,
        test_query: TestQuery,
        top_k: int = 10,
        rerank_top_k: int = 5
    ) -> EvaluationResult:
        """
        Evaluate a single query
        
        Args:
            test_query: Test query with ground truth
            top_k: Number of chunks to retrieve
            rerank_top_k: Number of chunks after reranking
            
        Returns:
            EvaluationResult
        """
        log.info(f"Evaluating query: {test_query.query_id}")
        
        # Get response from pipeline
        start_time = time.time()
        response: RAGResponse = self.pipeline.query(
            query=test_query.query,
            top_k=top_k,
            rerank_top_k=rerank_top_k
        )
        generation_time = time.time() - start_time
        
        # Extract chunk IDs from retrieved results
        retrieved_chunk_ids = [
            chunk.split('[')[0].strip() if '[' in chunk else chunk[:50]
            for chunk in response.retrieved_chunks
        ]
        
        # Calculate retrieval metrics
        retrieval_metrics = self._calculate_retrieval_metrics(
            retrieved_chunk_ids=retrieved_chunk_ids,
            relevant_chunk_ids=test_query.relevant_chunk_ids
        )
        
        # Calculate generation metrics
        generation_metrics = self._calculate_generation_metrics(
            generated_answer=response.answer,
            expected_answer=test_query.expected_answer,
            predicted_citations=set(str(c) for c in response.citations),
            expected_citations=test_query.expected_citations,
            query=test_query.query,
            context=response.reranked_chunks
        )
        
        # Create result
        result = EvaluationResult(
            query_id=test_query.query_id,
            query=test_query.query,
            query_type=test_query.query_type,
            generated_answer=response.answer,
            generation_time=generation_time,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            retrieved_chunk_ids=retrieved_chunk_ids,
            relevant_chunk_ids=list(test_query.relevant_chunk_ids),
            predicted_citations=[str(c) for c in response.citations],
            expected_citations=list(test_query.expected_citations)
        )
        
        self.results.append(result)
        
        log.info(f"Query {test_query.query_id} evaluated successfully")
        
        return result
    
    def _calculate_retrieval_metrics(
        self,
        retrieved_chunk_ids: List[str],
        relevant_chunk_ids: set
    ) -> RetrievalMetrics:
        """Calculate retrieval metrics"""
        
        # Precision@K and Recall@K for different k values
        precision_at_k = {}
        recall_at_k = {}
        ndcg_at_k = {}
        
        for k in self.k_values:
            precision_at_k[k] = calculate_precision_at_k(
                retrieved_chunk_ids,
                relevant_chunk_ids,
                k
            )
            recall_at_k[k] = calculate_recall_at_k(
                retrieved_chunk_ids,
                relevant_chunk_ids,
                k
            )
            ndcg_at_k[k] = calculate_ndcg(
                retrieved_chunk_ids,
                relevant_chunk_ids,
                k
            )
        
        # MRR
        mrr = calculate_mrr(retrieved_chunk_ids, relevant_chunk_ids)
        
        # MAP
        map_score = calculate_map(retrieved_chunk_ids, relevant_chunk_ids)
        
        return RetrievalMetrics(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            mrr=mrr,
            ndcg_at_k=ndcg_at_k,
            map_score=map_score
        )
    
    def _calculate_generation_metrics(
        self,
        generated_answer: str,
        expected_answer: str,
        predicted_citations: set,
        expected_citations: set,
        query: str,
        context: List[str]
    ) -> GenerationMetrics:
        """Calculate generation metrics"""
        
        # BLEU score
        bleu_score = calculate_bleu(expected_answer, generated_answer)
        
        # ROUGE scores
        rouge_scores = calculate_rouge(expected_answer, generated_answer)
        
        # BERTScore (may be slow, can be disabled)
        try:
            bertscore = calculate_bertscore(expected_answer, generated_answer)
        except Exception as e:
            log.warning(f"BERTScore calculation failed: {e}")
            bertscore = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
        
        # Citation metrics
        citation_precision, citation_recall, citation_f1 = calculate_citation_metrics(
            predicted_citations,
            expected_citations
        )
        
        # Answer relevance
        answer_relevance = calculate_answer_relevance(
            generated_answer,
            query,
            context
        )
        
        return GenerationMetrics(
            bleu_score=bleu_score,
            rouge_scores=rouge_scores,
            bertscore=bertscore,
            citation_precision=citation_precision,
            citation_recall=citation_recall,
            citation_f1=citation_f1,
            answer_relevance=answer_relevance
        )
    
    def evaluate_all(
        self,
        top_k: int = 10,
        rerank_top_k: int = 5,
        query_types: Optional[List[str]] = None,
        difficulties: Optional[List[str]] = None
    ) -> List[EvaluationResult]:
        """
        Evaluate all test queries
        
        Args:
            top_k: Number of chunks to retrieve
            rerank_top_k: Number of chunks after reranking
            query_types: Filter by query types (None = all)
            difficulties: Filter by difficulties (None = all)
            
        Returns:
            List of EvaluationResult objects
        """
        # Filter queries
        queries = self.ground_truth.test_queries
        
        if query_types:
            queries = [q for q in queries if q.query_type in query_types]
        
        if difficulties:
            queries = [q for q in queries if q.difficulty in difficulties]
        
        log.info(f"Evaluating {len(queries)} queries")
        
        # Evaluate each query
        results = []
        for i, query in enumerate(queries, 1):
            log.info(f"Progress: {i}/{len(queries)}")
            
            try:
                result = self.evaluate_query(
                    test_query=query,
                    top_k=top_k,
                    rerank_top_k=rerank_top_k
                )
                results.append(result)
            except Exception as e:
                log.error(f"Error evaluating query {query.query_id}: {e}")
                continue
        
        log.info(f"Evaluation complete: {len(results)} queries evaluated")
        
        return results
    
    def get_aggregate_metrics(
        self,
        results: Optional[List[EvaluationResult]] = None
    ) -> Dict:
        """
        Get aggregated metrics across all results
        
        Args:
            results: List of results (uses self.results if None)
            
        Returns:
            Dictionary with aggregated metrics
        """
        results = results or self.results
        
        if not results:
            log.warning("No results to aggregate")
            return {}
        
        # Aggregate retrieval metrics
        retrieval_metrics_list = []
        for result in results:
            metrics = result.retrieval_metrics.to_dict()
            # Flatten nested dicts
            flat_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        flat_metrics[f"{key}_{k}"] = v
                else:
                    flat_metrics[key] = value
            retrieval_metrics_list.append(flat_metrics)
        
        retrieval_agg = aggregate_metrics(retrieval_metrics_list)
        
        # Aggregate generation metrics
        generation_metrics_list = []
        for result in results:
            metrics = result.generation_metrics.to_dict()
            # Flatten nested dicts
            flat_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        flat_metrics[f"{key}_{k}"] = v
                else:
                    flat_metrics[key] = value
            generation_metrics_list.append(flat_metrics)
        
        generation_agg = aggregate_metrics(generation_metrics_list)
        
        # Combine
        aggregated = {
            'num_queries': len(results),
            'retrieval': retrieval_agg,
            'generation': generation_agg,
            'avg_generation_time': sum(r.generation_time for r in results) / len(results)
        }
        
        return aggregated
    
    def get_metrics_by_query_type(self) -> Dict[str, Dict]:
        """Get metrics broken down by query type"""
        by_type = {}
        
        for query_type in ['fact', 'abstract', 'reasoning', 'comparative']:
            type_results = [r for r in self.results if r.query_type == query_type]
            if type_results:
                by_type[query_type] = self.get_aggregate_metrics(type_results)
        
        return by_type
    
    def save_results(self, filepath: str):
        """
        Save evaluation results to JSON file
        
        Args:
            filepath: Path to save file
        """
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'results': [r.to_dict() for r in self.results],
            'aggregate_metrics': self.get_aggregate_metrics(),
            'metrics_by_type': self.get_metrics_by_query_type(),
            'metadata': {
                'num_queries': len(self.results),
                'k_values': self.k_values,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        log.info(f"Saved evaluation results to: {save_path}")
    
    def generate_report(self, filepath: str):
        """
        Generate human-readable evaluation report
        
        Args:
            filepath: Path to save report
        """
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        agg_metrics = self.get_aggregate_metrics()
        by_type = self.get_metrics_by_query_type()
        
        report = []
        report.append("# RAG System Evaluation Report\n")
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append(f"Total Queries Evaluated: {len(self.results)}\n\n")
        
        # Overall metrics
        report.append("## Overall Performance\n\n")
        report.append("### Retrieval Metrics\n")
        
        if 'retrieval' in agg_metrics:
            for metric, value in sorted(agg_metrics['retrieval'].items()):
                if '_mean' in metric:
                    report.append(f"- {metric}: {value:.4f}\n")
        
        report.append("\n### Generation Metrics\n")
        
        if 'generation' in agg_metrics:
            for metric, value in sorted(agg_metrics['generation'].items()):
                if '_mean' in metric:
                    report.append(f"- {metric}: {value:.4f}\n")
        
        # By query type
        report.append("\n## Performance by Query Type\n\n")
        
        for query_type, metrics in by_type.items():
            report.append(f"### {query_type.capitalize()} Queries\n")
            report.append(f"Number of queries: {metrics.get('num_queries', 0)}\n\n")
            
            if 'retrieval' in metrics:
                report.append("**Retrieval:**\n")
                for metric, value in sorted(metrics['retrieval'].items()):
                    if '_mean' in metric:
                        report.append(f"- {metric}: {value:.4f}\n")
            
            if 'generation' in metrics:
                report.append("\n**Generation:**\n")
                for metric, value in sorted(metrics['generation'].items()):
                    if '_mean' in metric:
                        report.append(f"- {metric}: {value:.4f}\n")
            
            report.append("\n")
        
        # Write report
        with open(save_path, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        log.info(f"Generated evaluation report: {save_path}")


