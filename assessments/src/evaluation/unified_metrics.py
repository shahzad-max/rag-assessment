"""
Unified Metrics Calculator
Calculates ALL comprehensive metrics in one place
Used by API and Live Evaluation
"""

import time
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass, asdict

from src.evaluation.metrics import (
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_mrr,
    calculate_ndcg,
    calculate_map,
    calculate_bleu,
    calculate_rouge,
    calculate_bertscore,
    calculate_citation_metrics,
    calculate_answer_relevance
)
from src.utils import log


@dataclass
class ComprehensiveMetrics:
    """All metrics in one place"""
    # Timing
    latency_ms: float
    
    # Retrieval Metrics (require ground truth relevant docs)
    precision_at_1: Optional[float] = None
    precision_at_3: Optional[float] = None
    precision_at_5: Optional[float] = None
    precision_at_10: Optional[float] = None
    recall_at_1: Optional[float] = None
    recall_at_3: Optional[float] = None
    recall_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None
    mrr: Optional[float] = None
    ndcg_at_5: Optional[float] = None
    ndcg_at_10: Optional[float] = None
    map_score: Optional[float] = None
    
    # Generation Metrics (require ground truth answer)
    bleu_score: Optional[float] = None
    rouge1: Optional[float] = None
    rouge2: Optional[float] = None
    rougeL: Optional[float] = None
    bertscore_precision: Optional[float] = None
    bertscore_recall: Optional[float] = None
    bertscore_f1: Optional[float] = None
    
    # Citation Metrics (require ground truth citations)
    citation_precision: Optional[float] = None
    citation_recall: Optional[float] = None
    citation_f1: Optional[float] = None
    
    # Answer Quality Metrics (always calculated)
    answer_relevance: float = 0.0
    answer_length: int = 0
    answer_words: int = 0
    num_citations: int = 0
    
    # Retrieval Quality Metrics (always calculated)
    num_context_chunks: int = 0
    avg_retrieval_score: float = 0.0
    max_retrieval_score: float = 0.0
    min_retrieval_score: float = 0.0
    
    # Simple overlap metrics (if ground truth provided)
    token_f1: Optional[float] = None
    exact_match: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}


class UnifiedMetricsCalculator:
    """
    Calculates ALL metrics in one place
    Used by both API and Live Evaluation
    """
    
    def __init__(self):
        """Initialize calculator"""
        self.k_values = [1, 3, 5, 10]
    
    def calculate_all_metrics(
        self,
        query: str,
        answer: str,
        retrieved_chunks: List,
        retrieved_chunk_ids: List[str],
        context_texts: List[str],
        predicted_citations: Set[str],
        latency_ms: float,
        ground_truth_answer: Optional[str] = None,
        ground_truth_citations: Optional[Set[str]] = None,
        relevant_chunk_ids: Optional[Set[str]] = None
    ) -> ComprehensiveMetrics:
        """
        Calculate ALL comprehensive metrics
        
        Args:
            query: User query
            answer: Generated answer
            retrieved_chunks: List of retrieved chunk objects with scores
            retrieved_chunk_ids: List of retrieved chunk IDs (ordered)
            context_texts: List of context text strings
            predicted_citations: Set of predicted citations
            latency_ms: Response latency in milliseconds
            ground_truth_answer: Optional ground truth answer
            ground_truth_citations: Optional ground truth citations
            relevant_chunk_ids: Optional set of relevant chunk IDs
            
        Returns:
            ComprehensiveMetrics object with all calculated metrics
        """
        metrics = ComprehensiveMetrics(latency_ms=latency_ms)
        
        # ===== ALWAYS CALCULATED METRICS =====
        
        # Answer quality
        metrics.answer_length = len(answer)
        metrics.answer_words = len(answer.split())
        metrics.num_citations = answer.count('[')
        metrics.answer_relevance = calculate_answer_relevance(answer, query, context_texts)
        
        # Retrieval quality
        metrics.num_context_chunks = len(retrieved_chunks)
        if retrieved_chunks:
            scores = [chunk.score for chunk in retrieved_chunks if hasattr(chunk, 'score')]
            if scores:
                metrics.avg_retrieval_score = sum(scores) / len(scores)
                metrics.max_retrieval_score = max(scores)
                metrics.min_retrieval_score = min(scores)
        
        # ===== RETRIEVAL METRICS (require ground truth relevant docs) =====
        
        if relevant_chunk_ids:
            for k in self.k_values:
                # Precision@K
                precision = calculate_precision_at_k(retrieved_chunk_ids, relevant_chunk_ids, k)
                setattr(metrics, f'precision_at_{k}', precision)
                
                # Recall@K
                recall = calculate_recall_at_k(retrieved_chunk_ids, relevant_chunk_ids, k)
                setattr(metrics, f'recall_at_{k}', recall)
            
            # MRR
            metrics.mrr = calculate_mrr(retrieved_chunk_ids, relevant_chunk_ids)
            
            # NDCG@K
            metrics.ndcg_at_5 = calculate_ndcg(retrieved_chunk_ids, relevant_chunk_ids, 5)
            metrics.ndcg_at_10 = calculate_ndcg(retrieved_chunk_ids, relevant_chunk_ids, 10)
            
            # MAP
            metrics.map_score = calculate_map(retrieved_chunk_ids, relevant_chunk_ids)
        
        # ===== GENERATION METRICS (require ground truth answer) =====
        
        if ground_truth_answer:
            # BLEU
            metrics.bleu_score = calculate_bleu(ground_truth_answer, answer)
            
            # ROUGE
            rouge_scores = calculate_rouge(ground_truth_answer, answer)
            metrics.rouge1 = rouge_scores.get('rouge1', 0.0)
            metrics.rouge2 = rouge_scores.get('rouge2', 0.0)
            metrics.rougeL = rouge_scores.get('rougeL', 0.0)
            
            # BERTScore (expensive, optional)
            try:
                bertscore = calculate_bertscore(ground_truth_answer, answer)
                metrics.bertscore_precision = bertscore.get('precision', 0.0)
                metrics.bertscore_recall = bertscore.get('recall', 0.0)
                metrics.bertscore_f1 = bertscore.get('f1', 0.0)
            except Exception as e:
                log.warning(f"BERTScore calculation failed: {e}")
            
            # Simple token overlap F1
            pred_tokens = set(answer.lower().split())
            truth_tokens = set(ground_truth_answer.lower().split())
            common = pred_tokens & truth_tokens
            precision = len(common) / len(pred_tokens) if pred_tokens else 0
            recall = len(common) / len(truth_tokens) if truth_tokens else 0
            metrics.token_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Exact match
            metrics.exact_match = 1.0 if answer.strip().lower() == ground_truth_answer.strip().lower() else 0.0
        
        # ===== CITATION METRICS (require ground truth citations) =====
        
        if ground_truth_citations:
            cit_precision, cit_recall, cit_f1 = calculate_citation_metrics(
                predicted_citations,
                ground_truth_citations
            )
            metrics.citation_precision = cit_precision
            metrics.citation_recall = cit_recall
            metrics.citation_f1 = cit_f1
        
        return metrics


# Global instance
_calculator = UnifiedMetricsCalculator()


def calculate_comprehensive_metrics(
    query: str,
    answer: str,
    retrieved_chunks: List,
    retrieved_chunk_ids: List[str],
    context_texts: List[str],
    predicted_citations: Set[str],
    latency_ms: float,
    ground_truth_answer: Optional[str] = None,
    ground_truth_citations: Optional[Set[str]] = None,
    relevant_chunk_ids: Optional[Set[str]] = None
) -> Dict:
    """
    Convenience function to calculate all metrics
    Returns dictionary of metrics
    """
    metrics = _calculator.calculate_all_metrics(
        query=query,
        answer=answer,
        retrieved_chunks=retrieved_chunks,
        retrieved_chunk_ids=retrieved_chunk_ids,
        context_texts=context_texts,
        predicted_citations=predicted_citations,
        latency_ms=latency_ms,
        ground_truth_answer=ground_truth_answer,
        ground_truth_citations=ground_truth_citations,
        relevant_chunk_ids=relevant_chunk_ids
    )
    return metrics.to_dict()

# Made with Bob
