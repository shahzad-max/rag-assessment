"""
Evaluation metrics for RAG system
"""

import numpy as np
from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass

from src.utils import log


@dataclass
class RetrievalMetrics:
    """Metrics for retrieval evaluation"""
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    mrr: float
    ndcg_at_k: Dict[int, float]
    map_score: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'precision_at_k': self.precision_at_k,
            'recall_at_k': self.recall_at_k,
            'mrr': self.mrr,
            'ndcg_at_k': self.ndcg_at_k,
            'map': self.map_score
        }


@dataclass
class GenerationMetrics:
    """Metrics for generation evaluation"""
    bleu_score: float
    rouge_scores: Dict[str, float]
    bertscore: Dict[str, float]
    citation_precision: float
    citation_recall: float
    citation_f1: float
    answer_relevance: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'bleu': self.bleu_score,
            'rouge': self.rouge_scores,
            'bertscore': self.bertscore,
            'citation_precision': self.citation_precision,
            'citation_recall': self.citation_recall,
            'citation_f1': self.citation_f1,
            'answer_relevance': self.answer_relevance
        }


def calculate_precision_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int
) -> float:
    """
    Calculate Precision@K
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered)
        relevant_ids: Set of relevant document IDs
        k: Number of top results to consider
        
    Returns:
        Precision@K score
    """
    if k <= 0 or not retrieved_ids:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    
    return relevant_retrieved / k


def calculate_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int
) -> float:
    """
    Calculate Recall@K
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered)
        relevant_ids: Set of relevant document IDs
        k: Number of top results to consider
        
    Returns:
        Recall@K score
    """
    if not relevant_ids or k <= 0:
        return 0.0
    
    top_k = retrieved_ids[:k]
    relevant_retrieved = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    
    return relevant_retrieved / len(relevant_ids)


def calculate_mrr(
    retrieved_ids: List[str],
    relevant_ids: Set[str]
) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR)
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered)
        relevant_ids: Set of relevant document IDs
        
    Returns:
        MRR score
    """
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            return 1.0 / i
    
    return 0.0


def calculate_ndcg(
    retrieved_ids: List[str],
    relevant_ids: Set[str],
    k: int,
    relevance_scores: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K)
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered)
        relevant_ids: Set of relevant document IDs
        k: Number of top results to consider
        relevance_scores: Optional relevance scores for each document
        
    Returns:
        NDCG@K score
    """
    if k <= 0 or not relevant_ids:
        return 0.0
    
    # Use binary relevance if scores not provided
    if relevance_scores is None:
        relevance_scores = {doc_id: 1.0 for doc_id in relevant_ids}
    
    # Calculate DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], 1):
        rel = relevance_scores.get(doc_id, 0.0)
        dcg += rel / np.log2(i + 1)
    
    # Calculate IDCG (ideal DCG)
    ideal_scores = sorted(
        [relevance_scores.get(doc_id, 0.0) for doc_id in relevant_ids],
        reverse=True
    )[:k]
    
    idcg = sum(rel / np.log2(i + 1) for i, rel in enumerate(ideal_scores, 1))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def calculate_map(
    retrieved_ids: List[str],
    relevant_ids: Set[str]
) -> float:
    """
    Calculate Mean Average Precision (MAP)
    
    Args:
        retrieved_ids: List of retrieved document IDs (ordered)
        relevant_ids: Set of relevant document IDs
        
    Returns:
        MAP score
    """
    if not relevant_ids:
        return 0.0
    
    relevant_count = 0
    precision_sum = 0.0
    
    for i, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_ids:
            relevant_count += 1
            precision_sum += relevant_count / i
    
    if relevant_count == 0:
        return 0.0
    
    return precision_sum / len(relevant_ids)


def calculate_bleu(
    reference: str,
    hypothesis: str,
    max_n: int = 4
) -> float:
    """
    Calculate BLEU score
    
    Args:
        reference: Reference text
        hypothesis: Generated text
        max_n: Maximum n-gram size
        
    Returns:
        BLEU score
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        # Tokenize
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        # Calculate BLEU with smoothing
        smoothing = SmoothingFunction().method1
        score = sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            smoothing_function=smoothing
        )
        
        return score
        
    except ImportError:
        log.warning("NLTK not available, returning 0.0 for BLEU")
        return 0.0


def calculate_rouge(
    reference: str,
    hypothesis: str
) -> Dict[str, float]:
    """
    Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
    
    Args:
        reference: Reference text
        hypothesis: Generated text
        
    Returns:
        Dictionary with ROUGE scores
    """
    try:
        from rouge_score import rouge_scorer
        
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
        
        scores = scorer.score(reference, hypothesis)
        
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
        
    except ImportError:
        log.warning("rouge-score not available, returning zeros")
        return {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}


def calculate_bertscore(
    reference: str,
    hypothesis: str,
    model_type: str = 'microsoft/deberta-xlarge-mnli'
) -> Dict[str, float]:
    """
    Calculate BERTScore
    
    Args:
        reference: Reference text
        hypothesis: Generated text
        model_type: Model to use for BERTScore
        
    Returns:
        Dictionary with precision, recall, F1
    """
    try:
        from bert_score import score
        
        P, R, F1 = score(
            [hypothesis],
            [reference],
            model_type=model_type,
            verbose=False
        )
        
        return {
            'precision': P.item(),
            'recall': R.item(),
            'f1': F1.item()
        }
        
    except ImportError:
        log.warning("bert-score not available, returning zeros")
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}


def calculate_citation_metrics(
    predicted_citations: Set[str],
    ground_truth_citations: Set[str]
) -> Tuple[float, float, float]:
    """
    Calculate citation precision, recall, and F1
    
    Args:
        predicted_citations: Set of predicted citation strings
        ground_truth_citations: Set of ground truth citation strings
        
    Returns:
        Tuple of (precision, recall, f1)
    """
    if not predicted_citations and not ground_truth_citations:
        return 1.0, 1.0, 1.0
    
    if not predicted_citations:
        return 0.0, 0.0, 0.0
    
    if not ground_truth_citations:
        return 0.0, 0.0, 0.0
    
    # Calculate intersection
    correct = predicted_citations & ground_truth_citations
    
    # Precision: correct / predicted
    precision = len(correct) / len(predicted_citations)
    
    # Recall: correct / ground_truth
    recall = len(correct) / len(ground_truth_citations)
    
    # F1: harmonic mean
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1


def calculate_answer_relevance(
    answer: str,
    query: str,
    context: List[str]
) -> float:
    """
    Calculate answer relevance score (simple heuristic)
    
    Args:
        answer: Generated answer
        query: User query
        context: Retrieved context chunks
        
    Returns:
        Relevance score (0-1)
    """
    # Simple heuristic: check if answer contains query terms
    query_terms = set(query.lower().split())
    answer_terms = set(answer.lower().split())
    
    # Term overlap
    overlap = len(query_terms & answer_terms)
    term_score = overlap / len(query_terms) if query_terms else 0.0
    
    # Length check (not too short, not too long)
    answer_length = len(answer.split())
    if answer_length < 10:
        length_score = 0.5
    elif answer_length > 500:
        length_score = 0.7
    else:
        length_score = 1.0
    
    # Context grounding (answer should reference context)
    context_text = ' '.join(context).lower()
    answer_lower = answer.lower()
    
    # Check for common phrases from context
    context_phrases = set()
    for chunk in context:
        words = chunk.lower().split()
        for i in range(len(words) - 2):
            context_phrases.add(' '.join(words[i:i+3]))
    
    grounding_score = 0.0
    answer_words = answer_lower.split()
    for i in range(len(answer_words) - 2):
        phrase = ' '.join(answer_words[i:i+3])
        if phrase in context_phrases:
            grounding_score += 1
    
    grounding_score = min(grounding_score / 10, 1.0) if grounding_score > 0 else 0.5
    
    # Weighted combination
    relevance = 0.3 * term_score + 0.2 * length_score + 0.5 * grounding_score
    
    return relevance


def aggregate_metrics(
    metrics_list: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Aggregate metrics across multiple queries
    
    Args:
        metrics_list: List of metric dictionaries
        
    Returns:
        Dictionary with mean and std for each metric
    """
    if not metrics_list:
        return {}
    
    # Collect all metric names
    all_metrics = set()
    for metrics in metrics_list:
        all_metrics.update(metrics.keys())
    
    # Calculate mean and std for each metric
    aggregated = {}
    for metric_name in all_metrics:
        values = [m.get(metric_name, 0.0) for m in metrics_list]
        aggregated[f"{metric_name}_mean"] = np.mean(values)
        aggregated[f"{metric_name}_std"] = np.std(values)
        aggregated[f"{metric_name}_min"] = np.min(values)
        aggregated[f"{metric_name}_max"] = np.max(values)
    
    return aggregated


