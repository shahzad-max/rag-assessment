#!/usr/bin/env python3
"""Live RAG System Evaluation - Generates Excel/CSV reports with metrics"""

import sys
import os
import pickle
import time
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.embeddings.multi_provider_generator import MultiProviderEmbeddingGenerator
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.prompt_manager import PromptManager
from src.generation.multi_provider_llm import MultiProviderLLM
from src.evaluation.unified_metrics import calculate_comprehensive_metrics
from src.utils import log


class LiveEvaluator:
    """Live evaluation with comprehensive reporting"""
    
    def __init__(self):
        log.info("Initializing Live Evaluator...")
        
        # Load all parameters from environment
        self.top_k = int(os.getenv('FINAL_TOP_K', '5'))
        self.dense_k = int(os.getenv('DENSE_TOP_K', '20'))
        self.sparse_k = int(os.getenv('SPARSE_TOP_K', '20'))
        self.dense_weight = float(os.getenv('DENSE_WEIGHT', '0.5'))
        self.sparse_weight = float(os.getenv('SPARSE_WEIGHT', '0.5'))
        self.bm25_k1 = float(os.getenv('BM25_K1', '1.5'))
        self.bm25_b = float(os.getenv('BM25_B', '0.75'))
        self.fusion_method = os.getenv('FUSION_METHOD', 'rrf')
        self.rrf_k = int(os.getenv('RRF_K', '60'))
        self.alpha = float(os.getenv('ALPHA', '0.5'))
        self.llm_temperature = float(os.getenv('LLM_TEMPERATURE', '0.0'))
        self.llm_max_tokens = int(os.getenv('LLM_MAX_TOKENS', '1500'))
        self.context_preview_length = int(os.getenv('CONTEXT_PREVIEW_LENGTH', '200'))
        self.question_preview_length = int(os.getenv('QUESTION_PREVIEW_LENGTH', '50'))
        self.report_output_dir = os.getenv('REPORT_OUTPUT_DIR', 'results')
        self.report_format = os.getenv('REPORT_FORMAT', 'excel')
        
        # Load data from configured paths
        index_dir = os.getenv('INDEX_DIR', 'data/indexes')
        with open(f"{index_dir}/chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)
        
        import faiss
        self.index = faiss.read_index(f"{index_dir}/faiss_index.bin")
        
        # Initialize components
        self.embedding_generator = MultiProviderEmbeddingGenerator()
        self.dense_retriever = DenseRetriever(self.index, self.chunks, self.embedding_generator)
        self.sparse_retriever = SparseRetriever(self.chunks, k1=self.bm25_k1, b=self.bm25_b)
        self.hybrid_retriever = HybridRetriever(
            self.dense_retriever,
            self.sparse_retriever,
            fusion_method=self.fusion_method,  # type: ignore
            rrf_k=self.rrf_k,
            alpha=self.alpha
        )
        self.prompt_manager = PromptManager()
        self.llm_client = MultiProviderLLM()
        
        provider_info = self.llm_client.get_provider_info()
        log.info(f"✓ LLM: {provider_info['provider']} ({provider_info['model']})")
        log.info(f"✓ Loaded {len(self.chunks)} chunks, {self.index.ntotal} vectors")
        
        self.results = []
    
    def process_query(
        self,
        question: str,
        ground_truth: Optional[str] = None,
        ground_truth_citations: Optional[Set[str]] = None,
        relevant_chunk_ids: Optional[Set[str]] = None
    ) -> Dict[str, Any]:
        """
        Process a query and calculate comprehensive metrics
        
        Args:
            question: User question
            ground_truth: Optional ground truth answer
            ground_truth_citations: Optional ground truth citations
            relevant_chunk_ids: Optional relevant chunk IDs for retrieval metrics
        """
        log.info(f"\nProcessing: {question}")
        start_time = time.time()
        
        # Retrieve context
        retrieved_results = self.hybrid_retriever.retrieve(
            query=question,
            top_k=self.top_k,
            dense_k=self.dense_k,
            sparse_k=self.sparse_k,
            dense_weight=self.dense_weight,
            sparse_weight=self.sparse_weight
        )
        
        # Build prompt
        prompt = self.prompt_manager.get_prompt(
            query=question,
            context=[chunk.text for chunk in retrieved_results]
        )
        
        # Generate answer
        try:
            answer = self.llm_client.generate(
                prompt=prompt,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens
            )
        except Exception as e:
            log.error(f"Error generating answer: {e}")
            answer = "Error generating answer."
        
        latency = (time.time() - start_time) * 1000
        
        # Extract context
        context_chunks = []
        
        for i, chunk in enumerate(retrieved_results, 1):
            preview_len = self.context_preview_length
            context_chunks.append({
                'rank': i,
                'chunk_id': chunk.chunk_id,
                'score': chunk.score,
                'text': chunk.text[:preview_len] + "..." if len(chunk.text) > preview_len else chunk.text,
                'page': chunk.metadata.get('page_number', 'N/A'),
                'section': chunk.metadata.get('section', 'N/A')
            })
        
        # Extract predicted citations
        predicted_citations = set()
        import re
        citation_pattern = r'\[(Article \d+[a-z]?|Recital \d+|Annex [IVX]+)\]'
        for match in re.finditer(citation_pattern, answer):
            predicted_citations.add(match.group(1))
        
        # Get chunk IDs and context texts
        retrieved_chunk_ids = [chunk.chunk_id for chunk in retrieved_results]
        context_texts = [chunk.text for chunk in retrieved_results]
        
        # Calculate ALL comprehensive metrics using unified calculator
        metrics_dict = calculate_comprehensive_metrics(
            query=question,
            answer=answer,
            retrieved_chunks=retrieved_results,
            retrieved_chunk_ids=retrieved_chunk_ids,
            context_texts=context_texts,
            predicted_citations=predicted_citations,
            latency_ms=latency,
            ground_truth_answer=ground_truth,
            ground_truth_citations=ground_truth_citations,
            relevant_chunk_ids=relevant_chunk_ids
        )
        
        query_result = {
            'question': question,
            'answer': answer,
            'ground_truth': ground_truth,
            'context': context_chunks,
            'metrics': metrics_dict,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results.append(query_result)
        
        log.info(f"✓ Processed in {latency:.2f}ms")
        if ground_truth:
            log.info(f"  Token F1: {metrics_dict.get('token_f1', 0):.4f}")
            log.info(f"  BLEU: {metrics_dict.get('bleu_score', 0):.4f}")
            log.info(f"  ROUGE-L: {metrics_dict.get('rougeL', 0):.4f}")
        if relevant_chunk_ids:
            log.info(f"  Precision@5: {metrics_dict.get('precision_at_5', 0):.4f}")
            log.info(f"  Recall@5: {metrics_dict.get('recall_at_5', 0):.4f}")
            log.info(f"  NDCG@5: {metrics_dict.get('ndcg_at_5', 0):.4f}")
        log.info(f"  Avg Retrieval Score: {metrics_dict.get('avg_retrieval_score', 0):.4f}")
        log.info(f"  Answer Relevance: {metrics_dict.get('answer_relevance', 0):.4f}")
        
        return query_result
    
    def generate_report(self, output_format: str = 'excel') -> str:
        """
        Generate comprehensive report in Excel or CSV format
        
        Args:
            output_format: 'excel' or 'csv'
            
        Returns:
            Path to generated report file
        """
        if not self.results:
            log.warning("No results to generate report")
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Prepare main results dataframe
        main_data = []
        for i, result in enumerate(self.results, 1):
            row = {
                'Query_ID': i,
                'Question': result['question'],
                'Answer': result['answer'],
                'Ground_Truth': result.get('ground_truth', 'N/A'),
                'Timestamp': result['timestamp']
            }
            
            # Add metrics
            for metric_name, metric_value in result['metrics'].items():
                row[metric_name] = metric_value
            
            main_data.append(row)
        
        df_main = pd.DataFrame(main_data)
        
        # Prepare context details dataframe
        context_data = []
        for i, result in enumerate(self.results, 1):
            q_preview_len = self.question_preview_length
            for chunk in result['context']:
                context_data.append({
                    'Query_ID': i,
                    'Question': result['question'][:q_preview_len] + '...' if len(result['question']) > q_preview_len else result['question'],
                    'Rank': chunk['rank'],
                    'Chunk_ID': chunk['chunk_id'],
                    'Score': chunk['score'],
                    'Page': chunk['page'],
                    'Section': chunk['section'],
                    'Text_Preview': chunk['text']
                })
        
        df_context = pd.DataFrame(context_data)
        
        # Calculate summary statistics
        summary_data = {
            'Metric': [],
            'Value': []
        }
        
        summary_data['Metric'].append('Total Queries')
        summary_data['Value'].append(len(self.results))
        
        summary_data['Metric'].append('Avg Latency (ms)')
        summary_data['Value'].append(df_main['latency_ms'].mean())
        
        summary_data['Metric'].append('Avg Answer Length (chars)')
        summary_data['Value'].append(df_main['answer_length'].mean())
        
        summary_data['Metric'].append('Avg Answer Length (words)')
        summary_data['Value'].append(df_main['answer_words'].mean())
        
        summary_data['Metric'].append('Avg Retrieval Score')
        summary_data['Value'].append(df_main['avg_retrieval_score'].mean())
        
        summary_data['Metric'].append('Avg Citations per Answer')
        summary_data['Value'].append(df_main['num_citations'].mean())
        
        # Add evaluation metrics if ground truth exists
        if 'f1_score' in df_main.columns:
            summary_data['Metric'].append('Avg F1 Score')
            summary_data['Value'].append(df_main['f1_score'].mean())
            
            summary_data['Metric'].append('Avg BLEU Score')
            summary_data['Value'].append(df_main['bleu_score'].mean())
            
            summary_data['Metric'].append('Exact Match Rate')
            summary_data['Value'].append(df_main['exact_match'].mean())
        
        df_summary = pd.DataFrame(summary_data)
        
        # Save to file
        if output_format == 'excel':
            output_file = f"{self.report_output_dir}/live_evaluation_{timestamp}.xlsx"
            
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                df_main.to_excel(writer, sheet_name='Results', index=False)
                df_context.to_excel(writer, sheet_name='Context_Details', index=False)
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            log.info(f"✓ Excel report generated: {output_file}")
        
        else:  # CSV
            output_file = f"{self.report_output_dir}/live_evaluation_{timestamp}.csv"
            df_main.to_csv(output_file, index=False)
            
            context_file = f"{self.report_output_dir}/live_evaluation_context_{timestamp}.csv"
            df_context.to_csv(context_file, index=False)
            
            summary_file = f"{self.report_output_dir}/live_evaluation_summary_{timestamp}.csv"
            df_summary.to_csv(summary_file, index=False)
            
            log.info(f"✓ CSV reports generated:")
            log.info(f"  - Main: {output_file}")
            log.info(f"  - Context: {context_file}")
            log.info(f"  - Summary: {summary_file}")
        
        return output_file
    
    def generate_json_report(self) -> str:
        """
        Generate JSON report with all results and metrics
        
        Returns:
            Path to generated JSON file
        """
        if not self.results:
            log.warning("No results to generate JSON report")
            return ""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{self.report_output_dir}/live_evaluation_{timestamp}.json"
        
        # Save complete results as JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        log.info(f"✓ JSON report generated: {output_file}")
        return output_file
    
    def print_summary(self):
        """Print summary statistics to console"""
        if not self.results:
            print("No results to summarize")
            return
        
        print("\n" + "="*80)
        print("LIVE EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\nTotal Queries Processed: {len(self.results)}")
        
        latencies = [r['metrics']['latency_ms'] for r in self.results]
        print(f"\nLatency Statistics:")
        print(f"  Average: {sum(latencies)/len(latencies):.2f}ms")
        print(f"  Min: {min(latencies):.2f}ms")
        print(f"  Max: {max(latencies):.2f}ms")
        
        avg_scores = [r['metrics']['avg_retrieval_score'] for r in self.results]
        print(f"\nRetrieval Score Statistics:")
        print(f"  Average: {sum(avg_scores)/len(avg_scores):.4f}")
        print(f"  Min: {min(avg_scores):.4f}")
        print(f"  Max: {max(avg_scores):.4f}")
        
        answer_lengths = [r['metrics']['answer_words'] for r in self.results]
        print(f"\nAnswer Length Statistics (words):")
        print(f"  Average: {sum(answer_lengths)/len(answer_lengths):.1f}")
        print(f"  Min: {min(answer_lengths)}")
        print(f"  Max: {max(answer_lengths)}")
        
        # If ground truth exists
        if 'f1_score' in self.results[0]['metrics']:
            f1_scores = [r['metrics']['f1_score'] for r in self.results]
            print(f"\nF1 Score Statistics:")
            print(f"  Average: {sum(f1_scores)/len(f1_scores):.4f}")
            print(f"  Min: {min(f1_scores):.4f}")
            print(f"  Max: {max(f1_scores):.4f}")
        
        print("\n" + "="*80)


def run_ground_truth_evaluation():
    """Run evaluation on all 20 ground truth questions"""
    print("="*80)
    print("RUNNING FULL EVALUATION ON 20 GROUND TRUTH QUESTIONS")
    print("="*80)
    print("\nInitializing system...")
    
    evaluator = LiveEvaluator()
    
    # Load ground truth queries
    from src.evaluation.ground_truth import GroundTruthManager
    gt_manager = GroundTruthManager()
    
    print(f"\n✓ Loaded {len(gt_manager.test_queries)} ground truth queries")
    print("\nProcessing queries with comprehensive metrics...")
    print("="*80)
    
    # Process each query with ground truth
    for i, test_query in enumerate(gt_manager.test_queries, 1):
        print(f"\n[{i}/{len(gt_manager.test_queries)}] {test_query.query}")
        
        evaluator.process_query(
            question=test_query.query,
            ground_truth=test_query.expected_answer,
            ground_truth_citations=test_query.expected_citations,
            relevant_chunk_ids=test_query.relevant_chunk_ids
        )
    
    print("\n" + "="*80)
    print("GENERATING REPORTS")
    print("="*80)
    
    # Generate Excel report
    print("\nGenerating Excel report...")
    excel_file = evaluator.generate_report(output_format='excel')
    
    # Generate JSON report
    print("Generating JSON report...")
    json_file = evaluator.generate_json_report()
    
    # Print summary
    evaluator.print_summary()
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\n✓ Excel Report: {excel_file}")
    print(f"✓ JSON Report: {json_file}")
    print(f"\n✓ Processed {len(gt_manager.test_queries)} queries with comprehensive metrics")
    print("✓ All metrics exported to Excel and JSON")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Live RAG System Evaluation')
    parser.add_argument('--full', action='store_true', help='Run full evaluation on 20 ground truth questions')
    parser.add_argument('questions', nargs='*', help='Questions to process')
    
    args = parser.parse_args()
    
    if args.full:
        # Run full ground truth evaluation
        run_ground_truth_evaluation()
    
    elif args.questions:
        # Process provided questions
        print("="*80)
        print("LIVE RAG SYSTEM EVALUATION")
        print("="*80)
        print("\nInitializing system...")
        
        evaluator = LiveEvaluator()
        
        print(f"\nProcessing {len(args.questions)} questions...")
        
        for question in args.questions:
            evaluator.process_query(question)
        
        # Generate reports
        print("\nGenerating reports...")
        excel_file = evaluator.generate_report(output_format='excel')
        json_file = evaluator.generate_json_report()
        
        # Print summary
        evaluator.print_summary()
        
        print(f"\n✓ Excel Report: {excel_file}")
        print(f"✓ JSON Report: {json_file}")
        print("✓ Evaluation complete!")
    
    else:
        print("="*80)
        print("LIVE RAG SYSTEM EVALUATION")
        print("="*80)
        print("\nUsage:")
        print("  1. Run full evaluation (20 ground truth questions):")
        print("     python scripts/live_evaluation.py --full")
        print("\n  2. Process custom questions:")
        print('     python scripts/live_evaluation.py "Question 1" "Question 2" ...')
        print("\n  3. Use programmatically:")
        print("     from scripts.live_evaluation import LiveEvaluator")
        print("     evaluator = LiveEvaluator()")
        print('     evaluator.process_query("Your question")')
        print("     evaluator.generate_report()")
        print("     evaluator.generate_json_report()")


if __name__ == "__main__":
    main()

