"""
Run 20 questions without ground truth comparison
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.live_evaluation import LiveEvaluator

# 20 questions (just the queries, no ground truth)
QUESTIONS = [
    "What is the definition of an AI system according to the EU AI Act?",
    "What are the prohibited AI practices under the EU AI Act?",
    "Which AI systems are classified as high-risk?",
    "What are the transparency obligations for AI systems?",
    "What is the role of the AI Office?",
    "Explain the risk-based approach of the EU AI Act",
    "What is the purpose of conformity assessments for high-risk AI systems?",
    "How does the EU AI Act address fundamental rights?",
    "Explain the concept of human oversight in AI systems",
    "What is the significance of the CE marking for AI systems?",
    "What are the implications of classifying an AI system as high-risk?",
    "How might the prohibition of social scoring affect public authorities?",
    "What challenges might SMEs face in complying with the EU AI Act?",
    "How does the EU AI Act balance innovation with safety?",
    "What are the consequences of non-compliance with the EU AI Act?",
    "What is the difference between high-risk and limited-risk AI systems?",
    "How do obligations differ between AI providers and deployers?",
    "Compare the EU AI Act's approach to AI regulation with a principles-based approach",
    "What is the relationship between the EU AI Act and GDPR?",
    "How do requirements differ between general-purpose AI and specific-purpose AI systems?"
]

def main():
    print("=" * 80)
    print("RUNNING 20 QUESTIONS WITHOUT GROUND TRUTH")
    print("=" * 80)
    print()
    
    evaluator = LiveEvaluator()
    
    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n[{i}/20] {question}")
        evaluator.process_query(question)
    
    print("\n" + "=" * 80)
    print("GENERATING REPORTS")
    print("=" * 80)
    print()
    
    evaluator.generate_report()
    evaluator.generate_json_report()
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()

# Made with Bob
