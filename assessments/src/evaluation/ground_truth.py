"""
Ground truth test queries and expected answers for evaluation
"""

from typing import List, Set, Dict, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path

from src.utils import log


@dataclass
class TestQuery:
    """Represents a test query with ground truth"""
    query_id: str
    query: str
    query_type: str  # 'fact', 'abstract', 'reasoning', 'comparative'
    expected_answer: str
    relevant_chunk_ids: Set[str] = field(default_factory=set)
    expected_citations: Set[str] = field(default_factory=set)
    difficulty: str = "medium"  # 'easy', 'medium', 'hard'
    category: str = "general"
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'query_id': self.query_id,
            'query': self.query,
            'query_type': self.query_type,
            'expected_answer': self.expected_answer,
            'relevant_chunk_ids': list(self.relevant_chunk_ids),
            'expected_citations': list(self.expected_citations),
            'difficulty': self.difficulty,
            'category': self.category
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'TestQuery':
        """Create from dictionary"""
        return cls(
            query_id=data['query_id'],
            query=data['query'],
            query_type=data['query_type'],
            expected_answer=data['expected_answer'],
            relevant_chunk_ids=set(data.get('relevant_chunk_ids', [])),
            expected_citations=set(data.get('expected_citations', [])),
            difficulty=data.get('difficulty', 'medium'),
            category=data.get('category', 'general')
        )


class GroundTruthManager:
    """Manages ground truth test queries"""
    
    def __init__(self, filepath: Optional[str] = None):
        """
        Initialize ground truth manager
        
        Args:
            filepath: Path to ground truth JSON file
        """
        self.filepath = Path(filepath) if filepath else Path('data/ground_truth.json')
        self.test_queries: List[TestQuery] = []
        
        if self.filepath.exists():
            self.load()
        else:
            log.info("No ground truth file found, initializing with default queries")
            self._initialize_default_queries()
    
    def _initialize_default_queries(self):
        """Initialize with default EU AI Act test queries"""
        
        # Fact-based queries
        self.test_queries.extend([
            TestQuery(
                query_id="fact_001",
                query="What is the definition of an AI system according to the EU AI Act?",
                query_type="fact",
                expected_answer="An AI system is a machine-based system designed to operate with varying levels of autonomy and that may exhibit adaptiveness after deployment and that, for explicit or implicit objectives, infers, from the input it receives, how to generate outputs such as predictions, content, recommendations, or decisions that can influence physical or virtual environments.",
                expected_citations={"Article 3(1)"},
                difficulty="easy",
                category="definitions"
            ),
            TestQuery(
                query_id="fact_002",
                query="What are the prohibited AI practices under the EU AI Act?",
                query_type="fact",
                expected_answer="Prohibited AI practices include: (a) deploying subliminal techniques to materially distort behavior; (b) exploiting vulnerabilities of specific groups; (c) social scoring by public authorities; (d) real-time remote biometric identification in publicly accessible spaces for law enforcement (with exceptions).",
                expected_citations={"Article 5"},
                difficulty="medium",
                category="prohibitions"
            ),
            TestQuery(
                query_id="fact_003",
                query="Which AI systems are classified as high-risk?",
                query_type="fact",
                expected_answer="High-risk AI systems include those listed in Annex III, such as: biometric identification systems, critical infrastructure management, educational and vocational training systems, employment and worker management, access to essential services, law enforcement, migration and border control, and administration of justice.",
                expected_citations={"Article 6", "Annex III"},
                difficulty="medium",
                category="risk_classification"
            ),
            TestQuery(
                query_id="fact_004",
                query="What are the transparency obligations for AI systems?",
                query_type="fact",
                expected_answer="Transparency obligations include: informing users when interacting with AI systems, labeling AI-generated content, ensuring AI systems that generate or manipulate image, audio or video content are marked as artificially generated, and providing clear information about the capabilities and limitations of the system.",
                expected_citations={"Article 50", "Article 52"},
                difficulty="medium",
                category="transparency"
            ),
            TestQuery(
                query_id="fact_005",
                query="What is the role of the AI Office?",
                query_type="fact",
                expected_answer="The AI Office is established within the European Commission to ensure the effective implementation and enforcement of the EU AI Act, coordinate with national authorities, provide guidance, monitor general-purpose AI models, and support the development of standards and best practices.",
                expected_citations={"Article 64"},
                difficulty="easy",
                category="governance"
            )
        ])
        
        # Abstract/conceptual queries
        self.test_queries.extend([
            TestQuery(
                query_id="abstract_001",
                query="Explain the risk-based approach of the EU AI Act",
                query_type="abstract",
                expected_answer="The EU AI Act adopts a risk-based approach that categorizes AI systems into different risk levels: unacceptable risk (prohibited), high-risk (strict requirements), limited risk (transparency obligations), and minimal risk (no specific obligations). The level of regulatory requirements increases with the level of risk posed by the AI system to fundamental rights and safety.",
                expected_citations={"Recital 13", "Article 5", "Article 6"},
                difficulty="medium",
                category="principles"
            ),
            TestQuery(
                query_id="abstract_002",
                query="What is the purpose of conformity assessments for high-risk AI systems?",
                query_type="abstract",
                expected_answer="Conformity assessments ensure that high-risk AI systems comply with the requirements set out in the regulation before being placed on the market. They verify that the system meets safety, transparency, accuracy, robustness, and cybersecurity standards, thereby protecting fundamental rights and ensuring trustworthy AI deployment.",
                expected_citations={"Article 43", "Recital 71"},
                difficulty="medium",
                category="compliance"
            ),
            TestQuery(
                query_id="abstract_003",
                query="How does the EU AI Act address fundamental rights?",
                query_type="abstract",
                expected_answer="The EU AI Act protects fundamental rights by prohibiting AI practices that violate human dignity, freedom, equality, and other Charter rights. It requires fundamental rights impact assessments for high-risk systems, ensures human oversight, mandates transparency, and establishes accountability mechanisms to prevent discrimination and ensure fairness.",
                expected_citations={"Recital 27", "Article 9", "Article 29"},
                difficulty="hard",
                category="fundamental_rights"
            ),
            TestQuery(
                query_id="abstract_004",
                query="Explain the concept of human oversight in AI systems",
                query_type="abstract",
                expected_answer="Human oversight means that high-risk AI systems are designed to allow natural persons to understand the system's capabilities and limitations, monitor its operation, interpret outputs, and intervene or interrupt the system when necessary. It ensures humans remain in control and can prevent or minimize risks to health, safety, and fundamental rights.",
                expected_citations={"Article 14", "Recital 48"},
                difficulty="medium",
                category="human_oversight"
            ),
            TestQuery(
                query_id="abstract_005",
                query="What is the significance of the CE marking for AI systems?",
                query_type="abstract",
                expected_answer="The CE marking indicates that a high-risk AI system has undergone conformity assessment and complies with all applicable EU regulations. It allows the system to be freely marketed across the EU single market, demonstrates the provider's commitment to safety and quality, and enables market surveillance authorities to verify compliance.",
                expected_citations={"Article 49", "Recital 78"},
                difficulty="easy",
                category="market_access"
            )
        ])
        
        # Reasoning queries
        self.test_queries.extend([
            TestQuery(
                query_id="reasoning_001",
                query="What are the implications of classifying an AI system as high-risk?",
                query_type="reasoning",
                expected_answer="Classifying an AI system as high-risk triggers extensive compliance obligations including: risk management systems, data governance requirements, technical documentation, record-keeping, transparency provisions, human oversight measures, accuracy and robustness standards, cybersecurity measures, and conformity assessments. This increases development costs and time-to-market but ensures safety and trustworthiness.",
                expected_citations={"Article 6", "Article 8-15", "Article 43"},
                difficulty="hard",
                category="compliance_impact"
            ),
            TestQuery(
                query_id="reasoning_002",
                query="How might the prohibition of social scoring affect public authorities?",
                query_type="reasoning",
                expected_answer="The prohibition prevents public authorities from using AI to evaluate or classify individuals based on social behavior or personal characteristics, which could lead to discrimination or unfair treatment. This protects citizens' fundamental rights but may limit authorities' ability to use AI for certain administrative or social policy purposes, requiring alternative approaches to public service delivery.",
                expected_citations={"Article 5(1)(c)", "Recital 17"},
                difficulty="hard",
                category="policy_impact"
            ),
            TestQuery(
                query_id="reasoning_003",
                query="What challenges might SMEs face in complying with the EU AI Act?",
                query_type="reasoning",
                expected_answer="SMEs may face challenges including: limited resources for compliance activities, difficulty understanding complex regulatory requirements, costs of conformity assessments and technical documentation, lack of in-house expertise for risk management and data governance, and potential barriers to market entry. The Act provides support measures like regulatory sandboxes and reduced fees to address these challenges.",
                expected_citations={"Article 53", "Article 57", "Recital 89"},
                difficulty="hard",
                category="business_impact"
            ),
            TestQuery(
                query_id="reasoning_004",
                query="How does the EU AI Act balance innovation with safety?",
                query_type="reasoning",
                expected_answer="The Act balances innovation and safety through: a proportionate risk-based approach that only regulates high-risk systems, regulatory sandboxes for testing innovative AI, support for SMEs and startups, harmonized rules that enable single market access, and flexibility for future technological developments. This framework aims to foster trustworthy AI innovation while protecting fundamental rights.",
                expected_citations={"Recital 4", "Article 53", "Article 57"},
                difficulty="hard",
                category="policy_balance"
            ),
            TestQuery(
                query_id="reasoning_005",
                query="What are the consequences of non-compliance with the EU AI Act?",
                query_type="reasoning",
                expected_answer="Non-compliance can result in: administrative fines up to €35 million or 7% of global annual turnover for prohibited AI practices, up to €15 million or 3% for other violations, market withdrawal or recall of non-compliant systems, reputational damage, loss of market access, and potential civil liability for damages. The severity depends on the nature and gravity of the infringement.",
                expected_citations={"Article 71", "Article 99"},
                difficulty="medium",
                category="enforcement"
            )
        ])
        
        # Comparative queries
        self.test_queries.extend([
            TestQuery(
                query_id="comparative_001",
                query="What is the difference between high-risk and limited-risk AI systems?",
                query_type="comparative",
                expected_answer="High-risk AI systems pose significant risks to health, safety, or fundamental rights and must comply with strict requirements including conformity assessments, risk management, and human oversight. Limited-risk AI systems (like chatbots) pose minimal risks and only need to meet transparency obligations, such as informing users they are interacting with AI. The regulatory burden is much lighter for limited-risk systems.",
                expected_citations={"Article 6", "Article 52", "Recital 13"},
                difficulty="medium",
                category="risk_comparison"
            ),
            TestQuery(
                query_id="comparative_002",
                query="How do obligations differ between AI providers and deployers?",
                query_type="comparative",
                expected_answer="Providers must ensure AI systems comply with requirements before market placement, conduct conformity assessments, maintain technical documentation, and implement quality management. Deployers must use systems according to instructions, ensure human oversight, monitor operation, and report serious incidents. Providers have primary compliance responsibility, while deployers focus on proper use and monitoring.",
                expected_citations={"Article 16", "Article 26", "Article 29"},
                difficulty="medium",
                category="role_comparison"
            ),
            TestQuery(
                query_id="comparative_003",
                query="Compare the EU AI Act's approach to AI regulation with a principles-based approach",
                query_type="comparative",
                expected_answer="The EU AI Act uses a rules-based approach with specific requirements, classifications, and obligations, providing legal certainty and clear compliance pathways. A principles-based approach would offer general guidelines and flexibility but less certainty. The Act combines both: specific rules for high-risk systems while allowing flexibility through innovation-friendly measures like sandboxes and codes of conduct.",
                expected_citations={"Recital 8", "Article 53", "Article 69"},
                difficulty="hard",
                category="regulatory_approach"
            ),
            TestQuery(
                query_id="comparative_004",
                query="What is the relationship between the EU AI Act and GDPR?",
                query_type="comparative",
                expected_answer="The EU AI Act and GDPR are complementary. GDPR protects personal data and privacy, while the AI Act addresses AI-specific risks to safety and fundamental rights. AI systems processing personal data must comply with both regulations. The AI Act references GDPR requirements and ensures consistency, particularly regarding data governance, transparency, and individual rights.",
                expected_citations={"Recital 41", "Article 10", "Article 29"},
                difficulty="hard",
                category="legal_framework"
            ),
            TestQuery(
                query_id="comparative_005",
                query="How do requirements differ between general-purpose AI and specific-purpose AI systems?",
                query_type="comparative",
                expected_answer="General-purpose AI models (like foundation models) have specific obligations regarding transparency, documentation, and copyright compliance. When integrated into high-risk systems, additional requirements apply. Specific-purpose AI systems are classified by their intended use and risk level, with requirements tailored to their specific application. General-purpose AI has broader, model-level obligations, while specific-purpose AI has use-case-specific requirements.",
                expected_citations={"Article 51", "Article 53", "Annex III"},
                difficulty="hard",
                category="ai_types"
            )
        ])
        
        log.info(f"Initialized {len(self.test_queries)} default test queries")
    
    def add_query(self, query: TestQuery):
        """Add a test query"""
        self.test_queries.append(query)
        log.debug(f"Added test query: {query.query_id}")
    
    def get_query(self, query_id: str) -> Optional[TestQuery]:
        """Get query by ID"""
        for query in self.test_queries:
            if query.query_id == query_id:
                return query
        return None
    
    def get_queries_by_type(self, query_type: str) -> List[TestQuery]:
        """Get all queries of a specific type"""
        return [q for q in self.test_queries if q.query_type == query_type]
    
    def get_queries_by_difficulty(self, difficulty: str) -> List[TestQuery]:
        """Get all queries of a specific difficulty"""
        return [q for q in self.test_queries if q.difficulty == difficulty]
    
    def get_queries_by_category(self, category: str) -> List[TestQuery]:
        """Get all queries in a specific category"""
        return [q for q in self.test_queries if q.category == category]
    
    def save(self, filepath: Optional[str] = None):
        """
        Save ground truth to JSON file
        
        Args:
            filepath: Path to save file (uses self.filepath if None)
        """
        save_path = Path(filepath) if filepath else self.filepath
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'test_queries': [q.to_dict() for q in self.test_queries],
            'metadata': {
                'total_queries': len(self.test_queries),
                'by_type': self._count_by_field('query_type'),
                'by_difficulty': self._count_by_field('difficulty'),
                'by_category': self._count_by_field('category')
            }
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        log.info(f"Saved {len(self.test_queries)} test queries to: {save_path}")
    
    def load(self, filepath: Optional[str] = None):
        """
        Load ground truth from JSON file
        
        Args:
            filepath: Path to load file (uses self.filepath if None)
        """
        load_path = Path(filepath) if filepath else self.filepath
        
        if not load_path.exists():
            log.warning(f"Ground truth file not found: {load_path}")
            return
        
        with open(load_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.test_queries = [
            TestQuery.from_dict(q) for q in data['test_queries']
        ]
        
        log.info(f"Loaded {len(self.test_queries)} test queries from: {load_path}")
    
    def _count_by_field(self, field: str) -> Dict[str, int]:
        """Count queries by a specific field"""
        counts = {}
        for query in self.test_queries:
            value = getattr(query, field)
            counts[value] = counts.get(value, 0) + 1
        return counts
    
    def get_statistics(self) -> Dict:
        """Get statistics about test queries"""
        return {
            'total_queries': len(self.test_queries),
            'by_type': self._count_by_field('query_type'),
            'by_difficulty': self._count_by_field('difficulty'),
            'by_category': self._count_by_field('category'),
            'avg_citations_per_query': sum(
                len(q.expected_citations) for q in self.test_queries
            ) / len(self.test_queries) if self.test_queries else 0
        }


