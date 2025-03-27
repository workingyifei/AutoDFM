import os
import yaml
from typing import Dict, List, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings

class KnowledgeBase:
    def __init__(self, base_path: str = None):
        if base_path is None:
            base_path = Path(__file__).parent
        
        self.rules_path = Path(base_path) / "dfm_rules"
        self.supplier_path = Path(base_path) / "supplier_profiles"
        
        # Initialize ChromaDB for vector storage
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=str(Path(base_path) / "vector_store")
        ))
        
        # Create collections for rules and suppliers
        self.rules_collection = self.chroma_client.get_or_create_collection("dfm_rules")
        self.supplier_collection = self.chroma_client.get_or_create_collection("supplier_profiles")
        
        # Load rules and supplier profiles
        self.rules = self._load_rules()
        self.supplier_profiles = self._load_supplier_profiles()
        
    def _load_rules(self) -> Dict:
        """Load all DFM rules from YAML files"""
        rules = {}
        for rule_file in self.rules_path.glob("*.yaml"):
            with open(rule_file, 'r') as f:
                rules[rule_file.stem] = yaml.safe_load(f)
        return rules
    
    def _load_supplier_profiles(self) -> Dict:
        """Load all supplier profiles from YAML files"""
        profiles = {}
        for profile_file in self.supplier_path.glob("*.yaml"):
            with open(profile_file, 'r') as f:
                profile = yaml.safe_load(f)
                profiles[profile["supplier_id"]] = profile
        return profiles
    
    def get_rule(self, rule_category: str, rule_name: str) -> Optional[Dict]:
        """Get a specific DFM rule"""
        if rule_category in self.rules:
            category_rules = self.rules[rule_category]
            return category_rules.get(rule_name)
        return None
    
    def get_supplier_profile(self, supplier_id: str) -> Optional[Dict]:
        """Get a specific supplier profile"""
        return self.supplier_profiles.get(supplier_id)
    
    def query_relevant_rules(self, pcb_features: Dict) -> List[Dict]:
        """Query relevant DFM rules based on PCB features"""
        # Convert PCB features to a query string
        query = self._features_to_query(pcb_features)
        
        # Search in vector store
        results = self.rules_collection.query(
            query_texts=[query],
            n_results=5
        )
        
        return self._process_rule_results(results)
    
    def find_compatible_suppliers(self, pcb_requirements: Dict) -> List[Dict]:
        """Find suppliers compatible with PCB requirements"""
        compatible_suppliers = []
        
        for supplier_id, profile in self.supplier_profiles.items():
            if self._check_supplier_compatibility(profile, pcb_requirements):
                compatible_suppliers.append(profile)
        
        return compatible_suppliers
    
    def _features_to_query(self, pcb_features: Dict) -> str:
        """Convert PCB features to a search query string"""
        # Implement feature to query conversion logic
        query_parts = []
        
        if "trace_width" in pcb_features:
            query_parts.append(f"trace width {pcb_features['trace_width']}")
        if "via_diameter" in pcb_features:
            query_parts.append(f"via diameter {pcb_features['via_diameter']}")
        if "layer_count" in pcb_features:
            query_parts.append(f"{pcb_features['layer_count']} layers")
        
        return " ".join(query_parts)
    
    def _process_rule_results(self, results: Dict) -> List[Dict]:
        """Process and format the vector search results"""
        processed_results = []
        
        for idx, (doc_id, score) in enumerate(zip(results['ids'][0], results['distances'][0])):
            if score > 0.7:  # Relevance threshold
                rule_data = self._get_rule_by_id(doc_id)
                if rule_data:
                    processed_results.append(rule_data)
        
        return processed_results
    
    def _check_supplier_compatibility(self, supplier_profile: Dict, requirements: Dict) -> bool:
        """Check if a supplier's capabilities meet PCB requirements"""
        capabilities = supplier_profile["capabilities"]
        
        # Check trace requirements
        if "trace_width" in requirements:
            if requirements["trace_width"] < capabilities["trace_capabilities"]["min_width"]:
                return False
        
        # Check via requirements
        if "via_diameter" in requirements:
            if requirements["via_diameter"] < capabilities["via_capabilities"]["min_diameter"]:
                return False
        
        # Check layer count
        if "layer_count" in requirements:
            if (requirements["layer_count"] > capabilities["board_capabilities"]["max_layers"] or
                requirements["layer_count"] < capabilities["board_capabilities"]["min_layers"]):
                return False
        
        return True
    
    def _get_rule_by_id(self, rule_id: str) -> Optional[Dict]:
        """Retrieve rule data by its ID"""
        for category, rules in self.rules.items():
            if rule_id in rules:
                return {
                    "category": category,
                    "rule_id": rule_id,
                    "data": rules[rule_id]
                }
        return None 