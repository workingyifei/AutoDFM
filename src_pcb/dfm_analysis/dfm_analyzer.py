from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from ..knowledge_base.knowledge_base import KnowledgeBase
from ..pcb_processing.pcb_parser import PCBFeatures
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnableSequence
import os
import json
import re
import logging
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AnalysisMode(Enum):
    RULE_BASED = "rule-based"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    AUTO = "auto"  # Will try OpenAI -> Huggingface -> Rule-based in that order

@dataclass
class DFMViolation:
    rule_id: str
    severity: str  # 'error', 'warning', 'info'
    message: str
    location: Optional[Dict] = None
    recommendation: Optional[str] = None

@dataclass
class DFMReport:
    violations: List[DFMViolation]
    supplier_id: Optional[str]
    overall_score: float
    recommendations: List[str]
    is_manufacturable: bool
    source: str = "rule-based"  # Can be "rule-based", "openai", "huggingface"

class DFMAnalyzer:
    def __init__(self, knowledge_base: KnowledgeBase, analysis_mode: str = "auto"):
        """
        Initialize DFM Analyzer
        
        Args:
            knowledge_base: KnowledgeBase instance containing rules and supplier profiles
            analysis_mode: One of "rule-based", "openai", "huggingface", or "auto"
        """
        self.knowledge_base = knowledge_base
        self.llm = None
        self.analysis_chain = None
        self.api_source = "none"
        self.last_llm_response = None
        self.analysis_mode = AnalysisMode(analysis_mode.lower())
        
        # Load manufacturing rules from YAML
        self.manufacturing_rules = self.knowledge_base.rules
        
        if self.analysis_mode == AnalysisMode.RULE_BASED:
            logger.info("Using rule-based analysis only")
            return
            
        if self.analysis_mode in [AnalysisMode.OPENAI, AnalysisMode.AUTO]:
            if os.getenv("OPENAI_API_KEY"):
                try:
                    logger.info("Using OpenAI model for DFM analysis")
                    self.llm = ChatOpenAI(
                        model="gpt-3.5-turbo",
                        temperature=0.2,
                        openai_api_key=os.getenv("OPENAI_API_KEY"),
                    )
                    self.api_source = "openai"
                    logger.info("Successfully initialized OpenAI LLM")
                except Exception as e:
                    logger.error(f"Failed to initialize OpenAI LLM: {str(e)}")
                    self.llm = None
                    
        if (self.analysis_mode in [AnalysisMode.HUGGINGFACE, AnalysisMode.AUTO] and 
            not self.llm and (os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN"))):
            try:
                from langchain_community.llms import HuggingFaceHub
                logger.info("Using Hugging Face model for DFM analysis")
                # Use either token, preferring HUGGINGFACEHUB_API_TOKEN if available
                hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
                self.llm = HuggingFaceHub(
                    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                    huggingfacehub_api_token=hf_token,
                    model_kwargs={"temperature": 0.1, "max_length": 1024}
                )
                self.api_source = "huggingface"
                logger.info("Successfully initialized Hugging Face LLM")
            except ImportError:
                logger.error("Hugging Face packages not installed")
                self.llm = None
            except Exception as e:
                logger.error(f"Failed to initialize Hugging Face LLM: {str(e)}")
                self.llm = None
        
        if not self.llm and self.analysis_mode != AnalysisMode.RULE_BASED:
            # No API keys available or failed to initialize, fall back to rule-based
            logger.warning(f"Falling back to rule-based analysis as {self.analysis_mode.value} was not available")
            self.analysis_mode = AnalysisMode.RULE_BASED
        
        # Initialize LLM chain for DFM analysis if using LLM
        if self.llm:
            self.analysis_chain = self._setup_analysis_chain()
    
    def analyze_pcb(self, pcb_features: PCBFeatures, supplier_id: Optional[str] = None) -> DFMReport:
        """Analyze PCB features for DFM violations"""
        # Get relevant rules
        relevant_rules = self.knowledge_base.query_relevant_rules(pcb_features.__dict__)
        
        # Get supplier profile if specified
        supplier_profile = None
        if supplier_id:
            supplier_profile = self.knowledge_base.get_supplier_profile(supplier_id)
        
        # Check for violations
        violations = []
        
        # Basic rule checking
        violations.extend(self._check_trace_rules(pcb_features, relevant_rules))
        violations.extend(self._check_via_rules(pcb_features, relevant_rules))
        violations.extend(self._check_board_rules(pcb_features, relevant_rules))
        
        # LLM-based analysis
        llm_violations, llm_source = self._perform_llm_analysis(pcb_features, supplier_profile)
        violations.extend(llm_violations)
        
        # Calculate overall score
        score = self._calculate_manufacturability_score(violations)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(violations)
        
        # Determine manufacturability
        is_manufacturable = score >= 0.7 and not any(v.severity == 'error' for v in violations)
        
        return DFMReport(
            violations=violations,
            supplier_id=supplier_id,
            overall_score=score,
            recommendations=recommendations,
            is_manufacturable=is_manufacturable,
            source=llm_source
        )
    
    def _check_trace_rules(self, features: PCBFeatures, rules: List[Dict]) -> List[DFMViolation]:
        """Check trace-related DFM rules"""
        violations = []
        
        # Get trace rules
        trace_rules = next((r for r in rules if 'trace_rules' in r['data']), None)
        if not trace_rules:
            return violations
        
        rules_data = trace_rules['data']['trace_rules']
        
        # Check minimum trace width
        if features.min_trace_width < rules_data['min_width']:
            violations.append(DFMViolation(
                rule_id='trace_width',
                severity='error',
                message=f"Trace width {features.min_trace_width}mm is below minimum {rules_data['min_width']}mm",
                recommendation=f"Increase trace width to at least {rules_data['min_width']}mm"
            ))
        
        # Check minimum trace spacing
        if features.min_trace_spacing < rules_data['min_spacing']:
            violations.append(DFMViolation(
                rule_id='trace_spacing',
                severity='error',
                message=f"Trace spacing {features.min_trace_spacing}mm is below minimum {rules_data['min_spacing']}mm",
                recommendation=f"Increase trace spacing to at least {rules_data['min_spacing']}mm"
            ))
        
        return violations
    
    def _check_via_rules(self, features: PCBFeatures, rules: List[Dict]) -> List[DFMViolation]:
        """Check via-related DFM rules"""
        violations = []
        
        via_rules = next((r for r in rules if 'via_rules' in r['data']), None)
        if not via_rules:
            return violations
        
        rules_data = via_rules['data']['via_rules']
        
        # Check minimum via diameter
        if features.min_via_diameter < rules_data['min_diameter']:
            violations.append(DFMViolation(
                rule_id='via_diameter',
                severity='error',
                message=f"Via diameter {features.min_via_diameter}mm is below minimum {rules_data['min_diameter']}mm",
                recommendation=f"Increase via diameter to at least {rules_data['min_diameter']}mm"
            ))
        
        # Check minimum drill size
        if features.min_via_drill < rules_data['min_drill_size']:
            violations.append(DFMViolation(
                rule_id='via_drill',
                severity='error',
                message=f"Via drill size {features.min_via_drill}mm is below minimum {rules_data['min_drill_size']}mm",
                recommendation=f"Increase via drill size to at least {rules_data['min_drill_size']}mm"
            ))
        
        return violations
    
    def _check_board_rules(self, features: PCBFeatures, rules: List[Dict]) -> List[DFMViolation]:
        """Check board-related DFM rules"""
        violations = []
        
        board_rules = next((r for r in rules if 'board_rules' in r['data']), None)
        if not board_rules:
            return violations
        
        rules_data = board_rules['data']['board_rules']
        
        # Check layer count
        if features.layer_count < rules_data['min_layers']:
            violations.append(DFMViolation(
                rule_id='layer_count',
                severity='warning',
                message=f"Layer count {features.layer_count} is below recommended minimum {rules_data['min_layers']}",
                recommendation="Consider increasing layer count for better signal integrity"
            ))
        
        return violations
    
    def _setup_analysis_chain(self) -> RunnableSequence:
        """Set up LangChain for DFM analysis"""
        if not self.llm:
            return None

        if self.api_source == "openai":
            # Simpler and clearer template for OpenAI
            template = """
            You are a PCB DFM (Design for Manufacturing) expert. Analyze the PCB features and manufacturing rules to identify potential manufacturability issues.
            
            PCB Features:
            {pcb_features}
            
            Manufacturing Rules:
            {manufacturing_rules}
            
            Supplier Profile (if available):
            {supplier_profile}
            
            Format your response as a valid JSON array. Each item in the array must be a JSON object with exactly these three properties: "issue", "severity", and "recommendation".
            
            Example of the exact format required:
            [
                {{
                    "issue": "description of the issue",
                    "severity": "error",
                    "recommendation": "how to fix it"
                }}
            ]
            
            Rules:
            1. Use exactly these property names: "issue", "severity", "recommendation"
            2. The severity must be one of: "error", "warning", "info"
            3. Only include actual issues found in the provided PCB data
            4. Return only the JSON array with no other text
            5. Do not include any markdown formatting
            
            Analyze the PCB data now and return the results in exactly this format.
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.llm
            return chain
        else:
            # Create a more focused template for Hugging Face that avoids returning generic issues
            template = """You are a PCB DFM (Design for Manufacturing) expert tasked with analyzing a circuit board design.

Manufacturing Rules:
{manufacturing_rules}

Supplier Information:
{supplier_profile}

PCB Features:
{pcb_features}

Analyze ONLY the above PCB features against the manufacturing rules and supplier capabilities.
Focus ONLY on issues that can be identified from the specific data provided.
DO NOT list generic issues that aren't indicated by the provided data.

For each manufacturing issue you identify in the SPECIFIC provided data, format your response like this:

ISSUE: [Clear description of the specific manufacturing problem found in the provided data]
SEVERITY: [error/warning/info]
RECOMMENDATION: [Specific suggestion to fix the issue]

If no issues are found in the provided PCB features data, state "No issues found in the provided PCB data" and suggest general best practices.
"""
            
            prompt = PromptTemplate.from_template(template)
            chain = prompt | self.llm
            return chain
    
    def _perform_llm_analysis(self, features: PCBFeatures, supplier_profile: Optional[Dict] = None) -> Tuple[List[DFMViolation], str]:
        """Perform LLM-based DFM analysis on the PCB."""
        violations = []
        
        try:
            # Get the chain for analysis
            chain = self._setup_analysis_chain()
            supplier_data = ""
            if supplier_profile:
                supplier_data = str(supplier_profile)
                
            # Create inputs based on PCB features
            llm_inputs = {
                "pcb_features": str(features),
                "manufacturing_rules": self.manufacturing_rules,
                "supplier_profile": supplier_data
            }
            
            # Try OpenAI first if it's configured
            if self.api_source == "openai":
                try:
                    # For OpenAI the response is structured
                    result_text = chain.invoke(llm_inputs)
                    # Store the raw LLM response
                    self.last_llm_response = result_text
                    
                    # Try to parse OpenAI response
                    violations = self._parse_openai_response(result_text)
                    if violations:
                        return violations, "openai"
                except Exception as e:
                    logger.error(f"OpenAI analysis failed: {str(e)}")
                    # If OpenAI fails, try to initialize Hugging Face
                    if os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN"):
                        try:
                            from langchain_community.llms import HuggingFaceHub
                            logger.info("Falling back to Hugging Face model")
                            hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HUGGINGFACE_API_TOKEN")
                            self.llm = HuggingFaceHub(
                                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                                huggingfacehub_api_token=hf_token,
                                model_kwargs={"temperature": 0.1, "max_length": 1024}
                            )
                            self.api_source = "huggingface"
                            # Set up new chain with Hugging Face
                            chain = self._setup_analysis_chain()
                        except Exception as hf_e:
                            logger.error(f"Failed to initialize Hugging Face: {str(hf_e)}")
            
            # If we're using Hugging Face (either initially or as fallback)
            if self.api_source == "huggingface":
                # For Hugging Face, the response is plain text
                result_text = chain.invoke(llm_inputs)
                # Store the raw LLM response
                self.last_llm_response = result_text
                
                # Parse response using our improved parser
                try:
                    parsed_violations, recommendations, score, is_manufacturable = self._parse_text_response(result_text)
                    # Convert parsed violations to DFMViolation objects
                    for v in parsed_violations:
                        violations.append(DFMViolation(
                            rule_id=v["rule_id"],
                            severity=v["severity"],
                            message=v["message"],
                            recommendation=v["recommendation"]
                        ))
                    # Store recommendations and score for later use
                    self.recommendations = recommendations
                    self.overall_score = score
                    self.is_manufacturable = is_manufacturable
                    return violations, "huggingface"
                except Exception as e:
                    logger.error(f"Error parsing Hugging Face response: {str(e)}")
                    raise
            
            return violations, self.api_source
            
        except Exception as e:
            logger.error(f"Error performing LLM analysis: {str(e)}")
            violations.append(DFMViolation(
                rule_id="llm_analysis",
                severity="error",
                message=f"Failed to perform LLM analysis: {str(e)}",
                recommendation="Check API key validity or try again later."
            ))
            return violations, "none"

    def _parse_openai_response(self, result_text: str) -> List[DFMViolation]:
        """Parse OpenAI's JSON response format."""
        violations = []
        try:
            # Extract JSON from response text (in case there's text surrounding it)
            import re
            json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', result_text, re.DOTALL)
            if json_match:
                result_text = json_match.group(1)
                logging.debug(f"Extracted JSON from markdown: {result_text}")
            
            # Clean up any potential issues
            result_text = result_text.strip()
            
            # Log the text we're trying to parse
            logging.debug(f"Attempting to parse JSON: {result_text}")
            
            # Try to parse as JSON
            try:
                data = json.loads(result_text)
                logging.debug(f"Successfully parsed JSON data: {data}")
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}, trying to clean up the JSON")
                # Try to clean up common JSON issues
                result_text = re.sub(r'[\n\r\t]', ' ', result_text)
                result_text = re.sub(r'\s+', ' ', result_text)
                result_text = re.sub(r'\\+(?!["\\/bfnrtu])', '', result_text)
                data = json.loads(result_text)
                logging.debug(f"Successfully parsed JSON after cleanup: {data}")
            
            # Check if data is a list (JSON array as requested in our template)
            if isinstance(data, list):
                for item in data:
                    # Clean up keys by properly removing quotes that might be part of the key name
                    violation_data = {}
                    for key, value in item.items():
                        # More aggressive cleaning to handle various quoting issues
                        cleaned_key = key.replace('"', '').replace("'", '').replace('\n', '').replace('\\', '').strip()
                        violation_data[cleaned_key] = value
                    
                    # Now check for required fields with cleaned keys
                    if "issue" in violation_data and "severity" in violation_data:
                        violations.append(DFMViolation(
                            rule_id="llm_analysis",
                            severity=violation_data.get("severity", "info"),
                            message=violation_data.get("issue", ""),
                            recommendation=violation_data.get("recommendation", "No specific recommendation provided")
                        ))
                    else:
                        # If the expected keys aren't found, log what was found
                        logging.error(f"Missing required keys in violation data. Found keys: {list(violation_data.keys())}")
                        if any(key.endswith('issue') or 'issue' in key for key in violation_data.keys()):
                            # Try to find the closest match to 'issue'
                            for key in violation_data.keys():
                                if 'issue' in key:
                                    issue_key = key
                                    severity_key = next((k for k in violation_data.keys() if 'severity' in k), None)
                                    recommendation_key = next((k for k in violation_data.keys() if 'recommend' in k), None)
                                    
                                    violations.append(DFMViolation(
                                        rule_id="llm_analysis",
                                        severity=violation_data.get(severity_key, "info") if severity_key else "info",
                                        message=violation_data.get(issue_key, ""),
                                        recommendation=violation_data.get(recommendation_key, "No specific recommendation provided") if recommendation_key else "No specific recommendation provided"
                                    ))
                                    break
            # Also handle the case where we get a JSON object with violations
            elif isinstance(data, dict):
                # Check if it's a violations array
                if "violations" in data:
                    for v in data["violations"]:
                        # Clean up keys
                        violation_data = {}
                        for key, value in v.items():
                            cleaned_key = key.replace('"', '').replace('\n', '').strip()
                            violation_data[cleaned_key] = value
                        
                        violations.append(DFMViolation(
                            rule_id="llm_analysis",
                            severity=violation_data.get("severity", "info"),
                            message=violation_data.get("issue", ""),
                            recommendation=violation_data.get("recommendation", "No specific recommendation provided")
                        ))
                # Check if it's directly the violation format without being wrapped
                elif any(key.replace('"', '').replace('\n', '').strip() == "issue" for key in data.keys()):
                    # Clean up keys
                    violation_data = {}
                    for key, value in data.items():
                        cleaned_key = key.replace('"', '').replace('\n', '').strip()
                        violation_data[cleaned_key] = value
                    
                    violations.append(DFMViolation(
                        rule_id="llm_analysis",
                        severity=violation_data.get("severity", "info"),
                        message=violation_data.get("issue", ""),
                        recommendation=violation_data.get("recommendation", "No specific recommendation provided")
                    ))
        except Exception as e:
            logger.error(f"Error parsing OpenAI response: {str(e)}. Response was: {result_text}")
            raise
        
        return violations
    
    def _parse_text_response(self, text):
        """Parse plain text response from LLM models like Hugging Face's."""
        violations = []
        recommendations = []
        
        # Process each line for issues
        severity_map = {
            "error": "error",
            "warning": "warning", 
            "info": "info",
            "information": "info"
        }
        
        issue_blocks = text.split("ISSUE:")
        
        # If no clear issue blocks or "No issues found" is in the text
        if len(issue_blocks) <= 1 and "no issues found" in text.lower():
            violations.append({
                "rule_id": "llm_analysis",
                "severity": "info",
                "message": "No manufacturability issues found in the provided PCB data.",
                "location": None,
                "recommendation": "The design appears to meet all manufacturing requirements."
            })
            recommendations.append("The design appears to meet all manufacturing requirements.")
            return violations, recommendations, 0.95, True
            
        # Process each issue block
        for block in issue_blocks[1:]:  # Skip the first element which is text before the first "ISSUE:"
            lines = block.strip().split('\n')
            
            issue_desc = lines[0].strip() if lines else ""
            severity = "error"  # Default severity
            recommendation = None
            
            # Extract severity and recommendation if they exist
            for line in lines[1:]:
                if line.strip().startswith("SEVERITY:"):
                    severity_text = line.replace("SEVERITY:", "").strip().lower()
                    severity = severity_map.get(severity_text, "error")
                elif line.strip().startswith("RECOMMENDATION:"):
                    recommendation = line.replace("RECOMMENDATION:", "").strip()
            
            # Skip placeholder violations that contain square brackets
            if "[" in issue_desc or (recommendation and "[" in recommendation):
                continue
            
            # Create violation entry
            violation = {
                "rule_id": "llm_analysis",
                "severity": severity,
                "message": issue_desc,
                "location": None,
                "recommendation": recommendation
            }
            
            violations.append(violation)
            if recommendation:
                recommendations.append(recommendation)
        
        # Calculate overall score and manufacturability
        error_count = sum(1 for v in violations if v["severity"] == "error")
        warning_count = sum(1 for v in violations if v["severity"] == "warning")
        
        # If no violations were extracted but we had issue blocks, add a generic one
        if not violations and len(issue_blocks) > 1:
            violations.append({
                "rule_id": "llm_analysis",
                "severity": "warning",
                "message": "Issues were detected but could not be parsed properly.",
                "location": None,
                "recommendation": "Review the design manually or try again with a different model."
            })
            error_count = 0
            warning_count = 1
        
        # Calculate score - more severe issues reduce score more significantly
        score = max(0.1, 1.0 - (error_count * 0.15) - (warning_count * 0.05))
        is_manufacturable = error_count == 0
        
        return violations, recommendations, score, is_manufacturable
    
    def _perform_rule_based_analysis(self, features: PCBFeatures, supplier_id: Optional[str] = None) -> List[DFMViolation]:
        """Perform rule-based analysis using basic rules and supplier capabilities"""
        violations = []
        
        # First check against basic manufacturing rules
        basic_rules = self.knowledge_base.get_manufacturing_rules()
        
        # Check trace rules
        trace_rules = basic_rules.get('trace_rules', {})
        if trace_rules:
            if features.min_trace_width < trace_rules['min_width']:
                violations.append(DFMViolation(
                    rule_id='trace_width',
                    severity='error',
                    message=f"Trace width {features.min_trace_width}mm is below minimum {trace_rules['min_width']}mm",
                    recommendation=f"Increase trace width to at least {trace_rules['min_width']}mm"
                ))
            
            if features.min_trace_spacing < trace_rules['min_spacing']:
                violations.append(DFMViolation(
                    rule_id='trace_spacing',
                    severity='error',
                    message=f"Trace spacing {features.min_trace_spacing}mm is below minimum {trace_rules['min_spacing']}mm",
                    recommendation=f"Increase trace spacing to at least {trace_rules['min_spacing']}mm"
                ))
        
        # Check via rules
        via_rules = basic_rules.get('via_rules', {})
        if via_rules:
            if features.min_via_size < via_rules['min_diameter']:
                violations.append(DFMViolation(
                    rule_id='via_diameter',
                    severity='error',
                    message=f"Via diameter {features.min_via_size}mm is below minimum {via_rules['min_diameter']}mm",
                    recommendation=f"Increase via diameter to at least {via_rules['min_diameter']}mm"
                ))
            
            if features.min_drill_size < via_rules['min_drill_size']:
                violations.append(DFMViolation(
                    rule_id='via_drill',
                    severity='error',
                    message=f"Via drill size {features.min_drill_size}mm is below minimum {via_rules['min_drill_size']}mm",
                    recommendation=f"Increase via drill size to at least {via_rules['min_drill_size']}mm"
                ))
        
        # Check board rules
        board_rules = basic_rules.get('board_rules', {})
        if board_rules and hasattr(features, 'layer_count'):
            if features.layer_count < board_rules['min_layers']:
                violations.append(DFMViolation(
                    rule_id='layer_count',
                    severity='warning',
                    message=f"Layer count {features.layer_count} is below recommended minimum {board_rules['min_layers']}",
                    recommendation="Consider increasing layer count for better signal integrity"
                ))
            
            if features.layer_count > board_rules['max_layers']:
                violations.append(DFMViolation(
                    rule_id='layer_count',
                    severity='error',
                    message=f"Layer count {features.layer_count} exceeds maximum {board_rules['max_layers']}",
                    recommendation=f"Reduce layer count to {board_rules['max_layers']} or less"
                ))
        
        # If supplier ID is provided, check against supplier capabilities
        if supplier_id:
            supplier_profile = self.knowledge_base.get_supplier_profile(supplier_id)
            if supplier_profile:
                capabilities = supplier_profile.get('capabilities', {})
                
                # Check trace capabilities
                trace_caps = capabilities.get('trace_capabilities', {})
                if trace_caps:
                    if features.min_trace_width < trace_caps['min_width']:
                        violations.append(DFMViolation(
                            rule_id='supplier_trace_width',
                            severity='error',
                            message=f"Trace width {features.min_trace_width}mm is below supplier minimum {trace_caps['min_width']}mm",
                            recommendation=f"Increase trace width to meet supplier minimum of {trace_caps['min_width']}mm"
                        ))
                    
                    if features.min_trace_spacing < trace_caps['min_spacing']:
                        violations.append(DFMViolation(
                            rule_id='supplier_trace_spacing',
                            severity='error',
                            message=f"Trace spacing {features.min_trace_spacing}mm is below supplier minimum {trace_caps['min_spacing']}mm",
                            recommendation=f"Increase trace spacing to meet supplier minimum of {trace_caps['min_spacing']}mm"
                        ))
                
                # Check via capabilities
                via_caps = capabilities.get('via_capabilities', {})
                if via_caps:
                    if features.min_via_size < via_caps['min_diameter']:
                        violations.append(DFMViolation(
                            rule_id='supplier_via_diameter',
                            severity='error',
                            message=f"Via diameter {features.min_via_size}mm is below supplier minimum {via_caps['min_diameter']}mm",
                            recommendation=f"Increase via diameter to meet supplier minimum of {via_caps['min_diameter']}mm"
                        ))
                    
                    if features.min_drill_size < via_caps['min_drill_size']:
                        violations.append(DFMViolation(
                            rule_id='supplier_via_drill',
                            severity='error',
                            message=f"Via drill size {features.min_drill_size}mm is below supplier minimum {via_caps['min_drill_size']}mm",
                            recommendation=f"Increase via drill size to meet supplier minimum of {via_caps['min_drill_size']}mm"
                        ))
        
        return violations
    
    def _calculate_manufacturability_score(self, violations: List[DFMViolation]) -> float:
        """Calculate overall manufacturability score"""
        if not violations:
            return 1.0
        
        # Weight by severity
        weights = {
            'error': 1.0,
            'warning': 0.5,
            'info': 0.1
        }
        
        # Count violations by severity
        error_count = sum(1 for v in violations if v.severity == 'error')
        warning_count = sum(1 for v in violations if v.severity == 'warning')
        
        # Base score calculation
        base_score = 1.0
        
        # Reduce score based on violations - more sophisticated algorithm
        if error_count > 0:
            # Each error reduces score by 0.15, with diminishing returns
            error_penalty = min(0.6, 0.15 * error_count)
            base_score -= error_penalty
        
        if warning_count > 0:
            # Each warning reduces score by 0.05, with diminishing returns
            warning_penalty = min(0.3, 0.05 * warning_count)
            base_score -= warning_penalty
        
        return max(0.0, min(1.0, base_score))
    
    def _generate_recommendations(self, violations: List[DFMViolation]) -> List[str]:
        """Generate ordered list of recommendations"""
        recommendations = []
        
        # Process violations by severity (errors first, then warnings, then info)
        for severity in ['error', 'warning', 'info']:
            for violation in [v for v in violations if v.severity == severity]:
                if violation.recommendation and violation.recommendation not in recommendations:
                    recommendations.append(violation.recommendation)
        
        return recommendations 