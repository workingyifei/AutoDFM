import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any
from dataclasses import dataclass

# We're now inside src_pcb, so import directly from parent directory
from ..pcb_processing.pcb_parser import PCBParser, PCBFeatures
from ..knowledge_base.knowledge_base import KnowledgeBase
from ..dfm_analysis.dfm_analyzer import DFMAnalyzer, AnalysisMode

# Import config from parent directory
import sys
from .. import config

def load_tokens():
    """Load API tokens from tokens.json file"""
    try:
        tokens_path = Path(__file__).parent.parent / "tokens.json"
        if tokens_path.exists():
            with open(tokens_path, 'r') as f:
                tokens = json.load(f)
                # Set environment variables
                if "OPENAI_API_KEY" in tokens:
                    os.environ["OPENAI_API_KEY"] = tokens["OPENAI_API_KEY"]
                if "HUGGINGFACE_API_TOKEN" in tokens:
                    os.environ["HUGGINGFACE_API_TOKEN"] = tokens["HUGGINGFACE_API_TOKEN"]
                    # Also set HUGGINGFACEHUB_API_TOKEN (the one LangChain actually uses)
                    os.environ["HUGGINGFACEHUB_API_TOKEN"] = tokens["HUGGINGFACE_API_TOKEN"]
                print(f"Loaded API tokens from {tokens_path}")
    except Exception as e:
        print(f"Warning: Failed to load tokens from tokens.json: {str(e)}")

def test_parser(gerber_dir: str) -> Dict[str, Any]:
    """Test PCB parser with Gerber files"""
    parser = PCBParser()
    features = parser.parse_gerber_folder(gerber_dir)
    print("PCB Features:")
    features_dict = {
        "board_outline": features.board_outline,
        "layer_count": features.layer_count,
        "min_trace_width": features.min_trace_width,
        "min_trace_spacing": features.min_trace_spacing,
        "min_via_size": features.min_via_size,
        "min_drill_size": features.min_drill_size,
        "component_density": features.component_density,
        "edge_clearance": features.edge_clearance,
        "copper_weight": features.copper_weight
    }
    print(json.dumps(features_dict, indent=2))
    return features_dict

def test_analyzer(pcb_features: Dict[str, Any], analysis_mode: str = "auto", supplier_id: Optional[str] = None) -> Dict[str, Any]:
    """Test DFM analyzer with PCB features"""
    try:
        kb = KnowledgeBase()
        analyzer = DFMAnalyzer(knowledge_base=kb, analysis_mode=analysis_mode)
        
        # Convert dictionary back to PCBFeatures object
        features_obj = PCBFeatures(
            board_outline=tuple(pcb_features["board_outline"]),
            layer_count=pcb_features["layer_count"],
            min_trace_width=pcb_features["min_trace_width"],
            min_trace_spacing=pcb_features["min_trace_spacing"],
            min_via_size=pcb_features["min_via_size"],
            min_drill_size=pcb_features["min_drill_size"],
            component_density=pcb_features["component_density"],
            edge_clearance=pcb_features["edge_clearance"],
            copper_weight=pcb_features["copper_weight"]
        )
        
        dfm_report = analyzer.analyze_pcb(features_obj, supplier_id)
        return dfm_report
    except Exception as e:
        print(f"Error in DFM analysis: {str(e)}")
        return {}

def test_main_workflow(gerber_dir: str, analysis_mode: str = "auto", supplier_id: Optional[str] = None, output_dir: Optional[str] = None) -> str:
    """Run the complete test workflow"""
    # First test the parser
    pcb_features = test_parser(gerber_dir)
    
    # Then test the analyzer
    dfm_report = test_analyzer(pcb_features, analysis_mode, supplier_id)
    
    # Export results if we have them
    if dfm_report and hasattr(dfm_report, 'violations'):
        # Get raw LLM output if available
        raw_llm_output = None
        if hasattr(dfm_report, 'source') and dfm_report.source != "rule-based":
            raw_llm_output = "LLM analysis was performed but output not captured"
        
        # Convert DFMReport to dict for export
        report_dict = {
            "violations": [vars(v) for v in dfm_report.violations],
            "supplier_id": dfm_report.supplier_id,
            "overall_score": dfm_report.overall_score,
            "recommendations": dfm_report.recommendations,
            "is_manufacturable": dfm_report.is_manufacturable,
            "source": dfm_report.source
        }
        
        return export_dfm_analysis(report_dict, raw_llm_output, output_dir)
    
    return ""

def export_dfm_analysis(dfm_report: Dict[str, Any], raw_llm_output: Optional[str] = None, output_dir: Optional[str] = None) -> str:
    """Export DFM analysis results to file"""
    if not output_dir:
        output_dir = "dfm_exports"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp and analysis source
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_source = "rule_based"
    if raw_llm_output:
        if "openai" in dfm_report.get("metadata", {}).get("model", "").lower():
            analysis_source = "openai"
        else:
            analysis_source = "huggingface"
    
    filename = f"dfm_analysis_{analysis_source}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Export results
    with open(filepath, 'w') as f:
        json.dump({
            "dfm_report": dfm_report,
            "raw_llm_output": raw_llm_output
        }, f, indent=2)
    
    print(f"DFM analysis exported to: {filepath}")
    return filepath

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PCB DFM Analysis")
    parser.add_argument("--gerber-dir", type=str, default="src_pcb/gerbers",
                      help="Directory containing Gerber files")
    parser.add_argument("--analysis-mode", type=str, default="auto",
                      choices=["auto", "rule-based", "openai", "huggingface"],
                      help="Analysis mode to use")
    parser.add_argument("--supplier-id", type=str,
                      help="Optional supplier ID for capability checking")
    parser.add_argument("--output-dir", type=str,
                      help="Optional output directory for DFM report")
    
    args = parser.parse_args()
    
    # Load tokens before running the workflow
    load_tokens()
    
    test_main_workflow(args.gerber_dir, args.analysis_mode, 
                      args.supplier_id, args.output_dir) 