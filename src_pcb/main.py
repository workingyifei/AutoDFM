import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json

# Add project root to Python path to import config
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from . import config

# Import core AutoDFM components
from .knowledge_base.knowledge_base import KnowledgeBase
from .pcb_processing.pcb_parser import PCBParser
from .dfm_analysis.dfm_analyzer import DFMAnalyzer


def direct_analyze_pcb(gerber_folder_path, supplier_id=None):
    """
    Directly analyze PCB Gerber files from a local folder
    
    Args:
        gerber_folder_path: Path to folder containing Gerber files
        supplier_id: Optional supplier ID for capability matching
    
    Returns:
        DFM analysis report and analyzer for raw LLM output
    """
    try:
        # Initialize components
        kb = KnowledgeBase()
        parser = PCBParser()
        analyzer = DFMAnalyzer(kb)
        
        # Parse PCB files
        pcb_features = parser.parse_gerber_folder(gerber_folder_path)
        
        # Analyze DFM
        dfm_report = analyzer.analyze_pcb(pcb_features, supplier_id)
        
        # Convert report to dict for output
        report_dict = {
            "violations": [vars(v) for v in dfm_report.violations],
            "supplier_id": dfm_report.supplier_id,
            "overall_score": dfm_report.overall_score,
            "recommendations": dfm_report.recommendations,
            "is_manufacturable": dfm_report.is_manufacturable,
            "analysis_type": "llm" if analyzer.llm else "rule-based"
        }
        
        return dfm_report, report_dict, analyzer
    
    except Exception as e:
        print(f"Error analyzing PCB files: {str(e)}")
        return None, None, None

def export_dfm_analysis(dfm_report, raw_llm_output=None, output_dir=None):
    """
    Export the DFM analysis results to a file
    
    Args:
        dfm_report: DFM analysis report object or dictionary
        raw_llm_output: Optional raw LLM output to include
        output_dir: Optional output directory path
    
    Returns:
        Path to the exported file
    """
    # Get export directory from config if not provided
    if output_dir is None:
        output_dir = config.get_export_dir()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source = getattr(dfm_report, "source", "unknown") if not isinstance(dfm_report, dict) else dfm_report.get("source", "unknown")
    filename = f"dfm_analysis_{source}_{timestamp}.json"
    output_path = os.path.join(output_dir, filename)
    
    # Convert report to dict for JSON output if it's not already a dict
    if not isinstance(dfm_report, dict):
        report_dict = {
            "violations": [vars(v) for v in dfm_report.violations],
            "supplier_id": dfm_report.supplier_id,
            "overall_score": dfm_report.overall_score,
            "recommendations": dfm_report.recommendations,
            "is_manufacturable": dfm_report.is_manufacturable,
            "source": dfm_report.source if hasattr(dfm_report, "source") else "unknown"
        }
    else:
        report_dict = dfm_report
    
    # Create export data
    export_data = {
        "raw_llm_output": raw_llm_output,
        "processed_report": report_dict,
        "metadata": {
            "timestamp": timestamp,
            "source": source,
            "export_version": "1.0"
        }
    }
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"DFM analysis exported to: {output_path}")
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PCB DFM Analysis Tool")
    parser.add_argument("--gerber", type=str, help="Path to gerber files")
    parser.add_argument("--supplier", type=str, help="Supplier ID")
    parser.add_argument("--export", action="store_true", help="Export analysis results")
    parser.add_argument("--output-dir", type=str, help="Output directory for export")
    
    args = parser.parse_args()
    
    if args.gerber:
        # Perform direct analysis
        gerber_path = args.gerber
        supplier_id = args.supplier
        
        print(f"Analyzing PCB Gerber files in: {gerber_path}")
        
        # Run analysis
        dfm_report, report_dict, analyzer = direct_analyze_pcb(gerber_path, supplier_id)
        
        if dfm_report:
            # Display the report
            print(f"\nDFM Report for {'supplier ' + supplier_id if supplier_id else 'general analysis'}:")
            print(f"Overall Manufacturability Score: {dfm_report.overall_score:.2f}")
            print(f"Is Manufacturable: {dfm_report.is_manufacturable}")
            
            print("\nViolations:")
            for violation in dfm_report.violations:
                print(f"- [{violation.severity.upper()}] {violation.message}")
                if violation.recommendation:
                    print(f"  Recommendation: {violation.recommendation}")
            
            print("\nRecommendations:")
            for rec in dfm_report.recommendations:
                print(f"- {rec}")
            
            # Export if requested
            if args.export:
                raw_llm_output = analyzer.last_llm_response if analyzer and analyzer.api_source in ["openai", "huggingface"] else None
                export_path = export_dfm_analysis(dfm_report, raw_llm_output, args.output_dir)
                print(f"Analysis exported to: {export_path}")
        else:
            print("Analysis failed. Please check the Gerber files and try again.")
    else:
        print("Please provide Gerber files path with --gerber")
        print("Example: python -m src_pcb.main --gerber ./gerbers --supplier PCBA_MFG_001 --export") 