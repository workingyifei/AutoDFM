import os
import json
import time
import sys
from datetime import datetime

# Import from parent directory
from ..main import export_dfm_analysis
from ..dfm_analysis.dfm_analyzer import DFMReport, DFMViolation

def test_direct_export():
    """Test the direct export functionality."""
    
    # Create a mock DFM report
    mock_report = DFMReport(
        violations=[
            DFMViolation(
                rule_id="test_rule",
                severity="warning",
                message="This is a test violation for the export functionality",
                recommendation="Fix the test issue"
            )
        ],
        supplier_id="TEST_SUPPLIER",
        overall_score=0.95,
        recommendations=["Fix the test issue"],
        is_manufacturable=True,
        source="test"
    )
    
    # Create a mock LLM output
    mock_llm_output = "This is a test LLM output for the export functionality"
    
    # Create test directory
    test_export_dir = "test_exports"
    os.makedirs(test_export_dir, exist_ok=True)
    
    # Export the report
    output_path = export_dfm_analysis(
        mock_report, 
        mock_llm_output,
        test_export_dir
    )
    
    print(f"\nSuccessfully exported DFM report to: {output_path}")
    
    # Print the data for verification
    with open(output_path, 'r') as f:
        exported_data = json.load(f)
    
    print("\nExported data:")
    print(json.dumps(exported_data, indent=2))
    
    return output_path

if __name__ == "__main__":
    test_direct_export() 