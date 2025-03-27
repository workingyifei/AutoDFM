import sys
import os
import json

# Import from root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import PCB parser
from src_pcb.pcb_processing.pcb_parser import PCBParser

def test_gerber_parser():
    """Test the PCB parser with Gerber files"""
    print("=== Testing PCB Parser ===\n")
    
    # Path to the Gerber files - updated to correct location
    gerber_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src_pcb/gerbers'))
    
    print(f"Gerber File Analysis Results:")
    print("-" * 80)
    
    # Check if the Gerber directory exists
    if not os.path.exists(gerber_dir):
        print(f"Gerber directory not found at {gerber_dir}")
        print("Please make sure the gerbers directory exists and contains Gerber files.")
        return
    
    try:
        # Create a PCB parser instance
        parser = PCBParser()
        
        # Parse the Gerber files using the correct method name
        features = parser.parse_gerber_folder(gerber_dir)
        
        # Convert PCBFeatures to a dictionary for JSON serialization
        features_dict = {
            "board_outline": str(features.board_outline),
            "layer_count": features.layer_count,
            "min_trace_width": float(features.min_trace_width),
            "min_trace_spacing": float(features.min_trace_spacing),
            "min_via_size": float(features.min_via_size),
            "min_drill_size": float(features.min_drill_size),
            "component_density": float(features.component_density),
            "edge_clearance": float(features.edge_clearance),
            "copper_weight": float(features.copper_weight)
        }
        
        # Print the extracted features
        print(f"Successfully extracted PCB features from {gerber_dir}")
        print(f"Features: {json.dumps(features_dict, indent=2)}")
        
    except Exception as e:
        print(f"Error during PCB parsing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gerber_parser() 