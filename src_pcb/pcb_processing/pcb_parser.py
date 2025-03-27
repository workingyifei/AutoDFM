from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass
from gerbonara import LayerStack, GerberFile, ExcellonFile
from gerbonara.graphic_objects import Line, Arc, Flash, Region
from gerbonara.apertures import CircleAperture
import os

@dataclass
class PCBFeatures:
    board_outline: Tuple[float, float]  # width, height in mm
    layer_count: int
    min_trace_width: float  # in mm
    min_trace_spacing: float  # in mm
    min_via_size: float  # in mm
    min_drill_size: float  # in mm
    component_density: float  # components per square mm
    edge_clearance: float  # minimum clearance from board edge in mm
    copper_weight: float  # copper weight in oz

class PCBParser:
    def __init__(self):
        self.layer_stack = None
        self.features = None
    
    def parse_gerber_folder(self, gerber_path: str) -> PCBFeatures:
        """Parse a folder containing Gerber and Drill files"""
        try:
            path = Path(gerber_path)
            
            # Lists to store different layer types
            copper_layers = []
            drill_layers = []
            outline_layer = None
            
            # Process each file in the directory
            for file in path.glob("*"):
                suffix = file.suffix.upper()
                name = file.name.upper()
                
                try:
                    # Load Gerber files
                    if suffix in ['.GTL', '.GBL', '.GML', '.GKO', '.GBO', '.GTO']:
                        layer = GerberFile.open(str(file))
                        
                        # Categorize layers
                        if suffix in ['.GTL', '.GBL'] or any(x in name for x in ['TOP.CU', 'BOT.CU']):
                            copper_layers.append(layer)
                        elif suffix in ['.GML', '.GKO'] or 'OUTLINE' in name:
                            outline_layer = layer
                    
                    # Load drill files
                    elif suffix in ['.TXT', '.DRL', '.XLN'] or 'DRILL' in name:
                        layer = ExcellonFile.open(str(file))
                        drill_layers.append(layer)
                        
                except Exception as e:
                    print(f"Warning: Could not parse file {file}: {str(e)}")
                    continue
            
            if not copper_layers:
                raise ValueError("No copper layers found in Gerber folder")

            # If no outline layer found, try to use the first copper layer
            if not outline_layer:
                outline_layer = copper_layers[0]
            
            # Create LayerStack first, then set the outline property
            self.layer_stack = LayerStack()
            self.layer_stack._copper_layers = copper_layers
            self.layer_stack._drill_layers = drill_layers
            
            # Set the outline property - will use internal setter method
            setattr(self.layer_stack, '_outline', outline_layer)
            
            # Extract features
            self.features = self._extract_features()
            return self.features
            
        except Exception as e:
            raise ValueError(f"Error analyzing PCB: {str(e)}")
    
    def _extract_features(self) -> PCBFeatures:
        """Extract features from the PCB layers"""
        # Extract board outline dimensions
        if not hasattr(self.layer_stack, '_outline') or self.layer_stack._outline is None:
            raise ValueError("No board outline found")
        
        # Get bounding box of the outline
        outline = self.layer_stack._outline
        try:
            # First try calling bounding_box as a method
            bounds = outline.bounding_box()
        except (AttributeError, TypeError) as e:
            try:
                # Then try accessing it as a property
                bounds = outline.bounding_box
                if callable(bounds):
                    bounds = bounds()
            except (AttributeError, TypeError) as e:
                # If both fail, provide a helpful error
                raise ValueError(f"Could not get bounding box from outline layer: {type(outline)} - {str(e)}")
        
        try:
            width = float(bounds[1][0] - bounds[0][0])  # max_x - min_x
            height = float(bounds[1][1] - bounds[0][1])  # max_y - min_y
        except (TypeError, IndexError) as e:
            raise ValueError(f"Invalid bounding box format: {bounds} - {str(e)}")
        
        # Extract copper layers features
        copper_layers = getattr(self.layer_stack, '_copper_layers', [])
        if not copper_layers:
            copper_layers = []
        
        # Default values
        min_trace_width = float('inf')
        min_trace_spacing = float('inf')
        
        # Analyze copper layers for minimum track width and clearance
        if copper_layers:
            min_trace_width = self._analyze_traces(copper_layers)
            min_trace_spacing = 0.15  # TODO: Implement spacing analysis
        else:
            min_trace_width = 0.0
            min_trace_spacing = 0.0
        
        # Extract drill features
        drill_layers = getattr(self.layer_stack, '_drill_layers', [])
        if not drill_layers:
            drill_layers = []
        
        min_via_size, min_drill_size = self._analyze_vias(drill_layers)
        
        # Calculate component density (placeholder)
        component_density = 0.0
        
        # Calculate edge clearance (placeholder)
        edge_clearance = 0.25  # Default value
        
        return PCBFeatures(
            board_outline=(width, height),
            layer_count=len(copper_layers),
            min_trace_width=min_trace_width,
            min_trace_spacing=min_trace_spacing,
            min_via_size=min_via_size,
            min_drill_size=min_drill_size,
            component_density=component_density,
            edge_clearance=edge_clearance,
            copper_weight=1.0
        )
    
    def _analyze_traces(self, layers) -> float:
        """Analyze trace widths in copper layers."""
        min_width = float('inf')
        
        # Check if layers is empty or None
        if not layers:
            return 0.0
            
        # Ensure layers is iterable
        if not hasattr(layers, '__iter__'):
            print(f"Warning: layers is not iterable: {type(layers)}")
            return 0.0
        
        for layer in layers:
            if hasattr(layer, 'objects'):
                for obj in layer.objects:
                    if isinstance(obj, (Line, Arc)) and isinstance(obj.aperture, CircleAperture):
                        width = obj.aperture.diameter
                        if width > 0:  # Ignore zero-width traces
                            min_width = min(min_width, width)
        
        # Convert to mm if not already
        if min_width != float('inf'):
            min_width = float(min_width)
        else:
            min_width = 0.0
        
        return min_width
    
    def _analyze_vias(self, drill_layers: List) -> Tuple[float, float]:
        """Analyze via sizes and drill holes."""
        min_via_size = float('inf')
        min_drill_size = float('inf')
        
        # Check if drill_layers is empty or None
        if not drill_layers:
            return 0.0, 0.0
            
        # Ensure drill_layers is iterable
        if not hasattr(drill_layers, '__iter__'):
            print(f"Warning: drill_layers is not iterable: {type(drill_layers)}")
            return 0.0, 0.0
        
        for layer in drill_layers:
            for obj in layer.objects:
                if isinstance(obj, Flash) and isinstance(obj.aperture, CircleAperture):
                    size = obj.aperture.diameter
                    if size > 0:  # Ignore zero-size holes
                        min_drill_size = min(min_drill_size, size)
                        # Estimate via size as drill size + 0.2mm annular ring
                        via_size = size + 0.2
                        min_via_size = min(min_via_size, via_size)
        
        if min_via_size == float('inf'):
            min_via_size = 0.0
        if min_drill_size == float('inf'):
            min_drill_size = 0.0
            
        return min_via_size, min_drill_size 