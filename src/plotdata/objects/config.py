"""Configuration classes for data processing.

This module provides configuration classes to replace dynamic attribute
copying with explicit, type-safe configuration objects.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ProcessingConfig:
    """Configuration for data processing operations.
    
    This class explicitly defines all processing parameters, making the code
    more maintainable and following the Single Responsibility Principle.
    """
    
    # Data directories
    data_dir: List[str]
    
    # Optional parameters with defaults
    mask: Optional[List[str]] = None
    model: Optional[str] = None
    ref_lalo: Optional[List[float]] = None
    region: Optional[List[float]] = None
    window_size: int = 10
    no_sources: bool = False
    
    # File paths that may be set during processing
    ascending: Optional[str] = None
    descending: Optional[str] = None
    horizontal: Optional[str] = None
    vertical: Optional[str] = None
    
    @classmethod
    def from_namespace(cls, inps):
        """Create configuration from argparse namespace.
        
        Args:
            inps: Argparse namespace object
            
        Returns:
            ProcessingConfig instance
        """
        return cls(
            data_dir=getattr(inps, 'data_dir', []),
            mask=getattr(inps, 'mask', None),
            model=getattr(inps, 'model', None),
            ref_lalo=getattr(inps, 'ref_lalo', None),
            region=getattr(inps, 'region', None),
            window_size=getattr(inps, 'window_size', 10),
            no_sources=getattr(inps, 'no_sources', False),
        )


@dataclass
class PlottingConfig:
    """Configuration for plotting operations."""
    
    # Common plotting parameters
    style: str = 'pixel'
    cmap: str = 'jet'
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    unit: str = 'mm/yr'
    no_colorbar: bool = False
    colorbar_size: float = 0.8
    no_dem: bool = False
    subset: Optional[str] = None
    zoom: Optional[float] = None
    
    # Contour settings
    contour: Optional[List[float]] = None
    color: str = 'black'
    contour_linewidth: float = 1.0
    inline: bool = False
    resolution: str = '03s'
    
    # Point and line settings
    lalo: Optional[List[float]] = None
    line: Optional[List] = None
    ref_lalo: Optional[List[float]] = None
    volcano: bool = False
    sources: Optional[dict] = None
    
    @classmethod
    def from_namespace(cls, inps):
        """Create configuration from argparse namespace.
        
        Args:
            inps: Argparse namespace object
            
        Returns:
            PlottingConfig instance
        """
        return cls(
            style=getattr(inps, 'style', 'pixel'),
            cmap=getattr(inps, 'cmap', 'jet'),
            vmin=getattr(inps, 'vmin', None),
            vmax=getattr(inps, 'vmax', None),
            unit=getattr(inps, 'unit', 'mm/yr'),
            no_colorbar=getattr(inps, 'no_colorbar', False),
            colorbar_size=getattr(inps, 'colorbar_size', 0.8),
            no_dem=getattr(inps, 'no_dem', False),
            subset=getattr(inps, 'subset', None),
            zoom=getattr(inps, 'zoom', None),
            contour=getattr(inps, 'contour', None),
            color=getattr(inps, 'color', 'black'),
            contour_linewidth=getattr(inps, 'contour_linewidth', 1.0),
            inline=getattr(inps, 'inline', False),
            resolution=getattr(inps, 'resolution', '03s'),
            lalo=getattr(inps, 'lalo', None),
            line=getattr(inps, 'line', None),
            ref_lalo=getattr(inps, 'ref_lalo', None),
            volcano=getattr(inps, 'volcano', False),
            sources=getattr(inps, 'sources', None),
        )
