"""Deformation source models for plotting.

This module contains classes representing different types of deformation sources
(e.g., Mogi, Spheroid, Penny-shaped crack, Okada fault) used in geophysical modeling.
Each class is responsible for plotting its own representation on a map.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle, Polygon
from matplotlib.transforms import Affine2D


class Mogi:
    """Mogi point source model representation."""
    
    def __init__(self, ax: Axes, xcen: float, ycen: float) -> None:
        """Initialize and plot Mogi source.
        
        Args:
            ax: Matplotlib axes object
            xcen: X coordinate of center
            ycen: Y coordinate of center
        """
        self.x = xcen
        self.y = ycen
        self._plot_source(ax)

    def _plot_source(self, ax: Axes) -> None:
        """Plot the Mogi source as a point marker."""
        ax.scatter(self.x, self.y, s=15, color="black", linewidth=2, marker="x")


class Spheroid:
    """Spheroid deformation source model."""
    
    def __init__(
        self, 
        ax: Axes, 
        xcen: float, 
        ycen: float, 
        s_axis_max: float, 
        ratio: float, 
        strike: float, 
        dip: float
    ) -> None:
        """Initialize and plot Spheroid source.
        
        Args:
            ax: Matplotlib axes object
            xcen: X coordinate of center
            ycen: Y coordinate of center
            s_axis_max: Maximum semi-axis length
            ratio: Axis ratio
            strike: Strike angle in degrees
            dip: Dip angle in degrees
        """
        self.x = xcen
        self.y = ycen
        self.s_axis = s_axis_max
        self.ratio = ratio
        self.strike = strike
        self.dip = dip
        self._plot_source(ax)

    def _plot_source(self, ax: Axes) -> None:
        """Plot the spheroid with major and minor axes."""
        # Calculate semi-minor axis
        s_minor = self.s_axis * self.ratio

        # Convert angles to radians
        strike_rad = np.radians(self.strike - 90)
        dip_rad = np.radians(self.dip)

        # Adjust the semi-major axis length for the dip projection
        s_axis_projected = self.s_axis * np.sin(dip_rad)

        # Calculate endpoints of the major axis (with dip projection)
        dx_major = s_axis_projected * np.cos(strike_rad)
        dy_major = s_axis_projected * np.sin(strike_rad)
        x_major = [self.x - dx_major, self.x + dx_major]
        y_major = [self.y - dy_major, self.y + dy_major]

        # Calculate endpoints of the minor axis (without dip projection)
        dx_minor = s_minor * np.sin(strike_rad)
        dy_minor = s_minor * -np.cos(strike_rad)
        x_minor = [self.x - dx_minor, self.x + dx_minor]
        y_minor = [self.y - dy_minor, self.y + dy_minor]

        ax.plot(x_major, y_major, 'r-', label='Major Axis')
        ax.plot(x_minor, y_minor, 'b-', label='Minor Axis')


class Penny:
    """Penny-shaped crack deformation source model."""
    
    def __init__(self, ax: Axes, xcen: float, ycen: float, radius: float) -> None:
        """Initialize and plot Penny source.
        
        Args:
            ax: Matplotlib axes object
            xcen: X coordinate of center
            ycen: Y coordinate of center
            radius: Radius of the penny-shaped crack
        """
        self.x = xcen
        self.y = ycen
        self.radius = radius
        self._plot_source(ax)

    def _plot_source(self, ax: Axes) -> None:
        """Plot the penny-shaped crack as a circle."""
        circle = plt.Circle(
            (self.x, self.y), 
            self.radius, 
            edgecolor='black', 
            color="#7cc0ff", 
            fill=True, 
            alpha=0.7, 
            label='Penny'
        )
        ax.add_patch(circle)


class Okada:
    """Okada rectangular dislocation fault model."""
    
    def __init__(
        self, 
        ax: Axes, 
        xtlc: float, 
        ytlc: float, 
        length: float, 
        width: float, 
        strike: float, 
        dip: float
    ) -> None:
        """Initialize and plot Okada fault.
        
        Args:
            ax: Matplotlib axes object
            xtlc: X coordinate of top-left corner
            ytlc: Y coordinate of top-left corner
            length: Fault length
            width: Fault width
            strike: Strike angle in degrees
            dip: Dip angle in degrees
        """
        self.xtlc = xtlc
        self.ytlc = ytlc
        self.length = length
        self.width = width
        self.strike = strike
        self.dip = dip
        self._plot_source(ax)

    def _plot_source(self, ax: Axes) -> None:
        """Plot the Okada fault as a rectangle with dip indicators."""
        dip_radians = np.radians(self.dip)
        projected_width = self.width * np.cos(dip_radians)
        height = abs(projected_width)

        # Create rectangle in local coordinates
        local_rect = Rectangle(
            (0.0, -height),
            self.length, 
            height,
            edgecolor='black',
            lw=1,
            alpha=0.2
        )
        
        # Rotate around local origin and translate to position
        t = Affine2D().rotate_deg(90 - self.strike).translate(self.xtlc, self.ytlc)
        local_rect.set_transform(t + ax.transData)
        ax.add_patch(local_rect)

        # Add triangles along the fault to indicate dip direction
        self._add_dip_indicators(ax, t, height)

    def _add_dip_indicators(self, ax: Axes, transform: Affine2D, height: float) -> None:
        """Add small triangles to indicate dip direction.
        
        Args:
            ax: Matplotlib axes object
            transform: Affine2D transformation
            height: Projected height of fault
        """
        try:
            base_half_main = max(0.04 * self.length, 0.01 * self.length)
            tri_color = 'black'
            tri_edge = 'black'
            tri_alpha = 0.3

            # Add several triangles along the fault length
            n_extra = 6
            extra_positions = np.linspace(0.1, 0.9, n_extra)
            base_half_small = base_half_main * 0.5
            tip_offset_small = 0.6
            
            for pos in extra_positions:
                # Skip center to avoid overlap
                if abs(pos - 0.5) < 1e-6:
                    continue
                    
                left = (pos * self.length - base_half_small, 0.0)
                right = (pos * self.length + base_half_small, 0.0)
                tip = (pos * self.length, -height * tip_offset_small)
                
                tri_small = Polygon(
                    [left, right, tip], 
                    closed=True,
                    facecolor=tri_color, 
                    edgecolor=tri_edge, 
                    linewidth=0.6,
                    zorder=29, 
                    alpha=tri_alpha
                )
                tri_small.set_transform(transform + ax.transData)
                ax.add_patch(tri_small)
        except Exception:
            # Non-fatal: continue without dip indicators if something goes wrong
            pass


class SourcePlotter:
    """Factory-like class for plotting deformation sources."""
    
    # Mapping of source types to their classes and required attributes
    SOURCE_TYPES: Dict[str, Dict[str, Any]] = {
        "mogi": {
            "class": Mogi, 
            "attributes": ["xcen", "ycen"]
        },
        "spheroid": {
            "class": Spheroid, 
            "attributes": ["xcen", "ycen", "s_axis_max", "ratio", "strike", "dip"]
        },
        "penny": {
            "class": Penny,  
            "attributes": ["xcen", "ycen", "radius"]
        },
        "okada": {
            "class": Okada,  
            "attributes": ["ytlc", "xtlc", "length", "width", "strike", "dip"]
        },
    }
    
    @classmethod
    def plot_sources(cls, ax: Axes, sources: Optional[Dict[str, Dict[str, float]]]) -> None:
        """Plot deformation sources on the given axes.
        
        Args:
            ax: Matplotlib axes object
            sources: Dictionary of source definitions
        """
        if not sources:
            return
            
        for source_name, source_params in sources.items():
            source_keys = set(source_params.keys())
            
            # Find matching source type
            for source_type, config in cls.SOURCE_TYPES.items():
                if set(config["attributes"]) == source_keys:
                    model_class = config["class"]
                    model_class(ax, **source_params)
                    break
