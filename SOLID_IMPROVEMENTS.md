# SOLID Principles Improvements

This document summarizes the improvements made to the PlotData codebase to better follow SOLID principles.

## Overview

The codebase has been refactored to improve maintainability, testability, and extensibility by following SOLID design principles.

## Key Improvements

### 1. Single Responsibility Principle (SRP)

**Problem:** The `VelocityPlot` class (1024 lines) was handling too many responsibilities:
- Velocity data plotting
- DEM/hillshade rendering
- Scale bar drawing
- Deformation source plotting
- Axis management

**Solution:** Extracted specialized utility classes:

#### New Files Created:

1. **`plot_utilities.py`** - Plotting utilities with focused responsibilities:
   - `ScalePlotter` - Handles scale bar rendering
   - `DEMPlotter` - Handles DEM/hillshade visualization
   - `AxisLimitsManager` - Manages axis limits and zoom functionality

2. **`deformation_sources.py`** - Deformation source models:
   - `Mogi` - Mogi point source
   - `Spheroid` - Spheroid deformation source
   - `Penny` - Penny-shaped crack
   - `Okada` - Okada rectangular dislocation
   - `SourcePlotter` - Factory-like class for plotting sources

3. **`config.py`** - Configuration classes:
   - `ProcessingConfig` - Data processing configuration
   - `PlottingConfig` - Plotting parameters configuration

#### Benefits:
- Each class now has a single, well-defined responsibility
- Code is easier to test in isolation
- Changes to one aspect don't affect others
- Easier to understand and maintain

### 2. Open/Closed Principle (OCP)

**Problem:** Hard-coded logic in source model plotting made it difficult to add new source types.

**Solution:** Created `SourcePlotter` class with a configurable `SOURCE_TYPES` dictionary:

```python
SOURCE_TYPES = {
    "mogi": {"class": Mogi, "attributes": ["xcen", "ycen"]},
    "spheroid": {"class": Spheroid, "attributes": ["xcen", "ycen", "s_axis_max", "ratio", "strike", "dip"]},
    # ... more source types
}
```

#### Benefits:
- New source types can be added without modifying existing code
- Source type matching is configuration-driven
- Extension is straightforward and safe

### 3. Dependency Inversion Principle (DIP)

**Problem:** Classes depended on concrete implementations through dynamic attribute copying:

```python
# OLD: Dynamic attribute copying (anti-pattern)
for attr in dir(inps):
    if not attr.startswith('__') and not callable(getattr(inps, attr)):
        setattr(self, attr, getattr(inps, attr))
```

**Solution:** Use composition and explicit configuration:

```python
# NEW: Explicit dependencies with composition
self.scale_plotter = ScalePlotter()
self.dem_plotter = DEMPlotter()
self.axis_manager = AxisLimitsManager()
```

#### Benefits:
- Clear dependencies make code easier to understand
- Dependencies can be mocked for testing
- Type hints can be added
- IDE autocomplete works properly
- Reduces hidden coupling

### 4. Explicit Configuration over Dynamic Attributes

**Problem:** Dynamic attribute copying made it hard to:
- Know what attributes a class needs
- Track object state
- Refactor safely
- Use type hints

**Original problematic pattern in ALL plotter classes:**
```python
# Anti-pattern used in VelocityPlot, ProfilePlot, TimeseriesPlot, EarthquakePlot, VectorsPlot
def __init__(self, dataset, inps):
    for attr in dir(inps):
        if not attr.startswith('__') and not callable(getattr(inps, attr)):
            setattr(self, attr, getattr(inps, attr))  # Hidden dependencies!
```

**Solution:** Created explicit configuration in ALL plotter classes:

**VelocityPlot:**
```python
def __init__(self, dataset, inps):
    # Explicit configuration
    self.style = getattr(inps, 'style', 'pixel')
    self.cmap = getattr(inps, 'cmap', 'jet')
    self.vmin = getattr(inps, 'vmin', None)
    # ... explicit attributes with defaults
    
    # Use composition
    self.scale_plotter = ScalePlotter()
    self.dem_plotter = DEMPlotter()
    self.axis_manager = AxisLimitsManager()
```

**ProfilePlot:**
```python
def __init__(self, dataset, inps):
    # Extract parameters explicitly
    self.line = getattr(inps, 'line', None)
    self.norm = getattr(inps, 'norm', False)
    self.denoise = getattr(inps, 'denoise', None)
    self.unit = getattr(inps, 'unit', 'mm/yr')
```

**TimeseriesPlot:**
```python
def __init__(self, dataset, inps):
    # Extract parameters explicitly
    self.start_date = getattr(inps, 'start_date', None)
    self.end_date = getattr(inps, 'end_date', None)
    self.unit = getattr(inps, 'unit', 'mm/yr')
    self.offset = getattr(inps, 'offset', 0)
```

**EarthquakePlot:**
```python
def __init__(self, dataset, inps):
    # Extract parameters explicitly
    self.start_date = getattr(inps, 'start_date', None)
    self.end_date = getattr(inps, 'end_date', None)
    self.lalo = getattr(inps, 'lalo', None)
    self.region = getattr(inps, 'region', None)
```

**VectorsPlot:**
```python
def __init__(self, dataset, inps):
    # Extract parameters explicitly
    self.line = getattr(inps, 'line', None)
    self.region = getattr(inps, 'region', None)
    self.num_vectors = getattr(inps, 'num_vectors', 50)
    self.scale = getattr(inps, 'scale', None)
```

Or use configuration dataclasses:

```python
config = PlottingConfig.from_namespace(inps)
```

#### Benefits:
- Clear interface - easy to see what parameters are used
- Type safety with dataclasses
- Better IDE support
- Easier to refactor and maintain
- Self-documenting code
- No more hidden dependencies!

### 5. Improved DataExtractor Class

**Problem:** `DataExtractor` copied all attributes from `ProcessData` using dynamic reflection.

**Solution:** Use composition with `__getattr__` for backward compatibility:

```python
def __getattr__(self, name):
    """Delegate attribute access to process object for backward compatibility."""
    return getattr(self.process, name)
```

#### Benefits:
- Maintains a clear reference to the process object
- Backward compatible during migration
- Clear ownership of data
- Easier to refactor incrementally

## Code Quality Improvements

### Removed Code Duplication

- Source model classes (Mogi, Spheroid, Penny, Okada) were extracted to a separate module
- Plotting utilities (scale, DEM) extracted to reusable classes
- Reduces maintenance burden and bugs

### Better Separation of Concerns

- **Plotting logic** separated from **data processing**
- **Configuration** separated from **behavior**
- **Utility functions** organized by responsibility

### Easier to Extend

Adding new features is now easier:

1. **New plot type**: Add to dispatch map in DataExtractor
2. **New source model**: Add class and entry in SOURCE_TYPES
3. **New plotting utility**: Create focused utility class
4. **New configuration option**: Add to config dataclass

## Migration Guide

### For Adding New Features

1. Determine which component is responsible (following SRP)
2. Add new functionality to the appropriate class
3. Use configuration classes for parameters
4. Avoid dynamic attribute copying

### For Using VelocityPlot

```python
# OLD (still works)
plot = VelocityPlot(dataset, inps)

# NEW (recommended - with explicit config)
from plotdata.objects.config import PlottingConfig
config = PlottingConfig.from_namespace(inps)
plot = VelocityPlot(dataset, config)
```

## Testing Improvements

The refactored code is now easier to test:

```python
# Can test utilities in isolation
scale_plotter = ScalePlotter()
scale_plotter.plot_scale(mock_ax, zorder=1)

# Can test with mock dependencies
dem_plotter = DEMPlotter()
dem_plotter.plot_dem(mock_ax, mock_data, mock_attributes, region)

# Can test source plotting
SourcePlotter.plot_sources(mock_ax, {"source1": {"xcen": 0, "ycen": 0}})
```

## Future Improvements

While these changes significantly improve the codebase, further improvements could include:

1. **Complete config migration**: Convert all classes to use config dataclasses
2. **Strategy pattern for plot types**: Extract plot type logic into separate strategies
3. **Backend abstraction**: Create interface for matplotlib/pygmt operations
4. **File operations extraction**: Separate file I/O from business logic in ProcessData
5. **Add type hints**: Annotate all methods with type information
6. **Unit tests**: Add comprehensive test coverage for new utilities

## Summary

These refactorings make the codebase:

- ✅ **More maintainable** - Clear responsibilities and dependencies
- ✅ **More testable** - Can test components in isolation
- ✅ **More extensible** - Easy to add new features without modifying existing code
- ✅ **More understandable** - Explicit configuration and clear interfaces
- ✅ **More type-safe** - Can add type hints and use IDE features
- ✅ **Less error-prone** - Reduced dynamic behavior and hidden coupling

**All plotter classes refactored:**
- ✅ VelocityPlot
- ✅ ProfilePlot
- ✅ TimeseriesPlot
- ✅ EarthquakePlot
- ✅ VectorsPlot
- ✅ DataExtractor

The code now better follows SOLID principles while maintaining backward compatibility. **No more dynamic attribute copying in any plotter class!**
