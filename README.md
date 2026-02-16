[![CircleCI](https://dl.circleci.com/status-badge/img/gh/geodesymiami/PlotData/tree/main.svg?style=shield)](https://dl.circleci.com/status-badge/redirect/gh/geodesymiami/PlotData/tree/main)
# PlotData

## Recent Improvements

This codebase has been refactored to better follow SOLID design principles, improving:
- **Maintainability** - Clear responsibilities and dependencies
- **Testability** - Components can be tested in isolation
- **Extensibility** - Easy to add new features without modifying existing code
- **Type Safety** - Type hints for better IDE support and error prevention

See [SOLID_IMPROVEMENTS.md](SOLID_IMPROVEMENTS.md) for detailed documentation.

## TODOs
- [ ] Change examples
- [ ] Create notebook tutorial
- [ ] Run tests
# 1. [Installation](https://github.com/geodesymiami/PlotData/blob/main/docs/installation.md)

# 2. Download test data

Use the `get_plotdata_testdata.py` script to download sample data (--quick for hvGalapagos_mintpy.tar.gz) 
```bash
get_plotdata_testdata.py --quick
```

# 3. Run code
The package is coupled to **MintPy** and **MiaplPy**, so it leverages their data structures.

## Example Usage
Once you have test data (see section 2), you can run PlotData:

```python
plotdata hvGalapagos/mintpy --template default --ref-lalo -0.81 -91.190 --lalo -0.82528 -91.13791

# Example with Mauna Loa data (requires additional download)
plotdata MaunaLoaSenDT87/mintpy MaunaLoaSenAT124/mintpy --template default  --period 20181001:20191031 --ref-lalo 19.50068 -155.55856 --resolution '01s' --contour 2 --lalo 19.461,-155.558 --num-vectors 40
```

Check the [Installation guide](https://github.com/geodesymiami/PlotData/blob/main/docs/installation.md) for more details.
