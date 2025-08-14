# PlotData
## TODOs
- [ ] Change examples
- [ ] Create notebook tutorial
- [ ] Run tests
# 1. [Installation](https://github.com/geodesymiami/PlotData/blob/main/docs/installation.md)
# 2. Run code
The package is coupled to **MintPy** and **MiaplPy**, so it leverages their data structures.
Check the [Installation guide](https://github.com/geodesymiami/PlotData/blob/main/docs/installation.md) to download the following dataset
```python
plotdata MaunaLoaSenDT87/mintpy MaunaLoaSenAT124/mintpy --template default  --period 20181001:20191031 --ref-lalo 19.50068 -155.55856 --resolution '01s' --contour 2 --lalo 19.461,-155.558 --num-vectors 40
```
