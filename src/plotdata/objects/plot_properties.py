import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class PlotTemplate:
    def __init__(self, name="default"):
        self.name = name
        self.layout = self._get_layout(name)
        self.figsize = (10, 8)
        self.squeeze = False
        self.constrained_layout = True
        self.metadata = self._get_metadata(name)

    def _get_layout(self, name):
        layouts = {
            "default": [
                ["ascending.point", "horizontal.point", "seismicmap"],
                ["descending.point", "vertical.point", "seismicity.distance"],
                ["timeseries", "vectors", "seismicity.date"],
            ],
            "test": [
                ["ascending.section"],
                ["ascending.point"],
                ["seismicity.date"],
            ]
        }
        return layouts[name]

    def _get_metadata(self, name):
        return {
            "time_series": {"font_size": 8},
            "vector_plot": {"color": "blue"}
            # add per-panel options if needed
        }


class PlotRenderer:
    def __init__(self, inps, template: PlotTemplate, plotter_map: dict):
        self.inps = inps                      # argparse input object
        self.template = template              # Layout, figsize, settings, etc.
        self.plotter_map = plotter_map        # Mapping of plot types to classes + attributes

    def render(self, process_data):
        # 1. Create figure and axes from the template
        grid = PlotGrid(self.template, inps=process_data)
        fig, axes = grid.fig, grid.axes

        # 2. Instantiate the plotter objects with the appropriate data
        plotters = self._build_plotters(process_data)

        # 3. Loop over the layout and render into matching axes
        for name, plotter in plotters.items():
            if name not in axes:
                continue

            ax = axes[name]
            plotter.plot(ax)

        return fig

    def _build_plotters(self, process_data):
        """Build plotter instances using the configured classes and required file attributes."""
        plotters = {}

        for row in self.template.layout:
            for element in row:
                for name, configs in self.plotter_map.items():
                    if element.split('.')[0] in name:
                        cls = configs["class"]
                        files = [getattr(process_data, attr) for attr in configs["attributes"]]

                        plotter_instance = cls(files, self.inps)
                        plotters[element] = plotter_instance

        return plotters


class PlotGrid:
    def __init__(self, template: PlotTemplate, inps):
        self.template = template
        self.fig, self.axes = self._create_axes(inps)

    def _create_axes(self, inps):
        fig, axs = plt.subplot_mosaic(
            self.template.layout,
            figsize=(12, 10),
            constrained_layout=self.template.constrained_layout,
        )

        fig.suptitle(f"{inps.project}  {inps.start_date}-{inps.end_date}", fontsize=14, fontweight="bold")
        for ax in axs.values():
            # Apply custom layout settings
            ax.set_box_aspect(0.5)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=4))

            if inps and hasattr(inps, "font_size"):
                font_size = inps.font_size
                ax.tick_params(axis='x', labelsize=font_size)
                ax.tick_params(axis='y', labelsize=font_size)
        return fig, axs

    def get_axes(self):
        return self.fig, self.axes