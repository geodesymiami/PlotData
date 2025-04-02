import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class PlotGrid:
    """Handles figure and axes creation based on plot type and number of processors."""
    def __init__(self, processors):
        self.plot_type = processors[0].plot_type
        self.processors = processors
        self.num_cols = len(processors)
        self.num_rows = self._determine_rows()
        self.fig, self.axes = self._define_axes()
        self._set_axes_properties()

    def _determine_rows(self):
        """Determine the number of rows dynamically based on processor files."""
        max_rows = 0
        self.squeeze = False

        for processor in self.processors:
            if self.plot_type == "shaded_relief":
                return 1  # Only one row for shaded relief
            elif self.plot_type == "velocity":
                rows = sum(1 for file in [processor.ascending, processor.descending] if file)
            elif self.plot_type == "horzvert":
                rows = sum(1 for file in [processor.horizontal, processor.vertical] if file)
            elif self.plot_type == "vectors":
                return 3
            else:
                rows = 1
            max_rows = max(max_rows, rows)
        return max_rows

    def _define_axes(self):
        """Creates figure and axes layout."""
        fig, axs = plt.subplots(nrows=self.num_rows, ncols=self.num_cols, layout="constrained", squeeze=self.squeeze)
        title = self.processors[0].project
        fig.suptitle(title, fontsize=14, fontweight="bold")
        return fig, axs

    def _set_axes_properties(self):
        """Set properties for all axes."""
        for row in self.axes:
            for ax in row:
                ax.set_box_aspect(0.5)
                ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
                ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))

        for col, processor in enumerate(self.processors):
            self.axes[0,col].set_title(f"{processor.start_date}:{processor.end_date}", fontsize=10)

    def get_axes(self):
        return self.fig, self.axes