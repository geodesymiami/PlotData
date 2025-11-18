import pytest
from plotdata.objects.plotters import VelocityPlot

def test_velocity_plot():
    plot = VelocityPlot()
    assert plot is not None

if __name__ == "__main__":
    pytest.main()