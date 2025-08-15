import unittest
from plotdata.objects.plotters import VelocityPlot

class TestVelocityPlot(unittest.TestCase):
    def test_velocity_plot(self):
            pass
            plot = VelocityPlot()
            self.assertIsNotNone(plot)

if __name__ == "__main__":
    unittest.main()