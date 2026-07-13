import os
import tempfile
import unittest

from plotdata.helper_functions import (
    append_project_command_log,
    build_plot_output_basename,
    format_section_header_suffix,
    write_vectors_profile_txt,
)


class TestPlotOutputHelpers(unittest.TestCase):
    def test_build_plot_output_basename_without_tag(self):
        stem = build_plot_output_basename('Etna', '', 'vectors', '20141020', '20260626')
        self.assertEqual(stem, 'Etna_vectors_20141020_20260626')

    def test_build_plot_output_basename_with_tag(self):
        stem = build_plot_output_basename('Etna', 'Pernicarna', 'vectors', '20141020', '20260626')
        self.assertEqual(stem, 'Etna_Pernicarna_vectors_20141020_20260626')

    def test_format_section_header_suffix_prefers_raw_string(self):
        section = format_section_header_suffix(
            [[15.05101, 15.19727], [37.73522, 37.73522]],
            '37.73522:15.05101,37.73522:15.19727',
        )
        self.assertEqual(section, '37.73522:15.05101,37.73522:15.19727')

    def test_write_vectors_profile_txt(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, 'Etna_vectors_20141020_20260626.txt')
            rows = [(37.73522, 15.05101, 1200.0, 1.2, -0.5, 0.0)]
            write_vectors_profile_txt(out_path, rows, '37.73522:15.05101,37.73522:15.19727')
            with open(out_path, encoding='utf-8') as f:
                lines = f.read().splitlines()
            self.assertEqual(
                lines[0],
                'lat lon elevation_m horz vert distance_km [37.73522:15.05101,37.73522:15.19727]',
            )
            self.assertTrue(lines[1].startswith('37.73522000 15.05101000 1200.000'))

    def test_append_project_command_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            append_project_command_log(tmp, 'plot_data.py Etna/mintpy --tag Pernicarna')
            log_path = os.path.join(tmp, 'log')
            self.assertTrue(os.path.isfile(log_path))
            with open(log_path, encoding='utf-8') as f:
                line = f.read().strip()
            self.assertIn('plot_data.py Etna/mintpy --tag Pernicarna', line)
            self.assertRegex(line, r'^\d{8}-\d{2}:\d{2} \+ ')


if __name__ == '__main__':
    unittest.main()
