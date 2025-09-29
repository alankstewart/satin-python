"""
test_satin.py
"""
import csv
import os
from math import log
from functools import lru_cache
import pytest
from _pytest.python_api import approx

from src.satin import gaussian_calculation


def _read_csv(file_path):
    """
    Reads a CSV file and returns its content as a list of dictionaries.
    """
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        return [(
            int(row['input_power']),
            float(row['small_signal_gain']),
            int(row['saturation_intensity']),
            float(row['output_power']),
            float(row['log_output_power_divided_by_input_power']),
            float(row['output_power_minus_input_power'])
        ) for row in reader]


script_directory = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(script_directory, 'satin-all.csv')


@lru_cache(maxsize=None)
def get_gaussians(input_power, small_signal_gain):
    """
    Cached lookup for gaussian_calculation results.
    """
    return {
        g.saturation_intensity: g
        for g in gaussian_calculation([input_power], small_signal_gain)
    }


@pytest.mark.parametrize("params", _read_csv(csv_file_path))
def test_gaussian_calculation(params):
    """
    Test the gaussian calculation function with parameters from the CSV file.
    """
    (input_power,
     small_signal_gain,
     saturation_intensity,
     output_power,
     log_output_power_divided_by_input_power,
     output_power_minus_input_power) = params

    gaussians = get_gaussians(input_power, small_signal_gain)
    gaussian = gaussians.get(saturation_intensity)
    assert gaussian is not None, f"No Gaussian for saturation_intensity={saturation_intensity}"

    assert gaussian.output_power == approx(output_power, abs=5e-4)
    assert (log(gaussian.output_power / gaussian.input_power) ==
            approx(log_output_power_divided_by_input_power, abs=5e-4))
    assert (gaussian.output_power - gaussian.input_power ==
            approx(output_power_minus_input_power, abs=5e-4))
