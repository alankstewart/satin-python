"""
test_satin.py
"""
import csv
import os
from math import log
import pytest
from src.satin import gaussian_calculation


def _read_csv(file_path):
    """
    Reads a CSV file and returns its content as a list of dictionaries.
    """
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        return [(
            row['input_power'],
            row['small_signal_gain'],
            row['saturation_intensity'],
            row['output_power'],
            row['log_output_power_divided_by_input_power'],
            row['output_power_minus_input_power']
        ) for row in reader]


def _round_up(value):
    """
    Rounds up the value to three decimal places.
    """
    return round(value * 1000.0) / 1000.0


# Get the script directory and build the path to the CSV file
script_directory = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(script_directory, 'satin.csv')


@pytest.mark.parametrize(
    'params',
    _read_csv(csv_file_path)
)
def test_gaussian_calculation(params):
    """
    Test the gaussian calculation function with parameters from the CSV file.
    """
    input_power, small_signal_gain, saturation_intensity, output_power, log_output_power_divided_by_input_power, output_power_minus_input_power = params

    # Convert types where necessary
    input_power = int(input_power)
    small_signal_gain = float(small_signal_gain)

    for gaussian in gaussian_calculation(input_power, small_signal_gain):
        if gaussian.saturation_intensity == int(saturation_intensity):
            assert _round_up(gaussian.output_power) == float(output_power)
            assert (_round_up(log(gaussian.output_power / gaussian.input_power)) ==
                    float(log_output_power_divided_by_input_power))
            assert (_round_up(gaussian.output_power - gaussian.input_power) ==
                    float(output_power_minus_input_power))
