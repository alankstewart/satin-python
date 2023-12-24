import csv
import os

import pytest

from src.satin import gaussian_calculation


def _read_csv(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)


def _round_up(value):
    return round(value * 1000.0) / 1000.0


script_directory = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(script_directory, 'satin.csv')


@pytest.mark.parametrize(
    'input_power, small_signal_gain, saturation_intensity, output_power, '
    'log_output_power_divided_by_input_power, output_power_minus_input_power',
    _read_csv(csv_file_path)
)
def test_gaussian_calculation(input_power, small_signal_gain, saturation_intensity, output_power,
                              log_output_power_divided_by_input_power, output_power_minus_input_power):
    input_power = int(input_power)
    small_signal_gain = float(small_signal_gain)
    saturation_intensity = int(saturation_intensity)
    output_power = float(output_power)
    log_output_power_divided_by_input_power = float(log_output_power_divided_by_input_power)
    output_power_minus_input_power = float(output_power_minus_input_power)

    for gaussian in gaussian_calculation(input_power, small_signal_gain):
        if gaussian.saturation_intensity == saturation_intensity:
            assert _round_up(gaussian.output_power) == output_power
            assert _round_up(
                gaussian.log_output_power_divided_by_input_power()) == log_output_power_divided_by_input_power
            assert _round_up(gaussian.output_power_minus_input_power()) == output_power_minus_input_power
        else:
            pass
