import csv

import pytest

from src.satin import gaussian_calculation, Gaussian


def read_csv(file_path):
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return [dict(row) for row in reader]


@pytest.mark.parametrize(
    "input_power, small_signal_gain, saturation_intensity, output_power, log_output_power_divided_by_input_power, output_power_minus_input_power",
    read_csv('satin.csv')
)
def test_gaussian_calculation(input_power, small_signal_gain, saturation_intensity, output_power,
                              log_output_power_divided_by_input_power, output_power_minus_input_power):
    input_power = int(input_power)
    small_signal_gain = float(small_signal_gain)
    saturation_intensity = int(saturation_intensity)
    output_power = float(output_power)
    log_output_power_divided_by_input_power = float(log_output_power_divided_by_input_power)
    output_power_minus_input_power = float(output_power_minus_input_power)

    gaussians = gaussian_calculation(input_power, small_signal_gain)
    for gaussian in gaussians:
        if gaussian.saturation_intensity == saturation_intensity:
            assert Gaussian.round_up(gaussian.output_power) == output_power
            assert gaussian.log_output_power_divided_by_input_power() == log_output_power_divided_by_input_power
            assert gaussian.output_power_minus_input_power() == output_power_minus_input_power
        else:
            pass
