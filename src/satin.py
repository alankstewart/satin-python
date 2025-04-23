import datetime
import logging
import math
import multiprocessing
import re
import textwrap
from collections import namedtuple
from concurrent.futures import ALL_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from functools import reduce

PI = math.pi
RAD = 0.18
RAD2 = RAD ** 2
W1 = 0.3
DR = 0.002
DZ = 0.04
LAMBDA = 0.0106
AREA = PI * RAD2
Z1 = PI * W1 ** 2 / LAMBDA
Z12 = Z1 ** 2
EXPR = 2 * PI * DR
INCR = 8001
EXPR1 = [
    2 * ((i - INCR // 2) / 25) * DZ / (Z12 + ((i - INCR // 2) / 25) ** 2)
    for i in range(INCR)
]

LASER_FILE = 'laser.dat'
PIN_FILE = 'pin.dat'

Laser = namedtuple('Laser', 'output_file small_signal_gain discharge_pressure carbon_dioxide')
Gaussian = namedtuple('Gaussian', 'input_power output_power saturation_intensity')


class Satin:
    """
    Satin class for handling the processing of laser data using parallel execution.
    """

    @staticmethod
    def main():
        """
        Main method to start the laser data processing.
        """
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        _calculate()


def _calculate():
    """
    Perform the main calculation of the laser data processing. Reads laser data, processes
    each laser entry, and calculates Gaussian output for each power input.
    """
    start = datetime.datetime.now().timestamp()

    # Read and process laser data from file
    with open(LASER_FILE, encoding='utf-8') as laser_file:
        input_powers = _get_input_powers()
        laser_data = laser_file.read()
        laser_matches = re.findall(r'((?:md|pi)[a-z]{2}\.out)\s+(\d{2}\.\d)\s+(\d+)\s+(MD|PI)', laser_data)

        # Use threads for processing each laser entry
        with ThreadPoolExecutor() as executor:
            tasks = [
                executor.submit(_process, input_powers, Laser(laser[0], float(laser[1]), int(laser[2]), laser[3]))
                for laser in laser_matches
            ]
            wait(tasks, return_when=ALL_COMPLETED)

    logging.info('The time was %.3f seconds', datetime.datetime.now().timestamp() - start)


def _process(input_powers, laser):
    """
    Process the data for a given laser and write the results to a file.

    Args:
        input_powers (list): List of input power values.
        laser (Laser): The Laser namedtuple containing laser data.

    Returns:
        str: The name of the generated output file.
    """
    with open(f'{laser.output_file}', 'w', encoding='utf-8') as file:
        file.write(f'Start date: {datetime.datetime.now().isoformat()}\n')
        file.write(textwrap.dedent(f'''
            Gaussian Beam

            Pressure in Main Discharge = {laser.discharge_pressure}kPa
            Small-signal Gain = {laser.small_signal_gain}
            CO2 via {laser.carbon_dioxide}

            Pin       Pout                 Sat. Int      ln(Pout/Pin)   Pout-Pin
            (watts)   (watts)              (watts/cm2)                  (watts)
        '''))

        lines = [
            f'{gaussian.input_power:<10}'
            f'{gaussian.output_power:<21.14f}'
            f'{gaussian.saturation_intensity:<14}'
            f'{math.log(gaussian.output_power / gaussian.input_power):>5.3f}'
            f'{gaussian.output_power - gaussian.input_power:>16.3f}\n'
            for input_power in input_powers
            for gaussian in gaussian_calculation(input_power, laser.small_signal_gain)
        ]
        file.writelines(lines)

        file.write(f'\nEnd date: {datetime.datetime.now().isoformat()}')
        file.flush()

    return file.name


def _get_input_powers():
    """
    Get the list of input powers from the PIN_FILE.

    Returns:
        list: A list of input power values extracted from the PIN_FILE.
    """
    with open(PIN_FILE, encoding='utf-8') as pin_file:
        return [int(match.group()) for match in re.finditer(r'\d+', pin_file.read())]


def gaussian_calculation(input_power, small_signal_gain):
    """
    Calculate the output power for each saturation intensity in the Gaussian range.

    Args:
        input_power (int): The input power for the Gaussian calculation.
        small_signal_gain (float): The small signal gain for the Gaussian calculation.

    Returns:
        list: A list of Gaussian namedtuples containing input power, output power, and saturation intensity.
    """
    saturation_intensities = range(10000, 25001, 1000)

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [
            executor.submit(_calculate_output_power, input_power, small_signal_gain, saturation_intensity)
            for saturation_intensity in saturation_intensities
        ]
        wait(futures, return_when=ALL_COMPLETED)
        return [Gaussian(input_power, future.result(), saturation_intensity) for future, saturation_intensity in
                zip(futures, saturation_intensities)]


def _calculate_output_power(input_power, small_signal_gain, saturation_intensity):
    """
    Calculate the output power based on Gaussian beam propagation and saturation intensity.

    Args:
        input_power (int): The input power for the calculation.
        small_signal_gain (float): The small signal gain for the calculation.
        saturation_intensity (int): The saturation intensity for the calculation.

    Returns:
        float: The calculated output power.
    """
    input_intensity = 2 * input_power / AREA
    expr2 = saturation_intensity * small_signal_gain / 32000 * DZ
    return sum(
        (
            reduce(
                lambda output_intensity, j: output_intensity * (
                        1 + expr2 / (saturation_intensity + output_intensity) - EXPR1[j]
                ), range(INCR), input_intensity * math.exp(-2 * r ** 2 / RAD2),
            ) * EXPR * r for r in (i * DR for i in range(int(0.5 / DR)))
        )
    )


if __name__ == '__main__':
    Satin.main()
