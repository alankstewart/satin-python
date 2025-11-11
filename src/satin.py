"""
satin.py
"""
import datetime
import logging
import math
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
_indices = np.arange(INCR)
EXPR1 = 2 * ((_indices - INCR // 2) / 25.0) * DZ / (Z12 + ((_indices - INCR // 2) / 25.0) ** 2)

LASER_FILE = 'laser.dat'
PIN_FILE = 'pin.dat'


@dataclass
class Laser:
    """
    Container for laser discharge properties.
    """
    output_file: str
    small_signal_gain: float
    discharge_pressure: int
    carbon_dioxide: str


@dataclass
class Gaussian:
    """
    Gaussian beam properties.
    """
    input_power: int
    output_power: float
    saturation_intensity: float

    @property
    def log_output_power_divided_by_input_power(self):
        """
        Natural log of output power divided by input power (ln(Pout / Pin)).
        """
        return math.log(self.output_power / self.input_power)

    @property
    def output_power_minus_input_power(self):
        """
        Difference between output power and input power (Pout - Pin).
        """
        return self.output_power - self.input_power

    def __str__(self):
        return (
            f'{self.input_power:<10}'
            f'{self.output_power:<21.14f}'
            f'{self.saturation_intensity:<14}'
            f'{self.log_output_power_divided_by_input_power:>5.3f}'
            f'{self.output_power_minus_input_power:>16.3f}\n'
        )


class Satin:
    """
    The Satin class handles laser beam calculations, including reading data from files,
    performing Gaussian beam computations, and writing results to output files.
    """

    @staticmethod
    def main():
        """
        Main method to configure logging and invoke the calculation process.
        """
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        Satin.calculate()

    @staticmethod
    def calculate():
        """
        Performs the main calculation process by reading laser data, calculating Gaussian beam
        properties, and saving the results to output files. Logs the output file paths.
        """
        start = datetime.datetime.now().timestamp()

        with open(LASER_FILE, encoding='utf-8') as laser_file:
            input_powers = _get_input_powers()
            laser_data = laser_file.read()
            laser_matches = re.findall(r'((?:md|pi)[a-z]{2}\.out)\s+(\d{2}\.\d)\s+(\d+)\s+(MD|PI)',
                                       laser_data)

            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        _process,
                        input_powers,
                        Laser(output_file=laser[0],
                              small_signal_gain=float(laser[1]),
                              discharge_pressure=int(laser[2]),
                              carbon_dioxide=laser[3])
                    ): laser[0]
                    for laser in laser_matches
                }

                for future in futures:
                    try:
                        result_path = future.result()
                        logging.debug("Successfully created %s", result_path)
                    except (RuntimeError, IOError, ValueError) as e:
                        logging.error("Error processing %s: %s", futures[future], e)

        logging.info('The time was %.3f seconds', datetime.datetime.now().timestamp() - start)


def _get_input_powers():
    """
    Reads the input powers from the pin.dat file.
    """
    with open(PIN_FILE, encoding='utf-8') as pin_file:
        return [int(match.group()) for match in re.finditer(r'\d+', pin_file.read())]


def _process(input_powers, laser):
    """
    Processes each laser entry, performs the calculations, and writes the results to an output file.
    Returns the Path to the output file.
    """
    output_path = Path(laser.output_file)

    header = textwrap.dedent(f"""\
        Start date: {datetime.datetime.now().isoformat()}

        Gaussian Beam

        Pressure in Main Discharge = {laser.discharge_pressure}kPa
        Small-signal Gain = {laser.small_signal_gain}
        CO2 via {laser.carbon_dioxide}

        Pin       Pout                 Sat. Int      ln(Pout/Pin)   Pout-Pin
        (watts)   (watts)              (watts/cm2)                  (watts)
    """)

    gaussian_lines = ''.join(
        str(g)
        for input_power in input_powers
        for g in gaussian_calculation(input_power, laser.small_signal_gain)
    )

    footer = f"\nEnd date: {datetime.datetime.now().isoformat()}"

    output_path.write_text(header + gaussian_lines + footer, encoding='utf-8')
    return output_path


def gaussian_calculation(input_power, small_signal_gain):
    """
    Vectorised Gaussian results for a single input_power and small_signal_gain.
    """
    saturation_intensities = np.arange(10000, 25001, 1000)

    nr = int(0.5 / DR)
    r_values = np.arange(nr) * DR
    input_intensity = 2.0 * float(input_power) / AREA
    exp_values = input_intensity * np.exp(-2.0 * r_values ** 2 / RAD2)

    n_sat = saturation_intensities.size
    output_intensity = np.broadcast_to(exp_values[:, None], (nr, n_sat)).copy()

    sat = saturation_intensities.astype(float)
    expr2 = sat * float(small_signal_gain) / 32000.0 * DZ

    for j in range(INCR):
        multiplier = 1.0 + (expr2[None, :] / (sat[None, :] + output_intensity)) - EXPR1[j]
        output_intensity *= multiplier

    integrand = output_intensity * (EXPR * r_values[:, None])
    output_power_per_sat = integrand.sum(axis=0)

    return [
        Gaussian(input_power, float(output_power_per_sat[i]), int(saturation_intensities[i]))
        for i in range(n_sat)
    ]


if __name__ == '__main__':
    Satin.main()
