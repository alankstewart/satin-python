"""
satin.py
"""
import datetime
import logging
import math
import re
import textwrap
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
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
NR = int(0.5 / DR)
R_VALUES = np.arange(NR) * DR
EXP_PROFILE = np.exp(-2.0 * R_VALUES ** 2 / RAD2)
INTEGRATION_FACTOR = EXPR * R_VALUES[:, None]

LASER_FILE = 'laser.dat'
PIN_FILE = 'pin.dat'


@dataclass
class Laser:
    output_file: str
    small_signal_gain: float
    discharge_pressure: int
    carbon_dioxide: str


@dataclass
class Gaussian:
    input_power: int
    output_power: float
    saturation_intensity: float

    @property
    def log_output_power_divided_by_input_power(self):
        return math.log(self.output_power / self.input_power)

    @property
    def output_power_minus_input_power(self):
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
    @staticmethod
    def main():
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        Satin.calculate()

    @staticmethod
    def calculate():
        start = datetime.datetime.now().timestamp()

        with open(LASER_FILE, encoding='utf-8') as laser_file:
            input_powers = _get_input_powers()
            laser_data = laser_file.read()
            laser_matches = re.findall(
                r'((?:md|pi)[a-z]{2}\.out)\s+(\d{2}\.\d)\s+(\d+)\s+(MD|PI)',
                laser_data
            )

            with ProcessPoolExecutor() as executor:
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
    with open(PIN_FILE, encoding='utf-8') as pin_file:
        return [int(match.group()) for match in re.finditer(r'\d+', pin_file.read())]


def _process(input_powers, laser):
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


@lru_cache(maxsize=None)
def _sat_and_expr2(small_signal_gain):
    saturation_intensities = np.arange(10000, 25001, 1000)
    expr2 = saturation_intensities * float(small_signal_gain) / 32000.0 * DZ
    return saturation_intensities, expr2


def gaussian_calculation(input_power, small_signal_gain):
    saturation_intensities, expr2 = _sat_and_expr2(small_signal_gain)

    input_intensity = 2.0 * float(input_power) / AREA
    exp_values = input_intensity * EXP_PROFILE

    n_sat = saturation_intensities.size
    output_intensity = np.broadcast_to(exp_values[:, None], (NR, n_sat)).copy()

    for j in range(INCR):
        multiplier = (
                1.0
                + (expr2[None, :] / (saturation_intensities[None, :] + output_intensity))
                - EXPR1[j]
        )
        output_intensity *= multiplier

    integrand = output_intensity * INTEGRATION_FACTOR
    output_power_per_sat = integrand.sum(axis=0)

    return [
        Gaussian(input_power, float(output_power_per_sat[i]), int(saturation_intensities[i]))
        for i in range(n_sat)
    ]


if __name__ == '__main__':
    Satin.main()
