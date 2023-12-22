import datetime
import math
import multiprocessing
import re
import textwrap
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, wait, ALL_COMPLETED
from dataclasses import dataclass
from typing import List

# Constants and Configuration
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


class Satin:
    @staticmethod
    def main():
        start = datetime.datetime.now().timestamp()

        with open(LASER_FILE, encoding='utf-8') as laser_file:
            input_powers = _get_input_powers()
            laser_data = laser_file.read()
            laser_matches = re.findall(r'((?:md|pi)[a-z]{2}\.out)\s+(\d{2}\.\d)\s+(\d+)\s+(MD|PI)', laser_data)
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(_process, input_powers, Laser(laser[0], float(laser[1]), int(laser[2]), laser[3]))
                    for laser in laser_matches]
                wait(futures, return_when=ALL_COMPLETED)

        print(f'The time was {datetime.datetime.now().timestamp() - start:.3f} seconds')


@dataclass(frozen=True)
class Laser:
    output_file: str
    small_signal_gain: float
    discharge_pressure: int
    carbon_dioxide: str


@dataclass(frozen=True, order=True)
class Gaussian:
    input_power: float
    output_power: float
    saturation_intensity: int

    def log_output_power_divided_by_input_power(self):
        return math.log(self.output_power / self.input_power)

    def output_power_minus_input_power(self):
        return self.output_power - self.input_power


def _process(input_powers, laser):
    with open(f'{laser.output_file}', 'w', encoding='utf-8') as file:
        file.write(f'Start date: {datetime.datetime.now().isoformat()}\n')
        file.write(textwrap.dedent(f'''
            Gaussian Beam

            Pressure in Main Discharge = {laser.discharge_pressure}kPa
            Small-signal Gain = {laser.small_signal_gain}
            CO2 via {laser.carbon_dioxide}

            {'Pin':<7}  {'Pout':<19}  {'Sat. Int':<12}  {'ln(Pout/Pin)':<13}  {'Pout-Pin':<8}
            {'(watts)':<7}  {'(watts)':<19}  {'(watts/cm2)':<12}  {'':<13}   {'(watts)':<8}
        '''))

        lines = [
            f'{gaussian.input_power:>7}  '
            f'{gaussian.output_power:<19}  '
            f'{gaussian.saturation_intensity:<12}  '
            f'{gaussian.log_output_power_divided_by_input_power():>12.3f}  '
            f'{gaussian.output_power_minus_input_power():>9.3f}\n'
            for input_power in input_powers
            for gaussian in gaussian_calculation(input_power, laser.small_signal_gain)
        ]
        file.writelines(lines)

        file.write(f'\nEnd date: {datetime.datetime.now().isoformat()}')
        file.flush()

    return file.name


def _get_input_powers():
    with open(PIN_FILE, encoding='utf-8') as pin_file:
        return [int(match.group()) for match in re.finditer(r'\d+', pin_file.read())]


def gaussian_calculation(input_power, small_signal_gain) -> List[Gaussian]:
    saturation_intensities = range(10000, 25001, 1000)
    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(_create_gaussian, input_power, small_signal_gain, saturation_intensity) for
                   saturation_intensity in saturation_intensities]
        wait(futures, return_when=ALL_COMPLETED)
        return [future.result() for future in futures]


def _create_gaussian(input_power, small_signal_gain, saturation_intensity):
    output_power = _calculate_output_power(input_power, small_signal_gain, saturation_intensity)
    return Gaussian(input_power, output_power, saturation_intensity)


# def _calculate_output_power(input_power, small_signal_gain, saturation_intensity):
#     input_intensity = 2 * input_power / AREA
#     expr2 = saturation_intensity * small_signal_gain / 32000 * DZ
#
#     def inner_sum(output_intensity, j):
#         return output_intensity * (1 + expr2 / (saturation_intensity + output_intensity) - EXPR1[j])
#
#     def outer_sum(r):
#         from functools import reduce
#         return reduce(inner_sum, range(INCR), input_intensity * math.exp(-2 * r ** 2 / RAD2)) * EXPR * r
#
#     return sum(outer_sum(i * DR) for i in range(int(0.5 / DR)))


def _calculate_output_power(input_power, small_signal_gain, saturation_intensity):
    input_intensity = 2 * input_power / AREA
    expr2 = saturation_intensity * small_signal_gain / 32000 * DZ

    output_power = 0.0
    for r in (i * DR for i in range(int(0.5 / DR))):
        output_intensity = input_intensity * math.exp(-2 * r ** 2 / RAD2)
        for j in range(INCR):
            output_intensity *= (1 + expr2 / (saturation_intensity + output_intensity) - EXPR1[j])

        output_power += output_intensity * EXPR * r

    return output_power


if __name__ == '__main__':
    Satin.main()
