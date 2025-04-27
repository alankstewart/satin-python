"""
satin.py
"""
import datetime
import logging
import math
import multiprocessing
import re
import textwrap
from concurrent.futures import ALL_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

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
            laser_matches = re.findall(
                r'((?:md|pi)[a-z]{2}\.out)\s+(\d{2}\.\d)\s+(\d+)\s+(MD|PI)', laser_data
            )

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
        for g in gaussian_calculation(input_powers, laser.small_signal_gain)
    )

    footer = f"\nEnd date: {datetime.datetime.now().isoformat()}"

    output_path.write_text(header + gaussian_lines + footer, encoding='utf-8')
    return output_path


def gaussian_calculation(input_powers, small_signal_gain):
    """
    Calculates Gaussian results in parallel.
    """
    saturation_intensities = list(range(10000, 25001, 1000))
    tasks = [
        (input_power, small_signal_gain, sat_intensity)
        for input_power in input_powers
        for sat_intensity in saturation_intensities
    ]

    with ProcessPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        futures = [executor.submit(_calculate_output_power, *task) for task in tasks]
        wait(futures, return_when=ALL_COMPLETED)

    return [
        Gaussian(input_power, future.result(), sat_intensity)
        for (input_power, _, sat_intensity), future in zip(tasks, futures)
    ]


def _calculate_output_power(input_power, small_signal_gain, saturation_intensity):
    """
    Calculates output power.
    """
    input_intensity = 2 * input_power / AREA
    expr2 = saturation_intensity * small_signal_gain / 32000 * DZ
    r_values = [i * DR for i in range(int(0.5 / DR))]
    exp_values = [input_intensity * math.exp(-2 * r ** 2 / RAD2) for r in r_values]

    total = 0.0
    for r, initial_intensity in zip(r_values, exp_values):
        output_intensity = initial_intensity
        for j in range(INCR):
            output_intensity *= (
                    1 + expr2 / (saturation_intensity + output_intensity) - EXPR1[j]
            )
        total += output_intensity * EXPR * r

    return total


if __name__ == '__main__':
    Satin.main()
