import datetime
import math
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import reduce

PI = math.pi


class Satin:
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

    def main(self):
        self.calculate()

    def calculate(self):
        start = datetime.datetime.now().timestamp()

        with open('laser.dat', encoding='utf-8') as laser_file:
            input_powers = self.get_input_powers()
            laser_data = laser_file.read()
            laser_matches = re.findall(r'((?:md|pi)[a-z]{2}\.out)\s+(\d{2}\.\d)\s+(\d+)\s+(MD|PI)', laser_data)

            with ProcessPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(self.process, input_powers, Laser(*laser)) for laser in laser_matches]
                for future in as_completed(futures):
                    future.result()

        print(f'The time was {datetime.datetime.now().timestamp() - start} seconds')

    def get_input_powers(self):
        with open('pin.dat', encoding='utf-8') as pin_file:
            return [int(match.group()) for match in re.finditer(r'\d+', pin_file.read())]

    def process(self, input_powers, laser):
        with open(f'{laser.output_file}', 'w', encoding='utf-8') as file:
            file.write(
                f'Start date: {datetime.datetime.now().isoformat()}\n\n'
                f'Gaussian Beam\n\nPressure in Main Discharge = {laser.discharge_pressure}kPa\n'
                f'Small-signal Gain = {laser.small_signal_gain}\nCO2 via {laser.carbon_dioxide}\n\n'
            )

            table_header = '{:<7}  {:<19}  {:<12}  {:<13}  {:<8}\n'
            file.write(table_header.format('Pin', 'Pout', 'Sat. Int', 'ln(Pout/Pin)', 'Pout-Pin'))
            file.write(table_header.format('(watts)', '(watts)', '(watts/cm2)', '', '(watts)'))

            for input_power in input_powers:
                for gaussian in self.gaussian_calculation(input_power, laser.small_signal_gain):
                    file.write(
                        '{:<7}  {:<19}  {:<12}  {:>12.3f}  {:>9.3f}\n'.format(
                            gaussian.input_power(),
                            gaussian.output_power(),
                            gaussian.saturation_intensity(),
                            gaussian.log_output_power_divided_by_input_power(),
                            gaussian.output_power_minus_input_power()
                        )
                    )

            file.write('\nEnd date: {}\n'.format(datetime.datetime.now().isoformat()))
            file.flush()

        return file.name

    def gaussian_calculation(self, input_power, small_signal_gain):
        return [self.create_gaussian(input_power, small_signal_gain, saturation_intensity)
                for saturation_intensity in range(10000, 25001, 1000)]

    def create_gaussian(self, input_power, small_signal_gain, saturation_intensity):
        output_power = self.calculate_output_power(input_power, small_signal_gain, saturation_intensity)
        return Gaussian(input_power, output_power, saturation_intensity)

    def calculate_output_power(self, input_power, small_signal_gain, saturation_intensity):
        expr1 = [
            2 * ((i - self.INCR // 2) / 25) * self.DZ / (self.Z12 + ((i - self.INCR // 2) / 25) ** 2)
            for i in range(self.INCR)
        ]

        input_intensity = 2 * input_power / self.AREA
        return sum(
            (
                (
                    (
                        (
                            reduce(
                                lambda output_intensity, j: output_intensity * (
                                    1 + (saturation_intensity * small_signal_gain / 32000 * self.DZ)
                                    / (saturation_intensity + output_intensity) - expr1[j]
                                ),
                                range(self.INCR),
                                input_intensity * math.exp(-2 * r ** 2 / self.RAD2),
                            )
                        ) * self.EXPR * r
                    )
                    for r in (i * self.DR for i in range(int(0.5 / self.DR)))
                )
            )
        )


class Laser:
    def __init__(self, output_file, small_signal_gain, discharge_pressure, carbon_dioxide):
        self.output_file = output_file
        self.small_signal_gain = float(small_signal_gain)
        self.discharge_pressure = int(discharge_pressure)
        self.carbon_dioxide = carbon_dioxide


class Gaussian:
    def __init__(self, input_power, output_power, saturation_intensity):
        self.input_power = lambda: input_power
        self.output_power = lambda: output_power
        self.saturation_intensity = lambda: saturation_intensity

    def log_output_power_divided_by_input_power(self):
        return self.round_up(math.log(self.output_power() / self.input_power()))

    def output_power_minus_input_power(self):
        return self.round_up(self.output_power() - self.input_power())

    def round_up(self, value):
        return round(value * 1000.0) / 1000.0


if __name__ == '__main__':
    Satin().main()
