import math


class ConstantAnnealing:
    def __init__(self, value):
        self.value = value

    def get_value(self, step):
        return self.value


class FixedIntervalDownAnnealing:
    def __init__(self, initial_value, min_value, interval, rate):
        self.initial_value = initial_value
        self.min_value = min_value
        self.interval = interval

        assert 0 < rate < 1, "'rate' must be in the range [0, 1]. Found {}!".format(rate)
        self.rate = rate

    def get_value(self, step):
        num_intervals = step // self.interval
        value = max(self.initial_value * (self.rate ** num_intervals), self.min_value)
        return value


class MultiStepLinearAnnealing:
    def __init__(self, anneal_values, anneal_points):
        """
        :param anneal_values: Values that will be set at particular time steps
        :param anneal_points: Time steps corresponding to those values
        """
        assert isinstance(anneal_values, (list, tuple)), type(anneal_values)
        assert isinstance(anneal_points, (list, tuple)), type(anneal_points)
        assert len(anneal_values) == len(anneal_points), "'anneal_values' and " \
            "'anneal_points' must have the same length!"

        self.anneal_values = anneal_values
        self.anneal_points = anneal_points

        self.slopes = []
        self.intercepts = []
        for i in range(0, len(anneal_points)-1):
            x1 = anneal_points[i]
            x2 = anneal_points[i+1]
            # assert x2 > x1, "The 'anneal_points[{}]' must be greater than the " \
            #     "'anneal_points[{}]'. Found {} and {} instead!".format(i, i+1, x1, x2)

            y1 = anneal_values[i]
            y2 = anneal_values[i+1]

            slope, intercept = self._solve_linear_eq(x1, x2, y1, y2)

            self.slopes.append(slope)
            self.intercepts.append(intercept)
        assert len(self.slopes) == len(self.intercepts) == len(anneal_points) - 1

    @staticmethod
    def _solve_linear_eq(x1, x2, y1, y2):
        """
        Solve the linear equation
        a * x1 + b = y1
        a * x2 + b = y2
        => a = (y2 - y1)/(x2 - x1)
        => b = y1 - a * x1
        """

        assert x1 != x2, "'x1' and 'x2' must be different! Found {} and {}!".format(x1, x2)
        slope = (y2 - y1)/(x2 - x1)
        intercept = y1 - slope * x1
        return slope, intercept

    def get_value(self, step):
        if step < self.anneal_points[0]:
            return self.anneal_values[0]
        elif self.anneal_points[0] <= step < self.anneal_points[-1]:
            for i in range(0, len(self.anneal_points)-1):
                if self.anneal_points[i] <= step < self.anneal_points[i+1]:
                    return self.slopes[i] * step + self.intercepts[i]
        else:
            return self.anneal_values[-1]


class ScaledSigmoidAnnealing:
    # Annealing using the function (1 / e^(-x) + 1)
    def __init__(self, start_point, end_point, scale=1):
        # We want to draw a sigmoid function such that [start, end] corresponds to [-6, 6]

        assert 0 < start_point < end_point, \
            "start_point={} and end_point={}".format(start_point, end_point)
        self.start_point = start_point
        self.end_point = end_point

        assert scale >= 0, "scale={}".format(scale)
        self.scale = scale

    def _map_point(self, p):
        assert self.start_point <= p <= self.end_point, "p={}".format(p)
        return (p - self.start_point) / (self.end_point - self.start_point) * 12 - 6

    def get_value(self, step):
        if step < self.start_point:
            return 0
        elif step > self.end_point:
            return self.scale
        else:
            x = self._map_point(step)
            return self.scale * 1.0 / (math.exp(-x) + 1)


class StepAnnealing:
    def __init__(self, change_point, value_0, value_1):
        self.change_point = change_point
        self.value_0 = value_0
        self.value_1 = value_1

    def get_value(self, step):
        if step < self.change_point:
            return self.value_0
        else:
            return self.value_1


class LinearWithClipAnnealing:
    def __init__(self, change_point, value_0, value_1):
        self.change_point = change_point
        self.value_0 = value_0
        self.value_1 = value_1

    def get_value(self, step):
        if step < self.change_point:
            return self.value_0
        else:
            return self.value_1


# From MeanTeacher
class SigmoidRampup:
    def __init__(self, length, start_point=0):
        assert start_point >= 0, "'start_point' must be >= 0!"
        assert length >= 0, "'length' must be >= 0!"

        self.start_point = start_point
        self.length = length
        self.end_point = start_point + length

    def get_value(self, step):
        if self.length == 0:
            return 1.0
        else:
            if step < self.start_point:
                return 0.0
            elif self.start_point <= step < self.end_point:
                phase = 1.0 - (1.0 * max(0.0, step - self.start_point) / self.length)
                return math.exp(-5.0 * phase * phase)
            else:
                return 1.0


class SigmoidRampdown:
    def __init__(self, length, end_point):
        assert length >= 0, "'length' must be >= 0!"
        assert end_point >= length, "'end_point' must be >= 'length'!"

        self.length = length
        self.end_point = end_point
        self.start_point = end_point - length

    def get_value(self, step):
        if self.length == 0:
            return 1.0
        else:
            if step < self.start_point:
                return 1.0
            elif self.start_point <= step < self.end_point:
                phase = 1.0 - (1.0 * max(0.0, self.end_point - step) / self.length)
                return math.exp(-12.5 * phase * phase)
            else:
                return 0.0

    def __call__(self, step):
        return self.get_value(step)


class LinearRampup:
    def __init__(self, length, start_point=0):
        assert start_point >= 0, "'start_point' must be >= 0!"
        assert length >= 0, "'length' must be >= 0!"

        self.start_point = start_point
        self.length = length
        self.end_point = start_point + length
        
    def get_value(self, step):
        if self.length == 0:
            return 1.0
        else:
            if step < self.start_point:
                return 0.0
            elif self.start_point <= step < self.end_point:
                return (step - self.start_point) * 1.0 / self.length
            else:
                return 1.0

    def __call__(self, step):
        return self.get_value(step)


class CosineRampdown:
    def __init__(self, length, end_point):
        assert length >= 0, "'length' must be >= 0!"
        assert end_point >= length, "'end_point' must be >= 'length'!"

        self.length = length
        self.end_point = end_point
        self.start_point = end_point - length

    def get_value(self, step):
        if self.length == 0:
            return 1.0
        else:
            if step < self.start_point:
                return 1.0
            elif self.start_point <= step < self.end_point:
                return 0.5 * (math.cos(math.pi * (step - self.start_point) / self.length) + 1)
            else:
                return 0.0