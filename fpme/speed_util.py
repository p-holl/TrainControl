import numpy as np


def speeds(s15, exponent=1.3):
    return (*np.linspace(0, s15 ** (1/exponent), 15) ** exponent,)


def fit_speeds(measured: tuple):
    import scipy
    def loss(x):
        max_speed, exponent = x
        pred = speeds(max_speed, exponent)
        result = 0
        for m, p in zip(measured, pred):
            if measured is not None:
                result += (m - p) ** 2
        return result
    result = scipy.optimize.minimize(loss, (200., 1.3))
    print(result)
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib import pylab
    pylab.plot(measured)
    pylab.plot(speeds(*result.x))
    pylab.show()
    return result.x


if __name__ == '__main__':
    top_speed, exponent = fit_speeds((0, 2, 5, 10, 15, 22, 30, 41, 51, 64, 77, 91, 106, 120, 136))
    print(f"Top speed: {top_speed}, exponent: {exponent}")
