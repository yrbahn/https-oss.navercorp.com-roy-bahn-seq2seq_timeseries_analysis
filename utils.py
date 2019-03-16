import argparse
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame


def normalize(values):
    return (values - values.mean()) / np.std(values)


def denormalize(values):
    return values * (values.max() - values.min()) + values.min()


def date_range(date1, date2):
    for n in range(int((date2 - date1).days) + 1):
        yield date1 + timedelta(n)


def validate_date_type(s):
    try:
        return datetime.strptime(s, '%Y%m%d')
    except ValueError:
        msg = 'Not a valid date: "{0}".'.format(s)
        raise argparse.ArgumentTypeError(msg)


def validate_int_type(s):
    try:
        return int(s)
    except ValueError:
        msg = 'Not a valid integer: "{0}".'.format(s)
        raise argparse.ArgumentTypeError(msg)


def visualize(title, values):
    dates = values.index.strftime('%m-%d %H:%M')[-1008:]

    plt.plot(dates, values[-1008:], 'r', label='pv')
    plt.xticks(dates[::48 * 2])
    plt.title(title)
    plt.legend()
    plt.show()
    # plt.savefig(os.path.join(dir, '{}.png'.format(title)))

# def visualize(title, values: List[(DataFrame, str, str)]):
#     dates = values[0][0].index.strftime('%m-%d %H:%M')
#
#     for v in values:
#         plt.plot(dates, v[0], v[1], label=v[2])
#         plt.xticks(dates[::48 * 2])
#         plt.title(title)
#         plt.legend()
#     plt.show()
#     # plt.savefig(os.path.join(dir, '{}.png'.format(title)))

