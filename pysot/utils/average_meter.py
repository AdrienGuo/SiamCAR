from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from collections import defaultdict


class Meter(object):
    def __init__(self, name, val, avg):
        self.name = name
        self.val = val
        self.avg = avg

    def __repr__(self):
        return "{name}: {val:.6f} ({avg:.6f})".format(
            name=self.name, val=self.val, avg=self.avg
        )

    def __format__(self, *tuples, **kwargs):
        return self.__repr__()


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self, name) -> None:
        self.name = name
        self.reset()

    def reset(self):
        self.val = defaultdict(float)
        self.avg = defaultdict(float)
        self.sum = defaultdict(float)
        self.count = 0

    def update(self, val: dict, num=1):
        self.val = val
        for key, value in val.items():
            # TODO: num 應該都只會是 1
            self.sum[key] += value * num
        self.count += num
        for key, value in self.sum.items():
            self.avg[key] = value / self.count

    def display(self, type: str):
        attr = None
        if type == "val":
            attr = self.val
        elif type == "avg":
            attr = self.avg
        elif type == "sum":
            attr = self.sum
        else:
            assert False, "ERROR, invalid display type"

        fmtstr = ""
        for key, value in attr.items():
            fmtstr += f"{key}: {value:<6.3f} | "
        print(str(fmtstr))
