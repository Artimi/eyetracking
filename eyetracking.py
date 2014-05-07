#!/usr/bin/env python2.7
# -*- encoding: utf-8 -*-

from __future__ import division
import csv
import numpy as np
import scipy.cluster.vq as spvq
import itertools

FILE_NAME = 's1-s6_sorted.csv'
DELTA_T = 1.0
LEN_FIXATIONS_THRESHOLD = 10


class Record(object):

    def __init__(self, line):
        self.subject = line[0]
        self.known = line[1]
        self.position = np.array(map(float, line[2:]))
        self.position = np.reshape(self.position, (2, len(self.position) / 2), order='F')
        self.velocity = self.compute_velocity(self.position)
        self.peaks = self.get_peaks(self.velocity)
        self.len_fixations = self.get_len_fixations(self.peaks)

    def compute_velocity(self, position):
        velocity = np.zeros([position.shape[1]], dtype=float)
        for i in range(1, position.shape[1]):
            velocity[i] = np.sqrt((position[0, i] - position[0, i - 1]) ** 2
                                  + (position[1, i] - position[1, i - 1]) ** 2) / DELTA_T
        return velocity

    def get_peaks(self, velocity):
        clusters, labels = spvq.kmeans2(velocity, 2)
        labels = labels.astype(bool)
        if clusters[0] > clusters[1]:
            labels = np.logical_not(labels)
        return labels

    def get_len_fixations(self, peaks):
        len_fixations = np.array([])
        for bit, group in itertools.groupby(peaks):
            if not bit:
                len_fixations = np.append(len_fixations, sum(1 for _ in group) / DELTA_T)
        return len_fixations

    @property
    def MSA(self):
        return np.average(self.velocity[self.peaks])

    @property
    def MFD(self):
        return np.average(self.len_fixations[self.len_fixations > LEN_FIXATIONS_THRESHOLD])


def get_record(file_path=FILE_NAME):
    with open(file_path) as csvfile:
        for line in csv.reader(csvfile):
            yield Record(line)


def get_result_dict():
    result = {}
    for rec in get_record():
        try:
            result[rec.subject]
        except KeyError:
            result[rec.subject] = {"true":
                                   {"MSA": np.array([]),
                                    "MFD": np.array([])},
                                   "false":
                                   {"MSA": np.array([]),
                                    "MFD": np.array([])}}
        finally:
            result[rec.subject][rec.known]["MSA"] = np.append(result[rec.subject][rec.known]["MSA"], rec.MSA)
            result[rec.subject][rec.known]["MFD"] = np.append(result[rec.subject][rec.known]["MFD"], rec.MFD)
    return result


if __name__ == "__main__":
    get_record()
