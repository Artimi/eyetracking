#!/usr/bin/env python2.7
# -*- encoding: utf-8 -*-

from __future__ import division
import csv
import numpy as np
import scipy.cluster.vq as spvq
import itertools

FILE_NAME = 's1-s6_sorted.csv'


class Record(object):

    def __init__(self, line):
        self.subject = line[0]
        self.known = line[1]
        self.position = np.array(map(float, line[2:]))
        self.position = np.reshape(self.position, (2, len(self.position) / 2), order='F')
        self.velocity = self.compute_velocity(self.position)
        self.peaks = self.get_peaks(self.velocity)

    def compute_velocity(self, position):
        velocity = np.zeros([position.shape[1]], dtype=float)
        for i in range(1, position.shape[1]):
            velocity[i] = np.sqrt((position[0, i] - position[0, i - 1]) ** 2 + (position[1, i] - position[1, i - 1]) ** 2)
        return velocity

    def get_peaks(self, velocity):
        clusters, labels = spvq.kmeans2(velocity, 2)
        labels = labels.astype(bool)
        if clusters[0] > clusters[1]:
            labels = np.logical_not(labels)
        return labels

    def MSA(self):
        return np.average(self.velocity[self.peaks])

    def MFD(self):
        accumulator = 0
        fixations = 0
        for bit, group in itertools.groupby(self.peaks):
            if not bit:
                fixations += 1
                accumulator += sum(1 for _ in group)
        return accumulator / fixations


def get_record(file_path=FILE_NAME):
    with open(file_path) as csvfile:
        for line in csv.reader(csvfile):
            yield Record(line)

if __name__ == "__main__":
    get_record()
