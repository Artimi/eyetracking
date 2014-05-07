#!/usr/bin/env python2.7
# -*- encoding: utf-8 -*-

import csv
import numpy as np
import scipy.cluster.vq as spvq

FILE_NAME = 's1-s6_sorted.csv'
THRESHOLD = 8.0


class Record(object):

    def __init__(self, line):
        self.subject = line[0]
        self.known = line[1]
        self.position = np.array(map(float, line[2:]))
        self.position = np.reshape(self.position, (2, len(self.position) / 2), order='F')
        self.velocity = self.compute_velocity(self.position)

    def compute_velocity(self, position):
        velocity = np.zeros([position.shape[1]], dtype=float)
        for i in range(1, position.shape[1]):
            velocity[i] = np.sqrt((position[0, i] - position[0, i - 1]) ** 2 + (position[1, i] - position[1, i - 1]) ** 2)
        return velocity

    def get_peaks(self, velocity):
        clusters, labels = spvq.kmeans2(velocity, 2)
        labels = np.astype(bool)
        if clusters[0] > clusters[1]:
            labels = np.logical_not(labels)
        return labels


def get_record(file_path=FILE_NAME):
    with open(file_path) as csvfile:
        for line in csv.reader(csvfile):
            yield Record(line)

if __name__ == "__main__":
    get_record()
