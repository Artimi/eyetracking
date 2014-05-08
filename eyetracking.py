#!/usr/bin/env python2.7
# -*- encoding: utf-8 -*-

from __future__ import division
import csv
import numpy as np
import sklearn.cluster
import itertools
from collections import namedtuple

FILE_NAME = 's1-s6_sorted.csv'
DELTA_T = 1.0
LEN_FIXATIONS_THRESHOLD = 10

SubjectResult = namedtuple("SubjectResult", ['subject', "MFD_true", "MFD_SD_true", "MFD_false", "MFD_SD_false",
                           "MSA_true", "MSA_SD_true", "MSA_false", "MSA_SD_false"])
OverallResult = namedtuple('OverallResult', ['MFD_overall_true', 'MFD_overall_true_SD',
                                             'MFD_overall_false', 'MFD_overall_false_SD',
                                             'MSA_overall_true', 'MSA_overall_true_SD',
                                             'MSA_overall_false', 'MSA_overall_false_SD'])


class Record(object):

    def __init__(self, line):
        self.subject = line[0]
        self.known = line[1]
        self.position = np.array(map(float, line[2:]))
        self.position = np.reshape(self.position, (2, len(self.position) // 2), order='F')
        self.velocity = self.compute_velocity(self.position)
        self.peaks = self.get_peaks(self.velocity)
        self.len_fixations = self.get_len_fixations(self.peaks)
        self.MFD = self.get_MFD()
        self.MSA = self.get_MSA()

    def compute_velocity(self, position):
        velocity = np.zeros([position.shape[1]], dtype=float)
        for i in range(1, position.shape[1]):
            velocity[i] = np.sqrt((position[0, i] - position[0, i - 1]) ** 2
                                  + (position[1, i] - position[1, i - 1]) ** 2) / DELTA_T
        return velocity

    def get_peaks(self, velocity):
        k_means = sklearn.cluster.KMeans(n_clusters=2)
        k_means.fit(np.expand_dims(velocity, 1))
        labels = k_means.labels_
        clusters = k_means.cluster_centers_
        labels = labels.astype(bool)
        if clusters[0, 0] > clusters[1, 0]:
            labels = np.logical_not(labels)
        return labels

    def get_len_fixations(self, peaks):
        len_fixations = np.array([])
        for bit, group in itertools.groupby(peaks):
            if not bit:
                len_fixations = np.append(len_fixations, sum(1 for _ in group) / DELTA_T)
        return len_fixations

    def get_MSA(self):
        return np.average(self.velocity[self.peaks])

    def get_MFD(self):
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


def process_result(res):
    subject_results = []
    MFD_overall_true = np.array([])
    MFD_overall_false = np.array([])
    MSA_overall_true = np.array([])
    MSA_overall_false = np.array([])
    for subject in res:
        subject_results.append(SubjectResult(
            subject,
            np.average(res[subject]["true"]["MFD"]),
            np.std(res[subject]["true"]["MFD"]),
            np.average(res[subject]["false"]["MFD"]),
            np.std(res[subject]["false"]["MFD"]),
            np.average(res[subject]["true"]["MSA"]),
            np.std(res[subject]["true"]["MSA"]),
            np.average(res[subject]["false"]["MSA"]),
            np.std(res[subject]["false"]["MSA"])))
        MFD_overall_true = np.append(MFD_overall_true, res[subject]['true']['MFD'])
        MFD_overall_false = np.append(MFD_overall_false, res[subject]['false']['MFD'])
        MSA_overall_true = np.append(MSA_overall_true, res[subject]['true']['MSA'])
        MSA_overall_false = np.append(MSA_overall_false, res[subject]['false']['MSA'])
    overall_result = OverallResult(
        np.average(MFD_overall_true), np.std(MFD_overall_true),
        np.average(MFD_overall_false), np.std(MFD_overall_false),
        np.average(MSA_overall_true), np.std(MSA_overall_true),
        np.average(MSA_overall_false), np.std(MSA_overall_false))
    return subject_results, overall_result


def generate_result_csv(subject_results, overall_result):
    result = ""
    for subject_data in subject_results:
        line = " ".join(map(str, subject_data))
        line += " " + " ".join(map(str, overall_result))
        result += line + '\n'
    return result


if __name__ == "__main__":
    get_result_dict()
