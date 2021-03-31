"""
Created on Jan 11, 2018

@author: Siyuan Qi

Description of the file.

"""

import pickle

import torch.utils.data


class CAD120(torch.utils.data.Dataset):
    def __init__(self, filename):
        self.data = pickle.load(open(filename, 'rb'))
        self.sequence_ids = list()
        for sequence_id, val in self.data.items():
            self.sequence_ids.append(sequence_id)

    def __getitem__(self, index):
        sequence_id = self.sequence_ids[index]
        features = self.data[sequence_id]['features']
        labels = self.data[sequence_id]['labels']
        seg_lengths = self.data[sequence_id]['seg_lengths']
        total_length = self.data[sequence_id]['total_length']
        activity = self.data[sequence_id]['activity']
        return features, labels, seg_lengths, total_length, activity, sequence_id

    def __len__(self):
        return len(self.sequence_ids)


def main():
    pass


if __name__ == '__main__':
    main()
