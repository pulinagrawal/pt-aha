"""SequenceGenerator class."""
import numpy as np
from igraph import Graph
from random import randint, choice, shuffle
from itertools import islice


class SequenceGenerator:
    """Create a sequences of length equal to seq_length. Pairs are formed with digits from 0 to the number specified
   by characters. From Schapiro: There were eight items (A–H) grouped into four pairs (AB, CD, EF, GH). Items within a
    pair always occurred in a fixed order but the sequence of pairs was random. Specifically, the second item in a pair
    could transition to the first item in one of the three other pairs. Back-to-back repetitions of a pair were excluded,
    since this is a common constraint in statistical learning experiments and because allowing repetitions would dilute
    the temporal asymmetry (both AB and BA would be exposed). There was a moving window of two stimuli presented at
    a time. After AB, for example, BC, BE, or BG followed with equal probability; if BC was chosen, the next input would
be CD. """

    def __init__(self, characters, seq_length, type):
        self.characters = characters
        self.length = seq_length
        self.type = type
        self.all_pairs = [(a, b) for a in range(0, self.characters) for b in range(0, self.characters)]
        self.core_sequence = self._create_core_sequence()
        self.core_label_sequence = self._create_label_sequence()
        self.sequence = self._create_sequence()
        self.test_sequence = self._create_test_sequence()

    def _create_core_sequence(self):
        if self.characters % 2 != 0:
            self.characters += 1
            print("Number of characters was increased to next even number to create pairs.")

        seq = [(a, a + 1) for a in range(0, self.characters, 2)]
        return seq

    def _create_label_sequence(self):
        if self.type == "statistical":
            if self.characters % 2 != 0:
                self.characters += 1
                print("Number of characters was increased to next even number to create pairs.")

            seq = [(a, a + 1) for a in range(0, self.characters, 2)]
            seq_sec = [[(b, i) for i in range(0, self.characters, 2) if i != (b - 1)] for (a, b) in seq]
            seq_sec = sum(seq_sec, [])
            return seq + seq_sec
        elif self.type == "episodic":
            return self.core_sequence
        else:
            raise NotImplementedError('Learning type must be statistical or episodic in pairs structure experiments.')

    def _create_sequence(self):
        first = range(0, self.characters, 2)
        second = range(1, self.characters, 2)
        if self.type == "statistical":
            core_idx = np.random.randint(0, self.characters / 2)
            seq = [self.core_sequence[core_idx]]
            while len(seq) < self.length:
                next_seq = np.delete(first, core_idx)
                next_pair = [(second[core_idx], next_seq[a]) for a in range(0, len(next_seq))]
                idx = np.random.randint(0, len(next_seq))
                seq.extend([next_pair[idx]])
                core_idx = np.where(list(next_pair[idx])[1] == first)[0][0]
                seq.extend([self.core_sequence[core_idx]])
            del seq[len(seq) - 1]
        elif self.type == "episodic":
            core_idx = np.random.randint(0, self.characters / 2, self.length)
            seq = [(first[a], second[a]) for a in core_idx]
        else:
            raise NotImplementedError('Learning type must be statistical or episodic in pairs structure experiments.')

        return seq

    def _create_test_sequence(self):
        test_seq = [a for a in self.all_pairs if a not in self.core_sequence]
        identical = [(a, a) for a in range(0, self.characters)]
        test_seq = [a for a in test_seq if a not in identical]
        return test_seq


class SequenceGeneratorGraph:

    def __init__(self, characters, seq_length, experiment_type, communities):
        self.characters = characters
        self.community_size = int(characters / communities)
        self.communities = communities
        self.type = experiment_type
        self.length = seq_length
        self.all_pairs = [(a, b) for a in range(0, self.community_size) for b in range(0, self.community_size)]
        self.core_label_sequence, self.graph_sequences = self._create_label_sequence()
        self.sequence = self._create_sequence()


    def _create_label_sequence(self):
        edges = []
        within_internal = []
        within_boundary = []
        across_boundary = []
        for i in range(0, self.characters, self.community_size):
            for a in range(i + 1, i + self.community_size):
                if a == i + self.community_size - 1:
                    if a == self.communities * self.community_size - 1:
                        edges = edges + [(a, 0), (0, a)]
                        across_boundary = across_boundary + [(a, 0), (0, a)]
                    else:
                        edges = edges + [(a, a + 1), (a + 1, a)]
                        across_boundary = across_boundary + [(a, a + 1), (a + 1, a)]
                else:
                    edges = edges + [(i, a), (a, i)]
                    within_internal = within_internal + [(i, a), (a, i)]
                for b in range(a + 1, i + self.community_size):
                    edges = edges + [(a, b), (b, a)]
                    within_internal = within_internal + [(a, b), (b, a)]
            within_boundary = within_boundary + [(i, i + self.community_size - 1), (i + self.community_size - 1, i)]
        across_other = [a for a in self.all_pairs if a not in within_internal + within_boundary + across_boundary]

        return edges, [within_internal, within_boundary, across_boundary, across_other]

    def _create_sequence(self):
        if self.type == "static":
            seq = self.core_label_sequence.copy()
            seq = seq*int(self.length / len(seq))
            shuffle(seq)
            return seq
        elif self.type == "random":
            walk = list(islice(self._random_walk(), self.length))
            seq = [(walk[i], walk[i + 1]) for i in range(len(walk) - 1)]
            return seq
        else:
            raise NotImplementedError('Learning type must be random or static in community experiments')

    def _random_walk(self, start=0):
        g = Graph(self.core_label_sequence)
        current = randint(0, g.vcount() - 1) if start is None else start
        while True:
            yield current
            current = choice(g.successors(current))


class SequenceGeneratorTriads:

    def __init__(self, characters, seq_length, type, batch_size):
        self.characters = characters
        self.length = seq_length
        self.type = type
        self.sub_length = batch_size
        self.all_pairs = [(a, b) for a in range(0, self.characters) for b in range(0, self.characters)]
        self.core_label_sequence = self._create_label_sequence()
        self.sequence = self._create_sequence()
        self.core_sequence = self.core_label_sequence
        self.test_sequence = self._create_test_sequence()
        self.base_sequence = self._create_base_sequence()

    def _create_label_sequence(self):
        if self.characters % 3 != 0:
            self.characters += (3 - (10 % 3))
            print("Number of characters was increased to next even number to create pairs.")
        seq = []
        for a in range(0, self.characters, 3):
            seq = seq + [(a, a + 1), (a + 1, a + 2)]
        return seq


    def _create_sequence(self):
        if self.type == 'static':
            tmp = self.core_label_sequence.copy()
            shuffle(tmp)
            tmp = tmp*(self.sub_length//len(tmp))
            seq = tmp.copy()
            for _ in range(0, int(self.length / len(seq)) - 1):
                shuffle(tmp)
                seq.extend(tmp)
            return seq
        else:
            raise NotImplementedError('Learning type must be static in associative experiments')

    def _create_test_sequence(self):
        seq = []
        for a in range(0, self.characters, 3):
            seq = seq + [(a, a + 2)]
        return seq

    def _create_base_sequence(self):
        seq = [a for a in self.all_pairs if a not in self.core_sequence]
        seq = [a for a in seq if a not in self.test_sequence]
        return seq
