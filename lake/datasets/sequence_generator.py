"""SequenceGenerator class."""
import numpy as np


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
    self.core_sequence = self._create_core_sequence()
    self.sequence = self._create_sequence()


  def _create_core_sequence(self):
    if self.characters % 2 != 0:
      self.characters += 1
      print("Number of characters was increased to next even number to create pairs.")

    seq = [(a, a + 1) for a in range(0, self.characters, 2)]
    return seq

  def _create_sequence(self):
    first = range(0, self.characters, 2)
    second = range(1, self.characters, 2)
    if self.type == "statistical":
        core_idx = np.random.randint(0, self.characters/2)
        seq = [self.core_sequence[core_idx]]
        while len(seq) < self.length:
           next_seq = np.delete(first, core_idx)
           next_pair = [(second[core_idx], next_seq[a]) for a in range(0, len(next_seq))]
           idx = np.random.randint(0, len(next_seq))
           seq.extend([next_pair[idx]])
           core_idx = np.where(list(next_pair[idx])[1] == first)[0][0]
           seq.extend([self.core_sequence[core_idx]])
    elif self.type == "episodic":
        core_idx = np.random.randint(0, self.characters / 2, self.length)
        seq = [(first[a], second[a]) for a in core_idx]
    else:
        raise NotImplementedError('Learning type must be statistical or episodic.')

    return seq




