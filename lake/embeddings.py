import torch.nn
from scipy import stats
import csv

class Correlations:
    def __init__(self, path, seed):
        self.path = path
        self.seed = seed

    def correlation(self, data_A, data_B, file_name):
        data_A_flat = torch.flatten(data_A, start_dim=1)
        data_B_flat = torch.flatten(data_B, start_dim=1)
        correlation = [[stats.pearsonr(a, b)[0] for a in data_B_flat] for b in
                       data_A_flat]
        correlation = torch.tensor(correlation).numpy()

        with open(self.path + '/correlation_' + file_name + '.csv', 'w', encoding='UTF8') as f:
            writer_file = csv.writer(f)
            writer_file.writerows(correlation)

    def transitivity(self, ca1_outputs, characters):

        filtered_correlation = [['correlation_A', 'correlation_filter_A', 'similarity_A', 'similarity_filter_A',
                                        'correlation_B', 'correlation_filter_B', 'similarity_B', 'similarity_filter_B',
                                        'correlation_C', 'correlation_filter_C', 'similarity_C', 'similarity_filter_C',
                                        'neg', 'noABC', 'all', 'AorB', 'only_C']]
        for i in range(0, characters.size(0), 3):

            A_transitivity = torch.flatten(ca1_outputs[i])
            character_flat_A = torch.flatten(characters[i])
            character_flat_B = torch.flatten(characters[i + 1])
            character_flat_C = torch.flatten(characters[i + 2])
            filtered_output = torch.empty(A_transitivity.size(0), dtype=torch.float32)
            only_C = 0
            all = 0
            AorB = 0
            noABC = 0
            neg = 0
            for j in range(A_transitivity.size(0)):
                if A_transitivity[j] < 0:
                    filtered_output[j] = 0
                    neg = neg + 1
                else:
                    if A_transitivity[j] != 0:
                        if character_flat_C[j] != 0:
                            filtered_output[j] = A_transitivity[j]
                            if character_flat_A[j] == 0 and character_flat_B[j] == 0:
                                only_C = only_C + 1
                            else:
                                all = all + 1
                        else:
                            if character_flat_A[j] == 0 and character_flat_B[j] == 0:
                                filtered_output[j] = A_transitivity[j]
                                noABC = noABC + 1
                            else:
                                filtered_output[j] = 0
                                AorB = AorB + 1
                    else:
                        filtered_output[j] = 0

            correlation_A = stats.pearsonr(A_transitivity, character_flat_A)[0]
            correlation_filter_A = stats.pearsonr(filtered_output, character_flat_A)[0]
            correlation_B = stats.pearsonr(A_transitivity, character_flat_B)[0]
            correlation_filter_B = stats.pearsonr(filtered_output, character_flat_B)[0]
            correlation_C = stats.pearsonr(A_transitivity, character_flat_C)[0]
            correlation_filter_C = stats.pearsonr(filtered_output, character_flat_C)[0]

            cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
            similarity_A = cos(A_transitivity, character_flat_A).item()
            similarity_filter_A = cos(filtered_output, character_flat_A).item()
            similarity_B = cos(A_transitivity, character_flat_B).item()
            similarity_filter_B = cos(filtered_output, character_flat_B).item()
            similarity_C = cos(A_transitivity, character_flat_C).item()
            similarity_filter_C = cos(filtered_output, character_flat_C).item()

            filtered_correlation.append([correlation_A, correlation_filter_A, similarity_A, similarity_filter_A,
                                        correlation_B, correlation_filter_B, similarity_B, similarity_filter_B,
                                        correlation_C, correlation_filter_C, similarity_C, similarity_filter_C,
                                        neg, noABC, all, AorB, only_C])

        with open(self.path + '/filtered_correlation_'+str(self.seed)+'.csv', 'w', encoding='UTF8') as f:
            writer_file = csv.writer(f)
            writer_file.writerows(filtered_correlation)


class Overlap:
    def __init__(self, coefficient, option):
        self.coefficient = coefficient
        self.option = option

    def join(self, A, B):
        if self.option == 'add':
            overlapped_pair = self.coefficient * A + B
        else:
            A_tmp = torch.flatten(A, start_dim=0)
            B_tmp = torch.flatten(B, start_dim=0)

            overlapped_pair = torch.empty(A_tmp.size(0), dtype=torch.float32)
            for i in range(A_tmp.size(0)):
                if A_tmp[i] == B_tmp[i]:
                    overlapped_pair[i] = A_tmp[i]
                if A_tmp[i] != B_tmp[i] and A_tmp[i] != 0 and B_tmp[i] != 0:
                    if self.option == 'mean':
                        overlapped_pair[i] = torch.mean(torch.stack([A_tmp[i], B_tmp[i]]))
                    if self.option == 'minimum':
                        overlapped_pair[i] = torch.min(torch.stack([A_tmp[i], B_tmp[i]]))
                    if self.option == 'maximum':
                        overlapped_pair[i] = torch.max(torch.stack([A_tmp[i], B_tmp[i]]))
                if A_tmp[i] != B_tmp[i] and A_tmp[i] == 0:
                    overlapped_pair[i] = B_tmp[i]
                if A_tmp[i] != B_tmp[i] and B_tmp[i] == 0:
                    overlapped_pair[i] = self.coefficient * A_tmp[i]
        overlapped_pair = torch.reshape(overlapped_pair, A.size())
        return overlapped_pair

