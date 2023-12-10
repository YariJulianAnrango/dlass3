import torch
from torch.nn.utils.rnn import pad_sequence
import random
def pad(batch_sequences):
    # Pad sequences to the maximum length
    padded_sequences = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in batch_sequences], batch_first=True,
                                    padding_value=0)
    return padded_sequences

def seedshuffle():
    return 0.1


def get_batches(x_set, w2i, batch_size = 30):
    all_padded_batches_x = []
    all_tensor_batches_y = []
    for i in range(0, len(x_set), batch_size):
        batch_x = x_set[i:i + batch_size]
        batch_x = [[w2i[".start"]] + sublist + [w2i[".end"]] for sublist in batch_x]
        batch_y = []
        for j in batch_x:
            labels_tensor = torch.tensor(j, dtype=torch.float32)
            y_tensor = torch.cat([labels_tensor[1:], torch.tensor([0])])
            batch_y.append(y_tensor)
        padded_batch_x = pad(batch_x)
        padded_batch_y = pad(batch_y)
        all_padded_batches_x.append(padded_batch_x)
        all_tensor_batches_y.append(padded_batch_y)

        random.shuffle(all_padded_batches_x, seedshuffle)
        random.shuffle(all_tensor_batches_y, seedshuffle)

    return all_padded_batches_x, all_tensor_batches_y
