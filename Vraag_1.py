import torch
from torch.nn.utils.rnn import pad_sequence

def pad_and_convert_to_tensor(batch_sequences, batch_labels):
    # Pad sequences to the maximum length
    padded_sequences = pad_sequence([torch.tensor(seq, dtype=torch.long) for seq in batch_sequences], batch_first=True,
                                    padding_value=0)
    labels_tensor = torch.tensor(batch_labels, dtype=torch.float32)
    return padded_sequences, labels_tensor



def get_batches(x_set, y_set, batch_size = 32):
    all_padded_batches_x = []
    all_tensor_batches_y = []

    # Select batches of sentences in the trainset
    # Define the length of the longest sentence in that batch
    # Add zero's to all sentences to reach the same length of the longest sentence
    # In the end all_padded_batches_x contains 625 batches of size 32. In total 20000 sentences.
    for i in range(0, len(x_set), batch_size):
        batch_x = x_set[i:i + batch_size]
        batch_y = y_set[i:i + batch_size]

        padded_batch_x, tensor_batch_y = pad_and_convert_to_tensor(batch_x, batch_y)
        all_padded_batches_x.append(padded_batch_x)
        all_tensor_batches_y.append(tensor_batch_y)

    return all_padded_batches_x, all_tensor_batches_y