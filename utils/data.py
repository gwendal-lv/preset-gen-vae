
import numpy as np

import torch.utils.data


def random_split(full_dataset, datasets_proportions, random_gen_seed=0):
    """ Wrapper for torch.utils.data.random_split which takes 3 float proportions instead
     of integer lengths as input.

     Returns a dict of 3 sub-datasets with keys 'train', 'validation' and 'test'. """
    assert len(datasets_proportions) == 3
    sub_dataset_lengths = [int(np.floor(r * len(full_dataset))) for r in datasets_proportions]
    sub_dataset_lengths[2] = len(full_dataset) - sub_dataset_lengths[0] - sub_dataset_lengths[1]
    # Indexes are randomly split, but the gen seed must always be the same
    sub_datasets = torch.utils.data.random_split(full_dataset, sub_dataset_lengths,
                                                 generator=torch.Generator().manual_seed(random_gen_seed))
    return {'train': sub_datasets[0], 'validation': sub_datasets[1], 'test': sub_datasets[2]}

