import os
import json
from glob import glob
from collections import defaultdict
import numpy as np
from PIL import Image


class YouTubeVOS(object):
    """
    YouTube-VOS Dataset loader with custom train/val splits.
    
    Split structure:
    - train: Videos from original YouTube-VOS train set (full annotations)
    - val: 150 videos sampled from original YouTube-VOS train set (full annotations, for local eval)
    
    Data locations:
    - All splits use data in {root}/train/JPEGImages and {root}/train/Annotations
    """
    SUBSET_OPTIONS = ['train', 'val']
    TASKS = ['semi-supervised']
    DATASET_WEB = 'https://youtube-vos.org/'
    VOID_LABEL = 255

    def __init__(self, root, task='semi-supervised', subset='val', sequences='all', codalab=False):
        """
        Class to read the YouTube-VOS dataset
        :param root: Path to the YouTube-VOS folder that contains train/, ImageSets/.
        :param task: Task to load the annotations (only semi-supervised supported).
        :param subset: Set to load ('train' or 'val')
        :param sequences: Sequences to consider, 'all' to use all the sequences in a set.
        """
        if subset not in self.SUBSET_OPTIONS:
            raise ValueError(f'Subset should be in {self.SUBSET_OPTIONS}')
        if task not in self.TASKS:
            raise ValueError(f'The only tasks that are supported are {self.TASKS}')

        self.task = task
        self.subset = subset
        self.root = root
        
        # YouTube-VOS structure: all data in train folder
        self.img_path = os.path.join(self.root, 'train', 'JPEGImages')
        self.mask_path = os.path.join(self.root, 'train', 'Annotations')
        self.imagesets_path = os.path.join(self.root, 'ImageSets')

        self._check_directories()

        # Load video list from ImageSets
        if sequences == 'all':
            split_file = os.path.join(self.imagesets_path, f'{subset}.txt')
            with open(split_file, 'r') as f:
                sequences_names = [line.strip() for line in f.readlines() if line.strip()]
        else:
            sequences_names = sequences if isinstance(sequences, list) else [sequences]

        self.sequences = defaultdict(dict)

        for seq in sequences_names:
            images = np.sort(glob(os.path.join(self.img_path, seq, '*.jpg'))).tolist()
            if len(images) == 0 and not codalab:
                raise FileNotFoundError(f'Images for sequence {seq} not found.')
            self.sequences[seq]['images'] = images
            masks = np.sort(glob(os.path.join(self.mask_path, seq, '*.png'))).tolist()
            masks.extend([-1] * (len(images) - len(masks)))
            self.sequences[seq]['masks'] = masks

    def _check_directories(self):
        if not os.path.exists(self.root):
            raise FileNotFoundError(f'YouTube-VOS not found in the specified directory, download it from {self.DATASET_WEB}')
        if not os.path.exists(self.imagesets_path):
            raise FileNotFoundError(f'ImageSets folder not found at {self.imagesets_path}. '
                                    f'Run create_ytvos_splits.py first.')
        split_file = os.path.join(self.imagesets_path, f'{self.subset}.txt')
        if not os.path.exists(split_file):
            raise FileNotFoundError(f'Split file for {self.subset} not found at {split_file}. '
                                    f'Run create_ytvos_splits.py first.')
        if not os.path.exists(self.img_path):
            raise FileNotFoundError(f'JPEGImages folder not found at {self.img_path}')
        if not os.path.exists(self.mask_path):
            raise FileNotFoundError(f'Annotations folder not found at {self.mask_path}')

    def get_frames(self, sequence):
        for img, msk in zip(self.sequences[sequence]['images'], self.sequences[sequence]['masks']):
            image = np.array(Image.open(img))
            mask = None if msk == -1 else np.array(Image.open(msk))
            yield image, mask

    def _get_all_elements(self, sequence, obj_type):
        obj = np.array(Image.open(self.sequences[sequence][obj_type][0]))
        all_objs = np.zeros((len(self.sequences[sequence][obj_type]), *obj.shape))
        obj_id = []
        for i, obj in enumerate(self.sequences[sequence][obj_type]):
            all_objs[i, ...] = np.array(Image.open(obj))
            obj_id.append(''.join(obj.split('/')[-1].split('.')[:-1]))
        return all_objs, obj_id

    def get_all_images(self, sequence):
        return self._get_all_elements(sequence, 'images')

    def get_all_masks(self, sequence, separate_objects_masks=False):
        masks, masks_id = self._get_all_elements(sequence, 'masks')
        masks_void = np.zeros_like(masks)

        # Separate void and object masks
        for i in range(masks.shape[0]):
            masks_void[i, ...] = masks[i, ...] == 255
            masks[i, masks[i, ...] == 255] = 0

        if separate_objects_masks:
            num_objects = int(np.max(masks[0, ...]))
            tmp = np.ones((num_objects, *masks.shape))
            tmp = tmp * np.arange(1, num_objects + 1)[:, None, None, None]
            masks = (tmp == masks[None, ...])
            masks = masks > 0
        return masks, masks_void, masks_id

    def get_sequences(self):
        for seq in self.sequences:
            yield seq


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    subsets = ['val']

    for s in subsets:
        dataset = YouTubeVOS(root='/dev/shm/data/Youtube-VOS', subset=s)
        print(f"Loaded {s} split with {len(list(dataset.sequences.keys()))} sequences")
        for seq in dataset.get_sequences():
            g = dataset.get_frames(seq)
            img, mask = next(g)
            plt.subplot(2, 1, 1)
            plt.title(seq)
            plt.imshow(img)
            plt.subplot(2, 1, 2)
            plt.imshow(mask)
            plt.show(block=True)
            break

