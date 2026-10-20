import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
from mose.mose import MOSE
from mose.metrics import db_eval_boundary, db_eval_iou
from mose import utils
from mose.results import Results
from scipy.optimize import linear_sum_assignment


class MOSEEvaluation(object):
    def __init__(self, mose_root, task, gt_set, sequences='all', codalab=False):
        """
        Class to evaluate MOSE sequences from a certain set and for a certain task
        :param mose_root: Path to the MOSE folder that contains train/, valid/, meta files.
        :param task: Task to compute the evaluation (only semi-supervised for MOSE).
        :param gt_set: Set to compute the evaluation ('train' or 'valid')
        :param sequences: Sequences to consider for the evaluation, 'all' to use all the sequences in a set.
        """
        self.mose_root = mose_root
        self.task = task
        self.dataset = MOSE(root=mose_root, task=task, subset=gt_set, sequences=sequences, codalab=codalab)

    @staticmethod
    def _evaluate_semisupervised(all_gt_masks, all_res_masks, all_void_masks, metric):
        if all_res_masks.shape[0] > all_gt_masks.shape[0]:
            sys.stdout.write("\nIn your PNG files there is an index higher than the number of objects in the sequence!")
            sys.exit()
        elif all_res_masks.shape[0] < all_gt_masks.shape[0]:
            zero_padding = np.zeros((all_gt_masks.shape[0] - all_res_masks.shape[0], *all_res_masks.shape[1:]))
            all_res_masks = np.concatenate([all_res_masks, zero_padding], axis=0)
        j_metrics_res, f_metrics_res = np.zeros(all_gt_masks.shape[:2]), np.zeros(all_gt_masks.shape[:2])
        for ii in range(all_gt_masks.shape[0]):
            if 'J' in metric:
                j_metrics_res[ii, :] = db_eval_iou(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
            if 'F' in metric:
                f_metrics_res[ii, :] = db_eval_boundary(all_gt_masks[ii, ...], all_res_masks[ii, ...], all_void_masks)
        return j_metrics_res, f_metrics_res

    def evaluate(self, res_path, metric=('J', 'F'), debug=False):
        metric = metric if isinstance(metric, tuple) or isinstance(metric, list) else [metric]
        if 'T' in metric:
            raise ValueError('Temporal metric not supported!')
        if 'J' not in metric and 'F' not in metric:
            raise ValueError('Metric possible values are J for IoU or F for Boundary')

        # Containers
        metrics_res = {}
        if 'J' in metric:
            metrics_res['J'] = {"M": [], "R": [], "D": [], "M_per_object": {}}
        if 'F' in metric:
            metrics_res['F'] = {"M": [], "R": [], "D": [], "M_per_object": {}}

        # Sweep all sequences
        results = Results(root_dir=res_path)
        for seq in tqdm(list(self.dataset.get_sequences())):
            all_gt_masks, all_void_masks, all_masks_id = self.dataset.get_all_masks(seq, True)
            if self.task == 'semi-supervised':
                all_gt_masks, all_masks_id = all_gt_masks[:, 1:-1, :, :], all_masks_id[1:-1]
            all_res_masks = results.read_masks(seq, all_masks_id)
            j_metrics_res, f_metrics_res = self._evaluate_semisupervised(all_gt_masks, all_res_masks, None, metric)
            for ii in range(all_gt_masks.shape[0]):
                seq_name = f'{seq}_{ii+1}'
                if 'J' in metric:
                    [JM, JR, JD] = utils.db_statistics(j_metrics_res[ii])
                    metrics_res['J']["M"].append(JM)
                    metrics_res['J']["R"].append(JR)
                    metrics_res['J']["D"].append(JD)
                    metrics_res['J']["M_per_object"][seq_name] = JM
                if 'F' in metric:
                    [FM, FR, FD] = utils.db_statistics(f_metrics_res[ii])
                    metrics_res['F']["M"].append(FM)
                    metrics_res['F']["R"].append(FR)
                    metrics_res['F']["D"].append(FD)
                    metrics_res['F']["M_per_object"][seq_name] = FM

            # Show progress
            if debug:
                sys.stdout.write(seq + '\n')
                sys.stdout.flush()
        return metrics_res

