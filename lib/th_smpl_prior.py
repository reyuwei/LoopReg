"""
Author: Bharat
Cite: LoopReg: Self-supervised Learning of Implicit Surface Correspondences, Pose and Shape for 3D Human Mesh Registration, NeurIPS' 20.
"""
import numpy as np
import torch
import ipdb


def get_prior(gender='male', precomputed=False):
    if precomputed:
        prior = Prior(sm=None)
        return prior['Generic']
    else:
        from lib.smpl_paths import SmplPaths

        dp = SmplPaths(gender=gender)
        if gender == 'neutral':
            dp_prior = SmplPaths(gender='male')
        else:
            dp_prior = dp

        prior = Prior(dp_prior.get_smpl())
        return prior['Generic']

#TODO proior change to mano pose space

class th_Mahalanobis(object):
    def __init__(self, mean, prec, prefix):
        self.mean = torch.tensor(mean.astype('float32'), requires_grad=False).unsqueeze(axis=0).cuda()
        self.prec = torch.tensor(prec.astype('float32'), requires_grad=False).cuda()
        self.prefix = prefix

    def __call__(self, pose, prior_weight=1.):
        '''
        :param pose: Batch x pose_dims
        :return:
        '''
        # return (pose[:, self.prefix:] - self.mean)*self.prec
        temp = pose[:, self.prefix:] - self.mean[:, :pose.shape[-1]-self.prefix]
        temp2 = torch.matmul(temp, self.prec[:pose.shape[-1]-self.prefix,:pose.shape[-1]-self.prefix]) * prior_weight
        return (temp2 * temp2).sum(dim=1)
        

class Prior(object):
    def __init__(self, sm, prefix=3):
        self.prefix = prefix
        if sm is not None:
            # Compute mean and variance based on the provided poses
            # self.pose_subjects = sm.pose_subjects
            # all_samples = [p[prefix:] for qsub in self.pose_subjects
                        #    for name, p in zip(qsub['pose_fnames'], qsub['pose_parms'])]  # if 'CAESAR' in name or 'Tpose' in name or 'ReachUp' in name] 
            
            from pathlib import Path
            import pickle as pkl
            mpi_list = np.load("assets/hand_data_split_01.pkl", allow_pickle=True)
            mpi_list = mpi_list['train'] + mpi_list['val']
            all_samples = []
            for mpi in mpi_list:
                pose_dict = Path(mpi).parent.parent / "handsOnly_REGISTRATIONS_r_lm___POSES" / (Path(mpi).stem + ".pkl")
                with open(pose_dict, "rb") as f:
                    pose_param = pkl.load(f, encoding='bytes')[b'pose']
                all_samples.append(pose_param.reshape(-1)[prefix:])
            all_samples = np.stack(all_samples)

            self.priors = {'Generic': self.create_prior_from_samples(all_samples)}

            dat = {}
            dat['mean'] = np.asarray(all_samples).mean(axis=0)
            from numpy import asarray, linalg
            from sklearn.covariance import GraphicalLassoCV
            model = GraphicalLassoCV()
            model.fit(asarray(all_samples))
            dat['precision'] = linalg.cholesky(model.precision_)

            with open("assets/mano_pose_prior.pkl", "wb") as f:
                pkl.dump(dat, f)

        else:
            import pickle as pkl
            # Load pre-computed mean and variance
            dat = pkl.load(open('assets/mano_pose_prior.pkl', 'rb'))
            # self.priors = dat
            self.priors = {'Generic': th_Mahalanobis(dat['mean'],
                           dat['precision'],
                           self.prefix)}

    def create_prior_from_samples(self, samples):
        from sklearn.covariance import GraphicalLassoCV
        from numpy import asarray, linalg
        model = GraphicalLassoCV()
        model.fit(asarray(samples))
        return th_Mahalanobis(asarray(samples).mean(axis=0),
                           linalg.cholesky(model.precision_),
                           self.prefix)

    def __getitem__(self, pid):
        if pid not in self.priors:
            samples = [p[self.prefix:] for qsub in self.pose_subjects
                       for name, p in zip(qsub['pose_fnames'], qsub['pose_parms'])
                       if pid in name.lower()]
            self.priors[pid] = self.priors['Generic'] if len(samples) < 3 \
                               else self.create_prior_from_samples(samples)

        return self.priors[pid]

import os
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    Prior(sm='xx')

    get_prior(gender='male', precomputed=True)