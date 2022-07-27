"""
Dataloader for the network.
Author: Bharat
Cite: LoopReg: Self-supervised Learning of Implicit Surface Correspondences, Pose and Shape for 3D Human Mesh Registration, NeurIPS' 20.
"""

import os
from os.path import join, split, exists

import sys
from pathlib import Path

folder_root = Path().resolve()
sys.path.append(str(folder_root))

from models.volumetric_SMPL import VolumetricSMPL

import pickle as pkl
import numpy as np
from glob import glob
import codecs
# from kaolin.rep import TriangleMesh as tm
import trimesh
from psbody.mesh import Mesh
from lib.smpl_paths import SmplPaths
from torch.utils.data import Dataset, DataLoader
from make_data_split import DATA_PATH
from pathlib import Path
import torch
import math

# Number of points to sample from the scan
NUM_POINTS = 30000

class MyDataLoader(Dataset):
    def __init__(self, mode, batch_sz, num_labels,  data_path=DATA_PATH,
                 split_file='assets/data_split_01.pkl', num_workers=12,
                 augment=False, naked=False):
        self.mode = mode
        self.path = data_path
        with open(split_file, "rb") as f:
            self.split = pkl.load(f)

        self.data = self.split[mode]
        self.batch_size = batch_sz
        self.num_workers = num_workers
        self.augment = augment
        self.naked = naked
        sp = SmplPaths(gender='male')
        self.ref_smpl = sp.get_smpl()
        # self.vt, self.ft = sp.get_vt_ft()

        # Load smpl part labels
        if num_labels == 16:
            self.smpl_parts = np.loadtxt('assets/mano_label_right.txt').reshape(-1, 1)
        else:
            with open('assets/mano_parts_dense.pkl', 'rb') as f:
                dat = pkl.load(f, encoding='latin-1')
            # self.smpl_parts = np.zeros((6890, 1))
            self.smpl_parts = np.zeros((778, 1))
            for n, k in enumerate(dat):
                self.smpl_parts[dat[k]] = n
        print(np.unique(self.smpl_parts))

    def __len__(self):
        return len(self.data)

    def get_loader(self, shuffle=True):
        return DataLoader(self, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=shuffle)

    @staticmethod
    def worker_init_fn(worker_id):
        """
        Worker init function to ensure true randomness.
        """
        base_seed = int(codecs.encode(os.urandom(4), 'hex'), 16)
        np.random.seed(base_seed + worker_id)

    # @staticmethod
    # def worker_init_fn(worker_id):
    #     worker_info = torch.utils.data.get_worker_info()
    #     dataset = worker_info.dataset  # the dataset copy in this worker process
    #     overall_start = dataset.start
    #     overall_end = dataset.end
    #     # configure the dataset to only process the split workload
    #     per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    #     worker_id = worker_info.id
    #     dataset.start = overall_start + worker_id * per_worker
    #     dataset.end = min(dataset.start + per_worker, overall_end)
    #     base_seed = int(codecs.encode(os.urandom(4), 'hex'), 16)
    #     np.random.seed(base_seed + worker_id)


    @staticmethod
    def map_mesh_points_to_reference(pts, src, ref):
        """
        Finds closest points to pts on src.
        Maps the closest points on src to ref.
        """
        closest_face, closest_points = src.closest_faces_and_points(pts)
        vert_ids, bary_coords = src.barycentric_coordinates_for_points(closest_points, closest_face.astype('int32'))
        correspondences = (ref[vert_ids] * bary_coords[..., np.newaxis]).sum(axis=1)

        return correspondences

    @staticmethod
    def map_vitruvian_vertex_color(tgt_vertices, registered_smpl_mesh,
                                   path_to_cols='assets/mano_cols.npy'):
        """
        Vitruvian vertex color are defined for SMPL mesh. This function maps these colors from registered smpl to scan.
        """
        col = np.load(path_to_cols)
        vids, _ = registered_smpl_mesh.closest_vertices(tgt_vertices)
        vids = np.array(vids)
        return col[vids]

    @staticmethod
    def get_rnd_rotations():
        '''We want 2*pi rotation along z-axis and very small perturbations along x,y-axis'''
        from scipy.spatial.transform import Rotation as R
        rots = np.random.rand(1, 3)
        rots[:, 0] *= np.pi * 0.01
        rots[:, 2] *= np.pi * 0.01
        rots[:, 1] *= np.pi * 2
        t = R.from_rotvec(rots)
        return t

    def __getitem__(self, idx):
        path = self.data[idx]

        mpi_data_root = Path(path).parent.parent

        name = split(path)[1]
        # print(idx, name)
        if "_deobj.obj" in name:
            name = name.replace("_deobj.obj", ".ply")

        input_smpl = Mesh(filename= str(mpi_data_root / 'handsOnly_REGISTRATIONS_r_lm___POSES' / name))
        input_scan = Mesh(filename=path)

        # input_smpl = Mesh(filename=join(path, name + '_smpl.obj'))
        # if self.naked:
        #     input_scan = Mesh(filename=join(path, name + '_smpl.obj'))
        # else:
        #     input_scan = Mesh(filename=join(path, name + '.obj'))
        temp = trimesh.Trimesh(vertices=input_scan.v, faces=input_scan.f)
        points = temp.sample(NUM_POINTS)

        if self.augment:
            rot = self.get_rnd_rotations()
            points = rot.apply(points)
            input_smpl.v = rot.apply(input_smpl.v)

        ind, _ = input_smpl.closest_vertices(points, use_cgal=True)
        part_labels = self.smpl_parts[np.array(ind)]
        correspondences = self.map_mesh_points_to_reference(points, input_smpl, self.ref_smpl.r)

        if self.mode == 'train':
            # print(idx, name, "return")
            return {'scan': points.astype('float32'),
                    'correspondences': correspondences.astype('float32'),
                    'part_labels': part_labels.astype('float32'),
                    'name': path
                    }

        vc = self.map_vitruvian_vertex_color(points, input_smpl)
        return {'scan': points.astype('float32'),
                'smpl': input_smpl.v.astype('float32'),
                'correspondences': correspondences.astype('float32'),
                'part_labels': part_labels.astype('float32'),
                'scan_vc': vc,
                'name': path
                }

class MyDataLoaderCacher(MyDataLoader):
    """
    Loads scan points, cached SMPL parameters, GT correspondences.
    """

    def __init__(self, mode, batch_sz, num_labels,  pose_ncomp, data_path=DATA_PATH,
                 split_file='assets/data_split_01.pkl',
                 cache_suffix=None,
                 num_workers=12, augment=False, naked=False):
        self.mode = mode
        self.cache_suffix = cache_suffix
        self.path = data_path
        self.pose_ncomp = pose_ncomp
        with open(split_file, "rb") as f:
            self.split = pkl.load(f)

        self.data = self.split[mode]
        self.batch_size = batch_sz
        self.num_workers = num_workers
        self.augment = augment
        self.naked = naked
        sp = SmplPaths(gender='male')
        self.ref_smpl = sp.get_smpl()
        self.vt, self.ft = sp.get_vt_ft()

        # Load smpl part labels
        if num_labels == 16:
            self.smpl_parts = np.loadtxt('assets/mano_label_right.txt').reshape(-1, 1)
        else:
            with open('assets/mano_parts_dense.pkl', 'rb') as f:
                dat = pkl.load(f, encoding='latin-1')
            # self.smpl_parts = np.zeros((6890, 1))
            self.smpl_parts = np.zeros((778, 1))
            for n, k in enumerate(dat):
                self.smpl_parts[dat[k]] = n
        print(np.unique(self.smpl_parts))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data[idx]
        mpi_data_root = Path(path).parent.parent
        
        name = split(path)[1]
        # print(idx, name)
        if "_deobj.obj" in name:
            name = name.replace("_deobj.obj", ".ply")

        # input_smpl = Mesh(filename=join(path, name + '_smpl.obj'))
        # if self.naked:
        #     input_scan = Mesh(filename=join(path, name + '_smpl.obj'))
        # else:
        #     input_scan = Mesh(filename=join(path, name + '.obj'))

        input_smpl = Mesh(filename=str(mpi_data_root / 'handsOnly_REGISTRATIONS_r_lm___POSES' / name))
        input_scan = Mesh(filename=path)

        temp = trimesh.Trimesh(vertices=input_scan.v, faces=input_scan.f)
        points = temp.sample(NUM_POINTS)

        if self.augment:
            rot = self.get_rnd_rotations()
            points = rot.apply(points)
            input_smpl.v = rot.apply(input_smpl.v)

        ind, _ = input_smpl.closest_vertices(points, use_cgal=True)
        part_labels = self.smpl_parts[np.array(ind)]
        correspondences = self.map_mesh_points_to_reference(points, input_smpl, self.ref_smpl.r)
        # print(idx, name, "finish_compute")
        # Load cached SMPL params
        cache_list = []
        if self.cache_suffix is not None:
            cache_list = sorted(glob(join(path, self.cache_suffix, '*.pkl')))
        if len(cache_list) > 0:
            smpl_dict = pkl.load(open(cache_list[-1], 'rb'), encoding='latin-1')
            pose = smpl_dict['pose']
            betas = smpl_dict['betas']
            trans = smpl_dict['trans']
            # print('Loading from cache ', cache_list[-1])
        else:
            # pose = np.zeros((45+3,))
            pose = np.zeros((3+self.pose_ncomp))
            betas = np.zeros((10,))
            trans = np.zeros((3,))

        if self.mode == 'train':
            # print(idx, name, "return")
            return {'scan': points.astype('float32'),
                    'correspondences': correspondences.astype('float32'),
                    'part_labels': part_labels.astype('float32'),
                    'pose': pose.astype('float32'),
                    'betas': betas.astype('float32'),
                    'trans': trans.astype('float32'),
                    'name': path
                    }

        vc = self.map_vitruvian_vertex_color(points, input_smpl)
        return {'scan': points.astype('float32'),
                'smpl': input_smpl.v.astype('float32'),
                'correspondences': correspondences.astype('float32'),
                'part_labels': part_labels.astype('float32'),
                'pose': pose.astype('float32'),
                'betas': betas.astype('float32'),
                'trans': trans.astype('float32'),
                'scan_vc': vc,
                'name': path
                }

from tqdm import tqdm
if __name__ == "__main__":
    batch_size = 3
    num_labels= 6
    augment = False
    naked = False
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    split_file = "assets/hand_data_split_01_deobj.pkl"
    dataset = MyDataLoader('train', batch_size, num_labels, num_workers=0, augment=augment, 
                            naked=naked, split_file=split_file).get_loader()


    # dataset = MyDataLoaderCacher('train', batch_size, num_labels, 10, cache_suffix="cache", 
    #                                  split_file=split_file,
    #                                  num_workers=16, 
    #                                  augment=False, naked=naked).get_loader(shuffle=False)


    vsmpl = VolumetricSMPL('assets/volumetric_mano_function_64_1.6', torch.device("cuda"), 'male')
    loop = tqdm(dataset)
    for n, batch in enumerate(loop):
        # print(n)
        loop.set_description("%d" % n)