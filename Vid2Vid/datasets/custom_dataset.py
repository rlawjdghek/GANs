import os
from os.path import join as opj
from glob import glob

import torch

from .base_dataset import *

class VideoDataset(BaseDataset):
    def __init__(self, args, is_train):
        super().__init__(args)
        self.is_train = is_train
        self.torv = "train" if is_train else "valid"
        self.dir_A = opj(args.data_root_dir, args.data_name, self.torv, "ncct")
        self.dir_B = opj(args.data_root_dir, args.data_name, self.torv, "cect")
        self.paths_A = sorted(make_grouped_dataset(self.dir_A))  # make_gruoped_dataset은 재귀로 들어가서 하나의 폴더 (동영상)에 있는 이미지들을 정렬. 따라서 self.paths_A의 길이는 동영상의 갯수. 
        self.paths_B = sorted(make_grouped_dataset(self.dir_B))
        self.sum_frames = sum(list(map(len, self.paths_A)))
        self.check_paths()
        self.n_of_seqs = len(self.paths_A)
        self.seq_len_max = max([len(A) for A in self.paths_A])
        self.n_frames_total = args.n_frames_total
    def check_paths(self):
        assert len(self.paths_A) == len(self.paths_B), f"path lengths are not equal!! : {len(self.paths_A)} vs {len(self.paths_B)}"
        if self.args.local_rank == 0:
            print(f"# of A : {len(self.paths_A)}, B : {len(self.paths_B)}")
    def __len__(self):
        return len(self.paths_A)
    def __getitem__(self, idx):  # 여기의 idx는 동영상의 인덱스
        paths_A = self.paths_A[idx % self.n_of_seqs]
        paths_B = self.paths_B[idx % self.n_of_seqs]
        n_frames_total, start_idx, t_step = get_video_params(self.args, self.n_frames_total, len(paths_A), idx, is_train=self.is_train)
        img_B = Image.open(paths_B[start_idx]).convert("RGB")
        params = get_img_params()
        transform = get_transforms(self.args, params, is_train=self.is_train)
        A = B = 0
        for i in range(n_frames_total):
            path_A = paths_A[start_idx + i*t_step]
            path_B = paths_B[start_idx + i*t_step]
            img_A = Image.open(path_A).convert("RGB")
            img_B = Image.open(path_B).convert("RGB")
            img_A = transform(img_A)[:self.args.input_ch]
            img_B = transform(img_B)[:self.args.input_ch]

            A = img_A if i==0 else torch.cat([A, img_A], dim=0)
            B = img_B if i==0 else torch.cat([B, img_B], dim=0)

        return {"A": A, "B": B, "n_frames_total": n_frames_total}   
class VideoTestDataset(BaseDataset):
    def __init__(self, args, test_data_name):
        super().__init__(args)
        self.dir_A = opj(args.data_root_dir, args.data_name, test_data_name, "ncct")
        self.dir_B = opj(args.data_root_dir, args.data_name, test_data_name, "cect")
        self.paths_A = sorted(make_grouped_dataset(self.dir_A))
        self.paths_B = sorted(make_grouped_dataset(self.dir_B))
        self.sum_frames = sum(list(map(len, self.paths_A)))
        self.check_paths()
        self.n_of_seqs = len(self.paths_A)
        self.seq_len_max = max([len(A) for A in self.paths_A])
        self.n_frames_total = args.n_frames_total
        T_lst = [
            T.Resize((args.img_size, args.img_size), interpolation=Image.BICUBIC),
            T.ToTensor()
        ]
        if args.input_ch == 1: T_lst.append(T.Normalize((0.5), (0.5)))
        elif args.input_ch == 3: T_lst.append(T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)))
        self.transform = T.Compose(T_lst)        
    def check_paths(self):
        assert len(self.paths_A) == len(self.paths_B), f"path lengths are not equal!! : {len(self.paths_A)} vs {len(self.paths_B)}"
        if self.args.local_rank == 0:
            print(f"# of A : {len(self.paths_A)}, B : {len(self.paths_B)}")
    def __len__(self):
        return len(self.paths_A)
    def __getitem__(self, idx):
        paths_A = self.paths_A[idx]
        paths_B = self.paths_B[idx]
        assert len(paths_A) == len(paths_B), f"video length is not equal: {self.dir_A[idx]} - {len(paths_A)} vs {len(paths_B)}"
    
        n_frames = len(paths_A)
        for i in range(n_frames):
            path_A = paths_A[i]
            path_B = paths_B[i]
            img_A = Image.open(path_A).convert("RGB")
            img_B = Image.open(path_B).convert("RGB")
            img_A = self.transform(img_A)[:self.args.input_ch]
            img_B = self.transform(img_B)[:self.args.input_ch]
            A = img_A if i==0 else torch.cat([A, img_A], dim=0)
            B = img_B if i==0 else torch.cat([B, img_B], dim=0)
        return {"A":A, "B": B}