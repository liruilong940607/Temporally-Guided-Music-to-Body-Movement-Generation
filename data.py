import os
import gdown
import pickle
import librosa
import numpy as np

import torch
from torch.utils.data.dataset import Dataset


class Download():
    def __init__(self):
        self.data_dst = 'data/'
        self.checkpoint_dst = 'checkpoint/'
        if not os.path.exists(self.data_dst):
            os.makedirs(self.data_dst)
        if not os.path.exists(self.checkpoint_dst):
            os.makedirs(self.checkpoint_dst)
        self.train_dst = self.data_dst + 'train.pkl'
        self.test_dst = self.data_dst + 'test.pkl'
        self.wav_dst = self.data_dst + 'flower.wav'
        self.pretrain_model_dst = self.checkpoint_dst + 'checkpoint082820.pth'
        self.train_url = 'https://drive.google.com/uc?id=1QsghRzGwgzZBQz03MqtWZ0S7X0Y6NivC&export=download'
        self.test_url = 'https://drive.google.com/u/0/uc?id=1WQksHdEH65xES557nkbsIuNM69vSdtYq&export=download'
        self.wav_url = 'https://drive.google.com/u/0/uc?id=1WwSMkhe5ga0GQdk9OC4atfVaAWkPNd3X&export=download'
        self.pretrain_model_url = 'https://drive.google.com/u/0/uc?id=1uFrYoJJ5pGosN0vl87F0_cS9fkoWLGB1&export=download'
    
    def train_data(self):
        if not os.path.exists(self.train_dst):
            gdown.download(self.train_url, self.train_dst)
    
    def test_data(self):
        if not os.path.exists(self.test_dst):
            gdown.download(self.test_url, self.test_dst)
        if not os.path.exists(self.wav_dst):
            gdown.download(self.wav_url, self.wav_dst)
            
    def pretrain_model(self):
        if not os.path.exists(self.pretrain_model_dst):
            gdown.download(self.pretrain_model_url, self.pretrain_model_dst)

class audio_skeleton_dataset(Dataset):
    """
        aud: MFCC feature, size [N, T, D]
        keypoints: skeleton feature, size [N, T, (K*3)]
        seq_len: length of each sequence, size [N] 
    """
    def __init__(self, root, split, is_aist=False, **kwargs):
        self.is_aist = is_aist
        if is_aist:
            if os.path.exists("./data/aist_aux.npy"):
                print ("load aist aux data")
                aux_data = np.load("./data/aist_aux.npy", allow_pickle=True).item()
                self.aud_mean =  0.0 # aux_data["aud_mean"]
                self.aud_std =   1.0 # aux_data["aud_std"]
                self.keypoints_mean = 0.0 # aux_data["keypoints_mean"]
                self.keypoints_std = 1.0 # aux_data["keypoints_std"]
            else:
                self.aud_mean = 0.0
                self.aud_std = 1.0
                self.keypoints_mean = 0.0
                self.keypoints_std = 1.0
            self.init_aist_data(root, split, **kwargs)
        else:
            self.init_raw_data(root, split)
    
    def init_aist_data(self, root, split, **kwargs):
        print (kwargs)
        assert ("smpl_dir" in kwargs) and ("audio_dir" in kwargs)
        from aist_plusplus.loader import AISTDataset
        from tqdm import tqdm
        from smplx import SMPL

        aist_dataset = AISTDataset(root)
        smpl = SMPL(model_path=kwargs["smpl_dir"], gender='MALE', batch_size=1)

        seq_names_ignore = [
            f.strip() for f in open(
                os.path.join(root, "ignore_list.txt"), "r"
                ).readlines()]
        seq_names_train = [
            f.strip() for f in open(
                os.path.join(root, "splits/crossmodal_train.txt"), "r"
                ).readlines()]
        seq_names_val = [
            f.strip() for f in open(
                os.path.join(root, "splits/crossmodal_val.txt"), "r"
                ).readlines()]
        seq_names_test = [
            f.strip() for f in open(
                os.path.join(root, "splits/crossmodal_test.txt"), "r"
                ).readlines()]
        seq_names_testval = seq_names_val + seq_names_test
        seq_names_train = [f for f in seq_names_train if f not in seq_names_ignore]
        seq_names_testval = [f for f in seq_names_testval if f not in seq_names_ignore]

        self.seq_names = seq_names_train if split == "train" \
            else (seq_names_train + seq_names_testval if split == "all" else seq_names_testval)

        audio_names = list(set([f.split("_")[-2] for f in self.seq_names]))
        audio_feats = {}
        print (f"loading AIST++ audio. split={split}; len={len(audio_names)}")
        for audio_name in tqdm(audio_names):
            audio_file = os.path.join(kwargs["audio_dir"], audio_name+".wav")
            aud_feat = preprocess(audio_file, self.aud_mean, self.aud_std, sr=30720, hop=512)  # 60fps
            audio_feats[audio_name] = aud_feat

        self.aud = []
        self.keypoints = []
        self.seq_len = []
        print (f"loading AIST++ motion. split={split}; len={len(self.seq_names)}")
        for seq_name in tqdm(self.seq_names):
            audio_name = seq_name.split("_")[-2]
            smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(
                aist_dataset.motion_dir, seq_name)
            keypoints3d = smpl.forward(
                global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
                body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
                transl=torch.from_numpy(smpl_trans / smpl_scaling).float(),
                ).joints.detach().numpy()[:, 0:24, :]
            nframes = keypoints3d.shape[0]
            keypoints3d = keypoints3d.reshape(nframes, -1) # [T, (K*3)]
            self.aud.append(audio_feats[audio_name][:nframes])
            self.keypoints.append((keypoints3d - self.keypoints_mean) / self.keypoints_std)
            self.seq_len.append(nframes)
        
    def init_raw_data(self, root, split):
        with open(root, 'rb') as f:
            self.Data = pickle.load(f)
            
        self.data = self.Data[split]
        self.aud = self.data['aud']
        self.keypoints = self.data['keypoints']
        self.seq_len = self.data['seq_len']

    def __getitem__(self, index):
        aud = self.aud[index]
        keypoints = self.keypoints[index]
        seq_len = self.seq_len[index]
        
        if self.is_aist:
            max_len = 1200
            if seq_len >= max_len:
                start_id = np.random.randint(0, seq_len-max_len)
                aud = aud[start_id:start_id+max_len]
                keypoints = keypoints[start_id:start_id+max_len]
                seq_len = max_len
            else:
                aud = np.concatenate([
                    aud, np.zeros((max_len-seq_len, aud.shape[1]))], axis=0)
                keypoints = np.concatenate([
                    keypoints, np.zeros((max_len-seq_len, keypoints.shape[1]))], axis=0)
        aud = torch.tensor(aud, dtype=torch.float32).to('cuda:0' if torch.cuda.is_available() else 'cpu')
        keypoints = torch.tensor(keypoints, dtype=torch.float32).to('cuda:0' if torch.cuda.is_available() else 'cpu')
        seq_len = torch.tensor(seq_len).to('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        return aud, keypoints, seq_len

    def __len__(self):
        return len(self.aud)

def preprocess(audio, aud_mean, aud_std, sr=44100, hop=1470):
    """ Extract MFCC feature """
    n_fft = 4096
    y, sr = librosa.load(audio, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mfcc=13)
    energy = np.log(librosa.feature.rmse(y=y, frame_length=n_fft, hop_length=hop, center=True))
    mfcc_energy = np.vstack((mfcc, energy))
    mfcc_delta = librosa.feature.delta(mfcc_energy)
    aud = np.vstack((mfcc_energy, mfcc_delta)).T
    aud = (aud - aud_mean) / (aud_std + 1E-8)
    return aud

if __name__ == "__main__":
    dataset = audio_skeleton_dataset(
        root = "/media/ruilongli/hd1/Data/aist_plusplus/v1",
        split = "all",
        smpl_dir = "/media/ruilongli/hd1/Data/smpl",
        audio_dir = "/media/ruilongli/hd1/Data/aist_plusplus/wav",
        is_aist = True,
    )

    aud = np.concatenate(dataset.aud, axis=0) 
    keypoints = np.concatenate(dataset.keypoints, axis=0)
    aud_dim = aud.shape[-1]
    keypoints_dim = keypoints.shape[-1]

    print (aud.shape, keypoints.shape)    
    aud_mean = np.mean(aud.reshape(-1, aud_dim), axis=0)
    aud_std = np.std(aud.reshape(-1, aud_dim), axis=0)
    
    keypoints_mean = np.mean(keypoints.reshape(-1, keypoints_dim), axis=0)
    keypoints_std = np.std(keypoints.reshape(-1, keypoints_dim), axis=0)
    
    os.makedirs("./data", exist_ok=True)
    np.save("./data/aist_aux.npy", {
        "aud_mean": aud_mean,
        "aud_std": aud_std,
        "keypoints_mean": keypoints_mean,
        "keypoints_std": keypoints_std,
        })