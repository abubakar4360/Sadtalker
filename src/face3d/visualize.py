import cv2
import numpy as np
from src.face3d.models.bfm import ParametricFaceModel
from src.face3d.models.facerecon_model import FaceReconModel
import torch


from torch.nn.parallel import DataParallel
import subprocess, platform
import scipy.io as scio
from tqdm import tqdm


class CoeffFirstModel(torch.nn.Module):
    def __init__(self, coeff_first):
        super(CoeffFirstModel, self).__init__()
        self.coeff_first = coeff_first

    def forward(self):
        return self.coeff_first

class CoeffPredModel(torch.nn.Module):
    def __init__(self, coeff_pred):
        super(CoeffPredModel, self).__init__()
        self.coeff_pred = coeff_pred

    def forward(self):
        return self.coeff_pred

class DataParallelModel(torch.nn.Module):
    def __int__(self, model, device):
        super(DataParallelModel, self).__int__()
        self.model = DataParallelModel(model, device =device)

    def forward(self, *inputs):
        return self.Module(*inputs)

##Main function
def gen_composed_video(args, device, first_frame_coeff, coeff_path, audio_path, save_path, exp_dim=64):

    coeff_first_data = scio.loadmat(first_frame_coeff)['full_3dmm']
    coeff_pred_data = scio.loadmat(coeff_path)['coeff_3dmm']

    coeff_first_model = DataParallelModel(CoeffFirstModel(torch.tensor(coeff_first_data, device=device)))
    coeff_pred_model = DataParallelModel(CoeffPredModel(torch.tensor(coeff_pred_data, device=device)))

    # coeff_full = np.repeat(coeff_first_model, coeff_pred_model.shape[0], axis=0)  # 257
    coeff_full = np.repeat(coeff_first_model.module.coeff_first, coeff_pred_model.module.coeff_pred.shape[0],
                           axis=0)  # 257

    # coeff_full[:, 80:144] = coeff_pred_model[:, 0:64]
    # coeff_full[:, 224:227] = coeff_pred_model[:, 64:67]  # 3 dim translation
    # coeff_full[:, 254:] = coeff_pred_model[:, 67:]  # 3 dim translation

    coeff_full[:, 80:144] = coeff_pred_model.module.coeff_pred[:, 0:64]
    coeff_full[:, 224:227] = coeff_pred_model.module.coeff_pred[:, 64:67]  # 3 dim translation
    coeff_full[:, 254:] = coeff_pred_model.module.coeff_pred[:, 67:]  # 3 dim translation

    tmp_video_path = '/tmp/face3dtmp.mp4'

    facemodel = DataParallelModel(FaceReconModel(args))
    facemodel = facemodel.to(device)

    video = cv2.VideoWriter(tmp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (224, 224))

    for k in tqdm(range(coeff_pred_data.shape[0]), 'face3d rendering:'):
        # cur_coeff_first = coeff_first_model.module()  # Access the underlying module
        # cur_coeff_pred = coeff_pred_model.module()    # Access the underlying module

        # Combine the coefficients if necessary
        # coeff_full = torch.tensor(coeff_full[k:k+1], device=device)
        coeff_full = torch.tensor(coeff_full[k:k + 1])
        cur_coeff_full = coeff_full.to(device[0])  # Use the first device

        facemodel.forward(cur_coeff_full, device)

        predicted_landmark = facemodel.pred_lm  # TODO.
        predicted_landmark = predicted_landmark.cpu().numpy().squeeze()

        rendered_img = facemodel.pred_face
        # rendered_img = 255. * rendered_img.cpu().numpy().squeeze().transpose(1, 2, 0)
        rendered_img = (255. * rendered_img).squeeze().permute(1, 2, 1).cpu().numpy()
        out_img = rendered_img[:, :, :3].astype(np.uint8)

        video.write(np.uint8(out_img[:, :, ::-1]))

    video.release()

    command = 'ffmpeg -v quiet -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, tmp_video_path, save_path)
    subprocess.call(command, shell=platform.system() != 'Windows')

# Usage
# gen_composed_video(args, device, first_frame_coeff, coeff_path, audio_path, save_path, exp_dim=64)
