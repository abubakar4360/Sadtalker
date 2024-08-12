from glob import glob
import shutil
import torch
from time import strftime
import os, sys, time
from argparse import ArgumentParser
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data, updated_get_data
from src.generate_facerender_batch import get_facerender_data
from src.utils.init_path import init_path
from torch.nn.parallel import DataParallel

import torch.nn as nn

class CustomCropAndExtractWrapper(nn.Module):
    def __init__(self, sadtalker_paths, device):
        super(CustomCropAndExtractWrapper, self).__init__()
        self.crop_and_extract = CropAndExtract(sadtalker_paths, device)

    def generate(self, input_path, save_dir, crop_or_resize='crop', source_image_flag=False, pic_size=256):
        return self.crop_and_extract.generate(input_path, save_dir, crop_or_resize, source_image_flag, pic_size)

    def forward(self, input_path, save_dir, crop_or_resize='crop', source_image_flag=False, pic_size=256):
        return self.generate(input_path, save_dir, crop_or_resize, source_image_flag, pic_size)

class CustomAudio2CoeffWrapper(nn.Module):
    def __init__(self, sadtalker_paths, device):
        super(CustomAudio2CoeffWrapper, self).__init__()
        self.audio2coeff = Audio2Coeff(sadtalker_paths, device)

    def generate(self, batch, save_dir, pose_style, ref_pose_coeff_path):
        # Process each batch item individually and gather results
        results = []
        for item in batch:
            # print("Item:", item)
            result = self.audio2coeff.generate(item, save_dir, pose_style, ref_pose_coeff_path)
            results.append(result)
        return results

    def forward(self,batch, save_dir, pose_style, ref_pose_coeff_path):
        return self.generate(batch, save_dir, pose_style, ref_pose_coeff_path)

class CustomAnimateFromCoeffWrapper(nn.Module):
    def __init__(self, sadtalker_paths, device):
        super(CustomAnimateFromCoeffWrapper, self).__init__()
        self.animate_from_coeff = AnimateFromCoeff(sadtalker_paths, device)

    def generate(self, data, save_dir, pic_path, crop_info, enhancer=None, background_enhancer=None, preprocess=None, img_size=None):
        # Process each data item individually and gather results
        results = []
        for item in data:
            result = self.animate_from_coeff.generate(item, save_dir, pic_path, crop_info, enhancer, background_enhancer, preprocess, img_size)
            results.append(result)
        return results

    def forward(self, data, save_dir, pic_path, crop_info, enhancer=None, background_enhancer=None, preprocess=None, img_size=None):
        return self.generate(data, save_dir, pic_path, crop_info, enhancer, background_enhancer, preprocess, img_size)


def main(args):
    pic_path = args.source_image
    audio_path = args.driven_audio
    save_dir = os.path.join(args.result_dir, strftime("%Y_%m_%d_%H.%M.%S"))
    os.makedirs(save_dir, exist_ok=True)
    pose_style = args.pose_style
    device = args.device
    batch_size = args.batch_size
    input_yaw_list = args.input_yaw
    input_pitch_list = args.input_pitch
    input_roll_list = args.input_roll
    ref_eyeblink = args.ref_eyeblink
    ref_pose = args.ref_pose

    current_root_path = os.path.split(sys.argv[0])[0]

    sadtalker_paths = init_path(args.checkpoint_dir, os.path.join(current_root_path, 'src/config'), args.size,
                                args.old_version, args.preprocess)

    # Initialize CustomAnimateFromCoeffWrapper, CustomAudio2CoeffWrapper, and CustomCropAndExtractWrapper

    preprocess_model = CustomCropAndExtractWrapper(sadtalker_paths, device)
    audio_to_coeff = CustomAudio2CoeffWrapper(sadtalker_paths, device)
    animate_from_coeff = CustomAnimateFromCoeffWrapper(sadtalker_paths, device)


    # Wrap models with DataParallel
    preprocess_model = DataParallel(preprocess_model)
    audio_to_coeff = DataParallel(audio_to_coeff)
    animate_from_coeff = DataParallel(animate_from_coeff)


    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
        preprocess_model.to(device)
        audio_to_coeff.to(device)
        animate_from_coeff.to(device)
        batch_size = args.batch_size * torch.cuda.device_count()
    else:
        args.device = "cpu"

    # Crop image and extract 3DMM from image
    first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
    os.makedirs(first_frame_dir, exist_ok=True)
    print('3DMM Extraction for source image')

    first_coeff_path, crop_pic_path, crop_info = preprocess_model.module.generate(pic_path, first_frame_dir, args.preprocess, \
                                                                           source_image_flag=True, pic_size=args.size)
    if first_coeff_path is None:
        print("Can't get the coeffs of the input")
        return

    if ref_eyeblink is not None:
        ref_eyeblink_videoname = os.path.splitext(os.path.split(ref_eyeblink)[-1])[0]
        ref_eyeblink_frame_dir = os.path.join(save_dir, ref_eyeblink_videoname)
        os.makedirs(ref_eyeblink_frame_dir, exist_ok=True)
        print('3DMM Extraction for the reference video providing eye blinking')
        ref_eyeblink_coeff_path, _, _ = preprocess_model.generate(ref_eyeblink, ref_eyeblink_frame_dir, args.preprocess,
                                                                  source_image_flag=False)
    else:
        ref_eyeblink_coeff_path = None

    if ref_pose is not None:
        if ref_pose == ref_eyeblink:
            ref_pose_coeff_path = ref_eyeblink_coeff_path
        else:
            ref_pose_videoname = os.path.splitext(os.path.split(ref_pose)[-1])[0]
            ref_pose_frame_dir = os.path.join(save_dir, ref_pose_videoname)
            os.makedirs(ref_pose_frame_dir, exist_ok=True)
            print('3DMM Extraction for the reference video providing pose')
            ref_pose_coeff_path, _, _ = preprocess_model.generate(ref_pose, ref_pose_frame_dir, args.preprocess,
                                                                  source_image_flag=False)
    else:
        ref_pose_coeff_path = None

    global indiv_mels, ref_coeff, num_frames, ratio, audio_name, pic_name
    # indiv_mels, ref_coeff, num_frames, ratio, audio_name, pic_name = updated_get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)
    batch = get_data(first_coeff_path, audio_path, device, ref_eyeblink_coeff_path, still=args.still)

    # Audio to coefficient
    num_gpus = torch.cuda.device_count() if args.device == "cuda" else 1
    sub_batch_size = batch_size // num_gpus

    num_frames = batch['num_frames']
    indiv_mels = batch['indiv_mels']
    ref_coeff = batch['ref']
    ratio = batch['ratio_gt']
    audio_name = batch['audio_name']
    pic_name = batch['pic_name']

    for i in range(0, 2, sub_batch_size):
        start_idx = i
        end_idx = min(i + sub_batch_size, num_frames)
        # sub_batch_indiv_mels = indiv_mels[start_idx:end_idx, :, :, :, :]
        sub_batch_indiv_mels = indiv_mels[:, :, :, :, ]
        sub_ratio = ratio[start_idx:end_idx, :]

        updated_batch = {'indiv_mels': sub_batch_indiv_mels,'ref': ref_coeff,
                         'num_frames': num_frames,'ratio_gt': sub_ratio,
                         'audio_name': audio_name, 'pic_name': pic_name}
        sub_batch = [updated_batch]

        coeff_path = audio_to_coeff.module.generate(sub_batch, save_dir, pose_style, ref_pose_coeff_path)

    # 3D face render
    if args.face3dvis:
        from src.face3d.visualize import gen_composed_video
        gen_composed_video(args, device, first_coeff_path, coeff_path, audio_path, os.path.join(save_dir, '3dface.mp4'))

    # Coefficient to video
    data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path,
                               args.batch_size, input_yaw_list, input_pitch_list, input_roll_list,
                               expression_scale=args.expression_scale, still_mode=args.still,
                               preprocess=args.preprocess, size=args.size)

    for i in range(0, len(data), sub_batch_size):
        sub_data = data[i:i + sub_batch_size]
        result = animate_from_coeff.module.generate(sub_data, save_dir, pic_path, crop_info,
                                                    enhancer=args.enhancer, background_enhancer=args.background_enhancer,
                                                    preprocess=args.preprocess, img_size=args.size)

    shutil.move(result, save_dir + '.mp4')
    print('The generated video is named:', save_dir + '.mp4')

    if not args.verbose:
        shutil.rmtree(save_dir)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--driven_audio", default='./examples/driven_audio/bus_chinese.wav',
                        help="path to driven audio")
    parser.add_argument("--source_image", default='./examples/source_image/full_body_1.png',
                        help="path to source image")
    parser.add_argument("--ref_eyeblink", default=None, help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None, help="path to reference video providing pose")
    parser.add_argument("--checkpoint_dir", default='./checkpoints', help="path to output")
    parser.add_argument("--result_dir", default='./results', help="path to output")
    parser.add_argument("--pose_style", type=int, default=0, help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2, help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=256, help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float, default=1., help="the batch size of facerender")
    parser.add_argument('--input_yaw', nargs='+', type=int, default=None, help="the input yaw degree of the user ")
    parser.add_argument('--input_pitch', nargs='+', type=int, default=None, help="the input pitch degree of the user")
    parser.add_argument('--input_roll', nargs='+', type=int, default=None, help="the input roll degree of the user")
    parser.add_argument('--enhancer', type=str, default=None, help="Face enhancer, [gfpgan, RestoreFormer]")
    parser.add_argument('--background_enhancer', type=str, default=None, help="background enhancer, [realesrgan]")
    parser.add_argument("--cpu", dest="cpu", action="store_true")
    parser.add_argument("--face3dvis", action="store_true", help="generate 3d face and 3d landmarks")
    parser.add_argument("--still", action="store_true",
                        help="can crop back to the original videos for the full body animation")
    parser.add_argument("--preprocess", default='crop', choices=['crop', 'extcrop', 'resize', 'full', 'extfull'],
                        help="how to preprocess the images")
    parser.add_argument("--verbose", action="store_true", help="saving the intermediate output or not")
    parser.add_argument("--old_version", action="store_true", help="use the pth other than safetensor version")

    # net structure and parameters
    parser.add_argument('--net_recon', type=str, default='resnet50', choices=['resnet18', 'resnet34', 'resnet50'],
                        help='useless')
    parser.add_argument('--init_path', type=str, default=None, help='Useless')
    parser.add_argument('--use_last_fc', default=False, help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str, default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str, default='BFM_model_front.mat', help='bfm model')

    # default renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)

    args = parser.parse_args()

    if torch.cuda.is_available() and not args.cpu:
        args.device = "cuda"
    else:
        args.device = "cpu"

    main(args)