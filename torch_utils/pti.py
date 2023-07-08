import pickle
from PTI.utils.ImagesDataset import ImagesDataset, Image2Dataset
import torch
from PTI.utils.models_utils import load_old_G
from PTI.utils.alignment import align_face

from PTI.training.coaches.single_id_coach import SingleIDCoach
from PTI.configs import global_config, paths_config
import dlib

import os
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from string import ascii_uppercase
import sys
from pathlib import Path

sys.path.append(".")
# sys.path.append('PTI/')
# sys.path.append('PTI/training/')


def run_PTI(img, run_name):
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    global_config.run_name = run_name

    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    embedding_dir_path = f"{paths_config.embedding_base_dir}/{paths_config.input_data_id}/{paths_config.pti_results_keyword}"
    os.makedirs(embedding_dir_path, exist_ok=True)

    # dataset = ImagesDataset(paths_config.input_data_path, transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

    G = load_old_G()
    IMAGE_SIZE = 1024
    predictor = dlib.shape_predictor(paths_config.dlib)
    aligned_image = align_face(img, predictor=predictor, output_size=IMAGE_SIZE)
    img = aligned_image.resize([G.img_resolution, G.img_resolution])
    dataset = Image2Dataset(img)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    coach = SingleIDCoach(dataloader, use_wandb=False)

    new_G, w_pivot = coach.train()
    return new_G, w_pivot


def export_updated_pickle(new_G, out_path, run_name):
    image_name = "customIMG"

    with open(paths_config.stylegan2_ada_ffhq, "rb") as f:
        old_G = pickle.load(f)["G_ema"].cuda()

    embedding = Path(f"{paths_config.checkpoints_dir}/model_{run_name}_{image_name}.pt")
    with open(embedding, "rb") as f_new:
        new_G = torch.load(f_new).cuda()

    print("Exporting large updated pickle based off new generator and ffhq.pkl")
    with open(paths_config.stylegan2_ada_ffhq, "rb") as f:
        d = pickle.load(f)
        old_G = d["G_ema"].cuda()  # tensor
        old_D = d["D"].eval().requires_grad_(False).cpu()

    tmp = {}
    tmp["G"] = old_G.eval().requires_grad_(False).cpu()
    tmp["G_ema"] = new_G.eval().requires_grad_(False).cpu()
    tmp["D"] = old_D
    tmp["training_set_kwargs"] = None
    tmp["augment_pipe"] = None

    with open(out_path, "wb") as f:
        pickle.dump(tmp, f)
    # delete

    embedding.unlink()


# if __name__ == '__main__':
#     from PIL import Image
#     img = Image.open('PTI/test/test.jpg')
#     new_G, w_pivot = run_PTI(img, use_wandb=False, use_multi_id_training=False)
#     out_path = f'checkpoints/stylegan2_custom_512_pytorch.pkl'
#     export_updated_pickle(new_G, out_path)
