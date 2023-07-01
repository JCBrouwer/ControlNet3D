import argparse
import os
from glob import glob

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from tqdm import tqdm


class FlowPreprocessor(nn.Module):
    out_channels: int = 2

    def __init__(self) -> None:
        super().__init__()
        self.raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True).eval()

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        video = video.mul(2).sub(1)
        return self.raft.forward(video[:-1], video[1:])[-1]


class VideoFolder(Dataset):
    def __init__(self, input_dir: str):
        self.files = glob(os.path.join(input_dir, "*.mp4"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        video, _, _ = read_video(file, pts_unit="sec", output_format="TCHW")
        return file, video.div(255)


@torch.inference_mode()
def process_video_conditioning(
    input_dir: str, conditioning: str, num_workers: int = 8, prefetch_factor: int = 4, device: str = "cuda"
):
    """
    Preprocesses videos to conditioning for training.

    Args:
        input_dir (str): Path to the directory with videos.
        conditioning (str): Type of conditioning to use. Currently only "flow" is supported.
        num_workers (int, optional): Number of workers for dataloader. Defaults to 8.
        prefetch_factor (int, optional): Prefetch factor for the dataloader. Defaults to 4.
        device (str, optional): Device to use. Defaults to "cuda".
    """

    if conditioning == "flow":
        conditioner = FlowPreprocessor().to(device)
    else:
        raise NotImplementedError()

    dataset = VideoFolder(input_dir)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers, prefetch_factor=prefetch_factor)

    for (file,), video in tqdm(dataloader, desc="Processing videos..."):
        video = video.squeeze().to(device)
        cond = conditioner(video)
        torch.save(cond.cpu().half(), file.replace(".mp4", ".pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_dir", type=str, help="Path to the directory with videos.")
    parser.add_argument(
        "--conditioning", type=str, default="flow", choices=["flow"], help="Type of conditioning to use."
    )
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for the dataloader.")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="Prefetch factor for the dataloader.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
    args = parser.parse_args()

    process_video_conditioning(**vars(args))
