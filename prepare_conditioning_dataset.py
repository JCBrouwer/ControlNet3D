import argparse
import os
from glob import glob

import torch
from einops import rearrange
from kornia.filters import gaussian_blur2d, laplacian
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video, write_video
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from tqdm import tqdm


class FlowPreprocessor(nn.Module):
    out_channels: int = 3

    def __init__(self, smooth: float = 0.05, emphasis: float = 6.0) -> None:
        super().__init__()
        self.transform = Raft_Large_Weights.DEFAULT.transforms()
        self.raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True).eval()
        self.smooth = smooth
        self.emphasis = emphasis

    @torch.no_grad()
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        # prepare and run RAFT
        prev, curr = video[:-1], video[1:]
        prev, curr = self.transform(prev, curr)
        flow = self.raft.forward(prev, curr)[-1]

        # smooth flow
        sigma = min(flow.shape[3], flow.shape[2]) * self.smooth
        flow = gaussian_blur2d(flow, kernel_size=5, sigma=(sigma, sigma))

        # prepend zero flow to match video length
        flow = torch.cat((torch.zeros_like(flow[[0]]), flow))

        # normalize flow by video size
        flow[:, 0] /= video.shape[3]
        flow[:, 1] /= video.shape[2]

        # calculate flow laplacian magnitude (rough edge detection)
        flow_laplacian = laplacian(flow, kernel_size=5)
        magnitude = flow_laplacian.abs().sum(1, keepdim=True).pow(1 / self.emphasis).mul(2).sub(1)

        # concatenate to 3-channel conditioning
        conditioning = torch.cat((flow, magnitude), dim=1)

        # emphasize small values
        conditioning = torch.sign(conditioning) * conditioning.abs().pow(1 / self.emphasis)

        # re-scale to [0, 1]
        conditioning = conditioning.add(1).div(2)

        return conditioning


class VideoFolder(Dataset):
    def __init__(self, input_dir: str):
        self.files = sorted(filter(lambda x: not ".conditioning" in x, glob(os.path.join(input_dir, "*.mp4"))))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        video, _, info = read_video(file, pts_unit="sec", output_format="TCHW")
        return file, video.div(255), info["video_fps"]


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

    for (file,), video, fps in tqdm(dataloader, desc="Processing videos..."):
        video = video.squeeze().to(device)

        cond = conditioner(video)

        write_video(
            filename=file.replace(".mp4", ".conditioning.mp4"),
            video_array=rearrange(cond.mul(255).round().byte(), "f c h w -> f h w c").cpu(),
            fps=fps.squeeze().item(),
            options={"crf": "18"},
        )


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
