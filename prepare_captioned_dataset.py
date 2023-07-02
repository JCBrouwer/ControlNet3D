import argparse
import os
import random
from glob import glob
from typing import List, Tuple
from uuid import uuid4

import torch
import torch.multiprocessing
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video, write_video
from torchvision.transforms.functional import resize, to_pil_image
from tqdm import tqdm
from transformers import Blip2ForConditionalGeneration, Blip2Processor

torch.multiprocessing.set_sharing_strategy("file_system")
torch.multiprocessing.set_start_method("spawn", force=True)

EXTENSIONS = (".mp4", ".avi", ".mov", ".webm", ".flv", ".mjpeg")


class BLIP2Dataset(Dataset):
    def __init__(self, input_dir: str, clip_length: int = 48, height: int = 360, width: int = 640):
        super().__init__()

        self.files = sum([glob(f"{input_dir}/*{ext}") for ext in EXTENSIONS], [])

        self.clip_length = clip_length
        self.height = height
        self.width = width

        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[List[Tensor], List[Tensor]]:
        file = self.files[idx]
        video, _, _ = read_video(file, pts_unit="sec", output_format="TCHW")
        print(video.min(), video.max())
        video = resize(video, (self.height, self.width), antialias=True)
        video_clips = video.split(self.clip_length)[:-1]  # drop last clip which might be shorter than clip_length
        caption_frames = [clip[random.randint(0, self.clip_length - 1)] for clip in video_clips]
        preprocessed_frames = [self.processor(to_pil_image(frame))["pixel_values"][0] for frame in caption_frames]
        video_clips = [clip.permute(0, 2, 3, 1) for clip in video_clips]  # (T, H, W, C)
        return video_clips, preprocessed_frames


def get_dataloader(dataset: Dataset, batch_size: int, num_workers: int = 4, prefetch_factor: int = 4):
    dataloader = DataLoader(
        dataset, batch_size=1, num_workers=num_workers, prefetch_factor=prefetch_factor, shuffle=True
    )
    clip_batch, frame_batch = [], []
    for clips, frames in tqdm(dataloader, desc="Captioning videos..."):
        for clip, frame in zip(clips, frames):
            clip_batch.append(clip)
            frame_batch.append(frame)
            if len(clip_batch) == batch_size:
                yield torch.cat(clip_batch), torch.cat(frame_batch)
                clip_batch, frame_batch = [], []


class BLIP2Captioner(nn.Module):
    def __init__(self, beam_amount: int = 7, min_prompt_length: int = 10, max_prompt_length: int = 70):
        super().__init__()

        self.beam_amount = beam_amount
        self.min_length = min_prompt_length
        self.max_length = max_prompt_length

        print("Loading BLIP2 (this may take a bit)...")
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
        )

    def forward(self, pixel_values: Tensor) -> List[str]:
        generated_ids = self.model.generate(
            pixel_values=pixel_values.half(),
            num_beams=self.beam_amount,
            min_length=self.min_length,
            max_length=self.max_length,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [text.strip() for text in generated_text]


@torch.inference_mode()
def process_video_captions(
    input_dir: str,
    out_dir: str,
    clip_length: int = 48,
    height: int = 360,
    width: int = 640,
    batch_size: int = 16,
    num_workers: int = 4,
    prefetch_factor: int = 4,
    device: str = "cuda",
):
    """Generate captions for videos in input_dir and save them to out_dir.

    Args:
        input_dir (str): Directory containing videos.
        out_dir (str): Directory to save captioned video clips to.
        clip_length (int, optional): Length of video clips to use for captioning. Defaults to 48.
        height (int, optional): Height of video frames. Defaults to 360.
        width (int, optional): Width of video frames. Defaults to 640.
        batch_size (int, optional): Batch size for captioning. Defaults to 16.
        num_workers (int, optional): Number of workers for dataloader. Defaults to 4.
        prefetch_factor (int, optional): Prefetch factor for dataloader. Defaults to 4.
        device (str, optional): Device to use for captioning model. Defaults to "cuda".
    """

    dataset = BLIP2Dataset(input_dir, clip_length, height, width)

    dataloader = get_dataloader(
        dataset, batch_size=batch_size, num_workers=num_workers, prefetch_factor=prefetch_factor
    )

    blip2 = BLIP2Captioner().to(device)

    os.makedirs(out_dir, exist_ok=True)
    for videos, caption_frames in dataloader:
        captions = blip2(caption_frames.to(device))
        for video, caption in zip(videos, captions):
            vid_id = str(uuid4())
            write_video(f"{out_dir}/{vid_id}.mp4", video, fps=24)
            with open(f"{out_dir}/{vid_id}.txt", "w") as f:
                f.write(caption)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("input_dir", type=str, help="Directory containing videos.")
    parser.add_argument("out_dir", type=str, help="Directory to save captioned video clips to.")
    parser.add_argument("--clip_length", type=int, default=48, help="Length of video clips to use for captioning.")
    parser.add_argument("--height", type=int, default=360, help="Height of video frames.")
    parser.add_argument("--width", type=int, default=640, help="Width of video frames.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for captioning.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader.")
    parser.add_argument("--prefetch_factor", type=int, default=4, help="Prefetch factor for dataloader.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for captioning model.")
    args = parser.parse_args()

    process_video_captions(**vars(args))
