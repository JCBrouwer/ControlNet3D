import json
from typing import Tuple

import datasets
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


class BLIPedDataset(Dataset):
    def __init__(self, json_file: str, length: int = 24) -> None:
        super().__init__()

        self.length = length

        with open(json_file) as f:
            clips = json.load(f)["data"]

        self.data = []
        for clip in clips:
            for subclip in clip["data"]:
                self.data.append(
                    {
                        "video_file": clip["video_file"],
                        "start_idx": subclip["frame_index"],
                        "caption": subclip["prompt"],
                    }
                )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        video_file = self.data[index]["video_file"]
        start_idx = self.data[index]["start_idx"]
        caption = self.data[index]["caption"]

        video, _, _ = read_video(video_file, pts_unit="sec", output_format="TCHW")

        return video[start_idx : start_idx + self.length + 1].permute(1, 0, 2, 3), caption


class FlowPreprocessor(torch.nn.Module):
    out_channels: int = 2

    def __init__(self) -> None:
        super().__init__()
        self.raft_transform = Raft_Large_Weights.DEFAULT.transforms()
        self.raft = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=True).eval()

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        prev, curr = videos[:, :-1], videos[:, 1:]
        optical_flows = self.raft.forward(*self.raft_transform(prev, curr))[-1]
        return optical_flows


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-json", type=str)
    parser.add_argument("--resolution", type=Tuple[int, int], default=(640, 360))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    dataset = BLIPedDataset(args.data_json)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        shuffle=True,
    )

    if args.conditioning == "flow":
        conditioning_preprocessor = FlowPreprocessor().to(args.device)
    else:
        raise NotImplementedError()

    class ControlNetDataset(datasets.GeneratorBasedBuilder):
        def _info(self):
            return datasets.DatasetInfo(
                features=datasets.Features(
                    {
                        "video": datasets.Array4D(shape=(args.length, 3, args.resolution[1], args.resolution[0])),
                        "conditioning_video": datasets.Array4D(
                            shape=(
                                args.length,
                                conditioning_preprocessor.out_channels,
                                args.resolution[1],
                                args.resolution[0],
                            )
                        ),
                        "text": datasets.Value("string"),
                    }
                )
            )

        def _split_generators(self, dl_manager):
            return [datasets.SplitGenerator(name=datasets.Split.TRAIN)]

        def _generate_examples(self):
            for videos, captions in dataloader:
                conditionings = conditioning_preprocessor(videos)
                for video, conditioning, caption in zip(videos, conditionings, captions):
                    yield id, {"video": video, "conditioning_video": conditioning, "text": caption}
