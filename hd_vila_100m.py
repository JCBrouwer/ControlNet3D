from typing import Optional

import jsonlines
import torch
from numpy.random import RandomState
from yt_dlp import YoutubeDL

FORMATS = {144: "160", 240: "133", 360: "134", 480: "135", 720: "136"}


class HD_VILA_100m(torch.utils.data.Dataset):
    def __init__(self, length: int = 24, height: int = 360, seed: int = 42, limit: Optional[int] = None, part: int = 0):
        self.length = length
        self.height = height
        self.format = FORMATS[height]

        print("Reading clips...")
        self.clips = self.read_clips(f"./hd-vila-100m/hdvila_part{part}.jsonl", limit=limit)
        print("Shuffling clips...")
        RandomState(seed).shuffle(self.clips)

    def read_clips(self, metafile: str, limit: Optional[int] = None):
        vs = []
        num = 0
        with open(metafile, "r") as file:
            for line in jsonlines.Reader(file):
                for clip in line["clip"]:
                    hour, minute, second = clip["span"][0].split(":")
                    start_time = float(hour) * 3600 + float(minute) * 60 + float(second)
                    hour, minute, second = clip["span"][1].split(":")
                    end_time = float(hour) * 3600 + float(minute) * 60 + float(second)
                    vs.append((line["url"], start_time, end_time))
                    num += 1
                    if limit is not None and num >= limit:
                        return vs
        return vs

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        try:
            filename = None

            def get_filename(d):
                nonlocal filename
                filename = d.get("info_dict").get("_filename")

            url, start_time, end_time = self.clips[idx]
            with YoutubeDL(
                {
                    "outtmpl": "./hd-vila-100m/clips/%(id)s.%(ext)s",
                    "format": self.format,
                    "skip_download": False,
                    "quiet": True,
                    "download_ranges": lambda _, __: [{"start_time": start_time, "end_time": end_time}],
                    "progress_hooks": [get_filename],
                }
            ) as ydl:
                ydl.download([url])

            return self.clips[idx]

        except Exception:
            return self.__getitem__((idx * 123456789) % len(self.clips))


if __name__ == "__main__":
    dataset = HD_VILA_100m()
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=24, prefetch_factor=10**10
    )
    dataiter = iter(dataloader)

    from tqdm import trange

    clips = []
    with jsonlines.open("./hd-vila-100m/clips/my_subset.jsonl", "w") as writer:
        for i in trange(100_000, smoothing=0):
            url, start, stop = next(dataiter)
            writer.write({"url": url[0], "start": start.item(), "stop": stop.item()})
