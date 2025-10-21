from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def list_image_files(root: Path) -> List[Path]:
    root = root.expanduser().resolve()
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS])


def default_transform() -> Callable[[Image.Image], torch.Tensor]:
    return T.Compose([T.ToTensor()])


@dataclass
class Sample:
    lq: torch.Tensor
    gt: Optional[torch.Tensor]
    lq_path: Path
    rel_path: Path


class PairedImageDataset(Dataset):
    def __init__(
        self,
        input_root: Path | str,
        target_root: Path | str,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.input_root = Path(input_root).expanduser().resolve()
        self.target_root = Path(target_root).expanduser().resolve()
        self.transform = transform or default_transform()

        self.samples: List[Tuple[Path, Path, Path]] = []
        input_files = list_image_files(self.input_root)
        target_map = {p.relative_to(self.target_root): p for p in list_image_files(self.target_root)}

        for lq_path in input_files:
            rel = lq_path.relative_to(self.input_root)
            target_path = target_map.get(rel)
            if target_path is None:
                continue
            self.samples.append((lq_path, target_path, rel))

        if not self.samples:
            raise RuntimeError(f"No paired samples found in {self.input_root} and {self.target_root}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        lq_path, gt_path, rel_path = self.samples[index]
        lq_image = Image.open(lq_path).convert("RGB")
        gt_image = Image.open(gt_path).convert("RGB")
        lq = self.transform(lq_image)
        gt = self.transform(gt_image)
        return {
            "LQ": lq,
            "GT": gt,
            "LQ_path": str(lq_path),
            "GT_path": str(gt_path),
            "rel_path": str(rel_path),
        }


class UnpairedImageDataset(Dataset):
    def __init__(
        self,
        input_root: Path | str,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.input_root = Path(input_root).expanduser().resolve()
        self.transform = transform or default_transform()
        self.samples = list_image_files(self.input_root)
        if not self.samples:
            raise RuntimeError(f"No images found in {self.input_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        path = self.samples[index]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        rel_path = path.relative_to(self.input_root)
        return {
            "LQ": tensor,
            "LQ_path": str(path),
            "rel_path": str(rel_path),
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def create_paired_train_loader(
    train_root: Path | str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    train_root = Path(train_root)
    dataset = PairedImageDataset(train_root / "input", train_root / "target")
    return create_dataloader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def create_paired_eval_loader(
    eval_root: Path | str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    eval_root = Path(eval_root)
    dataset = PairedImageDataset(eval_root / "input", eval_root / "target")
    return create_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


def create_unpaired_eval_loader(
    eval_root: Path | str,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = UnpairedImageDataset(eval_root)
    return create_dataloader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
