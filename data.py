import os
import json
import random

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.datasets.voc import VOCDetection
from torch.utils.data import Dataset

import config


class YoloPascalVocDataset(Dataset):
    def __init__(self, image_set, normalize=False, augment=False):
        self.prepare_workspace()
        transform=T.Compose([T.ToTensor(), T.Resize(config.IMAGE_SIZE)])
        self.dataset = VOCDetection(
            root=config.DATA_PATH,
            year="2007",
            image_set=image_set,
            download=False,
            transform=transform
        )
        self.normalize = normalize
        self.augment = augment
        self.labels = self.load_labels()

    def __getitem__(self, i):
        data, label = self.dataset[i]
        original_data = data
        x_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[0])
        y_shift = int((0.2 * random.random() - 0.1) * config.IMAGE_SIZE[1])
        scale = 1 + 0.2 * random.random()

        # Augment images
        if self.augment:
            data = TF.affine(data, angle=0.0, scale=scale, translate=(x_shift, y_shift), shear=0.0)
            data = TF.adjust_hue(data, 0.2 * random.random() - 0.1)
            data = TF.adjust_saturation(data, 0.2 * random.random() + 0.9)
        # Normalize with ImageNet mean and std
        if self.normalize:
            data = TF.normalize(data, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        grid_size_x = data.size(dim=2) / config.S  # Images in PyTorch have size (channels, height, width)
        grid_size_y = data.size(dim=1) / config.S

        # Process bounding boxes into the SxSx(5*B+C) ground truth tensor
        boxes = {}
        class_names = {}                    # Track what class each grid cell has been assigned to
        depth = 5 * config.B + config.C     # 5 numbers per bbox, then one-hot encoding of label
        ground_truth = torch.zeros((config.S, config.S, depth))
        for j, bbox_pair in enumerate(utils.get_bounding_boxes(label)):
            name, coords = bbox_pair
            assert name in self.classes, f"Unrecognized class '{name}'"
            class_index = self.classes[name]
            x_min, x_max, y_min, y_max = coords

            # Augment labels
            if self.augment:
                half_width = config.IMAGE_SIZE[0] / 2
                half_height = config.IMAGE_SIZE[1] / 2
                x_min = utils.scale_bbox_coord(x_min, half_width, scale) + x_shift
                x_max = utils.scale_bbox_coord(x_max, half_width, scale) + x_shift
                y_min = utils.scale_bbox_coord(y_min, half_height, scale) + y_shift
                y_max = utils.scale_bbox_coord(y_max, half_height, scale) + y_shift

            # Calculate the position of center of bounding box
            mid_x = (x_max + x_min) / 2
            mid_y = (y_max + y_min) / 2
            col = int(mid_x // grid_size_x)
            row = int(mid_y // grid_size_y)

            if 0 <= col < config.S and 0 <= row < config.S:
                cell = (row, col)
                if cell not in class_names or name == class_names[cell]:
                    # Insert class one-hot encoding into ground truth
                    one_hot = torch.zeros(config.C)
                    one_hot[class_index] = 1.0
                    ground_truth[row, col, :config.C] = one_hot
                    class_names[cell] = name

                    # Insert bounding box into ground truth tensor
                    bbox_index = boxes.get(cell, 0)
                    if bbox_index < config.B:
                        bbox_truth = (
                            (mid_x - col * grid_size_x) / config.IMAGE_SIZE[0],     # X coord relative to grid square
                            (mid_y - row * grid_size_y) / config.IMAGE_SIZE[1],     # Y coord relative to grid square
                            (x_max - x_min) / config.IMAGE_SIZE[0],                 # Width
                            (y_max - y_min) / config.IMAGE_SIZE[1],                 # Height
                            1.0                                                     # Confidence
                        )

                        # Fill all bbox slots with current bbox (starting from current bbox slot, avoid overriding prev)
                        # This prevents having "dead" boxes (zeros) at the end, which messes up IOU loss calculations
                        bbox_start = 5 * bbox_index + config.C
                        ground_truth[row, col, bbox_start:] = torch.tensor(bbox_truth).repeat(config.B - bbox_index)
                        boxes[cell] = bbox_index + 1

        return data, ground_truth, original_data

    def __len__(self):
        return len(self.dataset)

    def prepare_workspace(self):
        if not os.path.exists(config.DATA_PATH):
            os.makedirs(config.DATA_PATH)

    def load_labels(self):
        if os.path.exists(config.LABEL_PATH):
            with open(config.LABEL_PATH, "r") as fin:
                return json.load(fin)
        else:
            # Load labels from dataset
            labels = {}
            index = 0
            for _, (_, label) in self.dataset:
                for _, (name, _) in self.get_bbox(label):
                    if name not in labels:
                        labels[name] = index
                        index += 1
            with open(config.LABEL_PATH, "w") as fout:
                json.dump(labels, fout)
            return labels

    def print_labels(self):
        for name in self.labels:
            idx = self.labels[name]
            print(name, idx)

    def get_bbox(self, label):
        size = label["annotation"]["size"]
        width, height = int(size["width"]), int(size["height"])
        x_scale = config.IMAGE_SIZE[0] / width
        y_scale = config.IMAGE_SIZE[1] / height
        bbox = []
        for obj in label["annotation"]["object"]:
            box = obj["bndbox"]
            coords = (
                int(int(box["xmin"]) * x_scale),
                int(int(box["xmax"]) * x_scale),
                int(int(box["ymin"]) * y_scale),
                int(int(box["ymax"]) * y_scale)
            )
            name = obj["name"]
            bbox.append((name, coords))
        return bbox


if __name__ == '__main__':
    dataset = YoloPascalVocDataset("train", normalize=True, augment=True)
    dataset.print_labels()

    #  negative_labels = 0
    #  smallest = 0
    #  largest = 0
    #  for data, label, _ in train_set:
    #      negative_labels += torch.sum(label < 0).item()
    #      smallest = min(smallest, torch.min(data).item())
    #      largest = max(largest, torch.max(data).item())
    #      utils.plot_boxes(data, label, obj_classes, max_overlap=float('inf'))
