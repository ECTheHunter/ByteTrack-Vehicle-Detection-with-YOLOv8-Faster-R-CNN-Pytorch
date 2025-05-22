from pycocotools.coco import COCO
import torch
from PIL import Image
import os
from torchvision.transforms import Compose, ToTensor, Resize
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader

# Dataset Class
class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        if transforms is None:
            transforms = Compose([
                Resize((320, 320)),
                ToTensor()
            ])
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms
        self.classes = ['bus', 'car', 'truck', 'van']

        cats = self.coco.loadCats(self.coco.getCatIds())
        name_to_coco_id = {cat['name']: cat['id'] for cat in cats}
        self.class_map = {}
        for i, cls_name in enumerate(self.classes, 1):
            if cls_name in name_to_coco_id:
                self.class_map[name_to_coco_id[cls_name]] = i

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = Image.open(img_path).convert("RGB")

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes, labels = [], []

        for ann in anns:
            coco_cat_id = ann['category_id']
            if coco_cat_id not in self.class_map:
                continue
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(self.class_map[coco_cat_id])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.tensor([])
            iscrowd = torch.tensor([])
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.as_tensor(
                [ann.get('iscrowd', 0) for ann in anns if ann['category_id'] in self.class_map],
                dtype=torch.int64
            )

        image_id = torch.tensor([img_id])
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        if self.transforms:
            img = self.transforms(img)
        return img, target

    def __len__(self):
        return len(self.image_ids)

def collate_fn(batch):
    return tuple(zip(*batch))

# Validation function
def validate(model, data_loader, device):
    model.train()  # Keep model in train mode to get loss dict
    total_loss = 0
    with torch.no_grad():  # disable gradients to save memory and prevent updates
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)  # returns dict in train mode
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()
    avg_loss = total_loss / len(data_loader)
    print(f"[VAL] Validation Loss: {avg_loss:.4f}")


# Main training function
def main():
    # Paths (adjust as needed)
    train_dir = "UA-DETRAC-COCO/train"
    train_ann = "UA-DETRAC-COCO/train/_annotations.coco.json"
    val_dir = "UA-DETRAC-COCO/valid"
    val_ann = "UA-DETRAC-COCO/valid/_annotations.coco.json"

    # Load datasets
    dataset = CocoDataset(train_dir, train_ann, transforms=ToTensor())
    val_dataset = CocoDataset(val_dir, val_ann, transforms=ToTensor())

    data_loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # Model setup
    num_classes = 5  # 4 classes + background
    print("Loading model...")
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    print("Model loaded!")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    num_epochs = 50
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()

        lr_scheduler.step()
        print(f"[TRAIN] Epoch Loss: {epoch_loss / len(data_loader):.4f}")

        # Validation step
        validate(model, val_loader, device)

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
    }, 'fasterrcnn_checkpoint.pth')
    print("Training complete and model saved!")

if __name__ == "__main__":
    main()
