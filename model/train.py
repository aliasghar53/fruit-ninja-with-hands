import time
from dataset import EgoHands
import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = torch.nn.functional.cross_entropy(x, target)

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]


def train_one_epoch(model, optimizer, train_loader, device, scaler):
    model.train()
    total_loss = 0
    for image, target in tqdm(train_loader):
        # copy inputs to GPU if available
        image, target = image.to(device), target.to(device)

        # forward pass and loss calculation with mixed precision
        with torch.cuda.amp.autocast():
            output = model(image)
            loss = criterion(output, target)
        
        # reset gradients
        optimizer.zero_grad()

        # scale loss
        loss = scaler.scale(loss)

        # backward pass
        loss.backward()

        # update optimizer and scaler
        scaler.step(optimizer)
        scaler.update()

        # update loss
        total_loss += loss

    return total_loss / len(train_loader)

def evaluate(model, test_loader, device):
    model.eval()
    total_loss = 0
    with torch.inference_mode():
        for image, target in tqdm(test_loader):
            
            image, target = image.to(device), target.to(device)

            output = model(image)

            loss = criterion(output, target)

            total_loss += loss

    return total_loss / len(test_loader)

def main():
    
    # use cuda if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create datasets
    train_dataset = EgoHands(mode="train")
    test_dataset = EgoHands(mode="eval")

    # create samplers and split dataset based on index
    train_sampler = SubsetRandomSampler(list(range(4000)))
    test_sampler =  SubsetRandomSampler(list(range(4000,4800)))

    # create dataloaders
    train_loader = DataLoader(
                                train_dataset,
                                batch_size = 2,
                                sampler = train_sampler,
                                num_workers = 2
                            )
    
    test_loader = DataLoader(
                                test_dataset,
                                batch_size = 2,
                                sampler = test_sampler,  
                                num_workers = 2                              
                            )
    
    # build model with pretrained weights and modify classifier heads
    model = deeplabv3_resnet50(weights = "DeepLabV3_ResNet50_Weights.DEFAULT", progress=True, aux_loss=True)
    model.classifier = DeepLabHead(in_channels = 2048, num_classes = 5)
    model.aux_classifier = FCNHead(in_channels = 1024, channels = 5)

    # copy model weights to GPU if available
    model.to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    # learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2)

    # scale gradients to avoid underflow of small gradients using auto mixed precision
    scaler = torch.cuda.amp.GradScaler()

    
    total_epochs = 30
    best_loss = float('inf')
    for epoch in range(total_epochs):

        start_time = time.time()

        print(epoch,'/',total_epochs)

        train_loss = train_one_epoch(model, optimizer, train_loader, device, scaler)
        print(f"Training loss at end of Epoch {epoch}: {train_loss:.6g}")
        print(f"Runtime = {time.time() - start_time:.4f}s")

        test_loss = evaluate(model, test_loader, device)
        print(f"Test loss at end of Epoch {epoch}: {test_loss:.4f}")

        scheduler.step(test_loss)

        checkpoint = {
                        "model" : model.state_dict(),
                        "optimizer" : optimizer.state_dict(),
                        "scheduler" : scheduler.state_dict(),
                        "epoch" : epoch
                    }
        
        torch.save(checkpoint, "./weights/last.pth")

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(checkpoint, "./weights/best.pth")

if __name__ == "__main__":
    main()

