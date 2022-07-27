from argparse import ArgumentParser
from dataset import EgoHands
import torch
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead
from torch.optim.lr_scheduler import LambdaLR, LinearLR
from tqdm import tqdm

def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = torch.nn.functional.cross_entropy(x, target)

    if len(losses) == 1:
        return losses["out"]

    return losses["out"] + 0.5 * losses["aux"]


def train_one_epoch(model, optimizer, train_loader, device, lr_scheduler, scaler):
    model.train()
    total_loss = 0
    for image, target in tqdm(train_loader):
        # copy inputs to GPU if available
        image, target = image.to(device), target.to(device)

        # forward pass and loss calculation with mixed precision
        with torch.cuda.amp.autocast():
            output = model(image)
            loss = criterion(output, target)
        
        # update loss
        total_loss += loss

        # reset gradients
        optimizer.zero_grad()

        # scale loss
        loss = scaler.scale(loss)

        # backward pass
        loss.backward()

        # update optimizer and scaler
        scaler.step(optimizer)
        scaler.update()

        # update lr scheduler
        lr_scheduler.step()        

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

def main(args):
    
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
                                batch_size = 8,
                                sampler = train_sampler,
                                num_workers = 8,
                                pin_memory=True
                            )
    
    test_loader = DataLoader(
                                test_dataset,
                                batch_size = 8,
                                sampler = test_sampler,  
                                num_workers = 8,
                                pin_memory=True                              
                            )
    
    # build model with pretrained weights and modify classifier heads
    weights = "ResNet50_Weights.DEFAULT" if args.pretrained else None
    model = deeplabv3_resnet50(weights_backbone = weights, progress=True, aux_loss=False)
    model.classifier = DeepLabHead(in_channels = 2048, num_classes = args.num_classes)
    # model.aux_classifier = FCNHead(in_channels = 1024, channels = 5)       

    # copy model weights to GPU if available
    model.to(device)

    # freeze backbone (only re-train new classifier head)
    if args.pretrained or args.resume:
        for param in model.backbone.parameters():
            param.requires_grad = False

    print("Training these parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-4)

    # learning rate scheduler
    iters_per_epoch = len(train_loader)
    main_lr_scheduler = LambdaLR(optimizer, lambda x: (1 - x / (iters_per_epoch * (args.epochs - args.lr_warmup_epochs))) ** 0.9 )
    if args.lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * args.lr_warmup_epochs      
        warmup_lr_scheduler = LinearLR(optimizer, start_factor=0.33, total_iters=warmup_iters)

        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                                                                optimizer, 
                                                                schedulers=[warmup_lr_scheduler, main_lr_scheduler], 
                                                                milestones=[warmup_iters]
                                                            )
    else:
        lr_scheduler = main_lr_scheduler

    # scale gradients to avoid underflow of small gradients using auto mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # initialize starting epoch
    start_epoch = 0

    if args.resume:
        resume_state_dict = torch.load(args.resume)
        model.load_state_dict(resume_state_dict["model"])
        optimizer.load_state_dict(resume_state_dict["optimizer"])
        lr_scheduler.load_state_dict(resume_state_dict["lr_scheduler"])
        scaler.load_state_dict(resume_state_dict["scaler"])
        start_epoch = resume_state_dict["epoch"] + 1

    total_epochs = args.epochs
    best_loss = float('inf')
    for epoch in range(start_epoch, total_epochs):

        print(epoch,'/',total_epochs)

        train_loss = train_one_epoch(model, optimizer, train_loader, device, lr_scheduler, scaler)
        print(f"Training loss at end of Epoch {epoch}: {train_loss.item():.4f}")

        test_loss = evaluate(model, test_loader, device)
        print(f"Test loss at end of Epoch {epoch}: {test_loss.item():.4f}")

        checkpoint = {
                        "model" : model.state_dict(),
                        "optimizer" : optimizer.state_dict(),
                        "lr_scheduler" : lr_scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch" : epoch
                    }
        
        torch.save(checkpoint, "./weights/last.pth")

        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(checkpoint, "./weights/best.pth")
    
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--lr-warmup-epochs", type=int, default=0)
    parser.add_argument("--num-classes", type=int, default=2)

    args = parser.parse_args()

    main(args)

