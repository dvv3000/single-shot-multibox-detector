from libs import *
from dataset import *
from model import SSD300
from lossfunction import MultiboxLoss
from augment import SSDAugmentation


def train(model, trainloader, criterion, optimizer, epochs, valloader=None):
    """Args:
            model: have to change to GPU first
            trainloader
            valloader
            criterions: Loss function
            optimizer
            epochs: number epochs u wanna train
    """
    print_freq = 100

    for epoch in range(epochs):
        epoch_loss = 0.0
        val_epoch_loss = 0.0
        
        epoch_start = time.time()
        iter_start = time.time()

        print("---" * 20)
        print('Epoch {} / {}:'.format(epoch+1, epochs))

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = trainloader
            else:
                model.eval()
                dataloader = valloader
                continue

            for i, (images, targets, _, _) in enumerate(dataloader):
                iter_start = time.time()
                
                images = images.to(device)
                targets = [t.to(device) for t in targets]

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                    if phase == 'train':
                        loss.backward()
                        nn.utils.clip_grad_value_(model.parameters(), clip_value=2.0)
                        optimizer.step()

                        iter_finish = time.time()
                        if (i + 1) % print_freq == 0:
                            time_per_iter = iter_finish - iter_start
                            print('\t\t Iter: {} time: {:.2f} s, loss: {:.4f}'.format(i + 1, time_per_iter, loss.item()))

                        epoch_loss += loss.item()

                    else:
                        val_epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(trainloader)

        if valloader is not None:
            val_epoch_loss = val_epoch_loss / len(valloader)



        epoch_finish = time.time()
        time_per_epoch = epoch_finish - epoch_start
        print('\t Time: {:.2f} s, train_loss: {:.4f}, val_loss: {:.4f}'.format(time_per_epoch, epoch_loss, val_epoch_loss))

        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), './weights/ssd300_' + str(epoch + 1) + '.pth') 
            print("Save weights on \'./weights/\'.")


if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    torch.backends.cudnn.benchmark = True

    classes = ["aeroplane", "bicycle", "bird",  "boat", "bottle", 
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

    transform = Compose([Resize(300), NormalizeCoords(), ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    
    trainset = VOC2007Detection('G:/VOC 2007/', classes, transform=SSDAugmentation())
    print('Length of trainset:', len(trainset))
    trainloader = DataLoader(trainset, batch_size=8, shuffle=True, collate_fn=collate_fn)


    model = SSD300(21).to(device)

    # weights = torch.load('G:/VOC 2007/weights/ssd300_trainval_200.pth')
    # model.load_state_dict(weights)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print('Trainable parameters =', trainable_params)
    print('Total parameters =', total_params)


    criterion = MultiboxLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)

    train(model, trainloader, criterion, optimizer, 100)

