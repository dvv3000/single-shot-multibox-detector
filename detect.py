from libs import *
from utils import *
from dataset import *
from model import SSD300


class Detect():
    """Args:
            min_score: gia tri score toi thieu cua cac boxes
            top_k: so boxes toi da tren moi image
            max_overlap: muc overlap toi da cua moi box tren cung class

            locs(tensor): (batch, 8732, 4)
            conf(tensor): (batch, 8732, 21)
            def_boxes(tensor): (8732, 4) default boxes
        
        Returns:
            boxes_per_batch(list of tensors, one tensor for each image): cac boxes thoa man
            labels_per_batch(list of tensors, one tensor for each image): label cua cac box tuong ung
            scores_per_batch(list of tensors, one tensor for each image): score cua cac box tuong ung

    """

    def __init__(self, min_score=0.2, top_k=200, max_overlap=0.45):
        self.softmax = nn.Softmax(dim=-1)
        self.min_score = min_score
        self.top_k = top_k
        self.max_overlap = max_overlap

    def forward(self, locs, confs, def_boxes):
        batch_size = confs.size(0)
        num_bbox = confs.size(1)
        num_class = confs.size(2)

        confs = self.softmax(confs) #(batch_size, 8732, num_class)
        # print('confs = ', confs.shape)
        confs_pred = confs.permute(0, 2, 1).contiguous() #(batch_size, num_class, 8732)
        # print(confs_pred.shape)

        # output = torch.zeros(batch_size, num_class, self.top_k, 5)
        boxes_per_batch = list()
        labels_per_batch = list()
        scores_per_batch = list()

        # Xuwr lys tuwngf anhr
        for i in range(batch_size):
            decode_boxes = decode(locs[i], def_boxes) # (8732, 4)
            decode_boxes.clamp_(min=0, max=1)

            confs_score = confs_pred[i].clone().detach() # (num_class, 8732)
            
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            for cl in range(1, num_class):  # Bỏ background
                c_mask = confs_score[cl].gt(self.min_score) # laays nhuwngx thawngf lonws hown min_score #(8732)
                class_scores = confs_score[cl][c_mask] #(x,)
                # print(c_mask.shape)
                if class_scores.size(0) == 0:
                    continue
                
                l_mask = c_mask.unsqueeze(1).expand_as(decode_boxes) # (8732, 4)
                # print(l_mask.shape)
                class_boxes = decode_boxes[l_mask].view(-1, 4) # (x, 4)
  
                indx = torchvision.ops.nms(class_boxes, class_scores, iou_threshold=self.max_overlap)
                

                image_boxes.append(class_boxes[indx] * 300)
                image_scores.append(class_scores[indx])
                image_labels.append(torch.LongTensor([cl] * len(indx)).to(device))
            


            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0, 0, 1, 1]]))
                image_labels.append(torch.LongTensor([0]))
                image_scores.append(torch.FloatTensor([0]))

            image_boxes = torch.cat(image_boxes, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            image_scores = torch.cat(image_scores, dim=0)

            if image_scores.size(0) > self.top_k:
                image_scores, indx = torch.sort(image_scores, dim=0, descending=True)

                image_boxes = image_boxes[indx][:self.top_k] 
                image_labels = image_labels[indx][:self.top_k]
                image_scores = image_scores[:self.top_k]

            # print(image_scores, image_labels)
            # print(image_boxes)

        boxes_per_batch.append(image_boxes)
        labels_per_batch.append(image_labels)
        scores_per_batch.append(image_scores)


        return boxes_per_batch, labels_per_batch, scores_per_batch



def show_pred(model, image):

    """Show predictions from model to image
        Args:
            model: change to GPU first
            image(tensor): (1, 3, 300, 300)
    """
    model.eval()
    
    locs, confs, def_boxes = model(image)
    img = image.squeeze(0)
    img = img.permute(1, 2, 0).contiguous().cpu().numpy()
    det = Detect(min_score=0.3, max_overlap=0.3)
    boxes_batch, labels_batch, scores_batch = det.forward(locs, confs, def_boxes) #list of tensor

    for item in range(len(boxes_batch)):
        scores = scores_batch[item]
        boxes = boxes_batch[item]
        labels = labels_batch[item]
        for i in range(scores.size(0)):
            box = boxes[i]
            start = (int(box[0]), int(box[1]))
            end = (int(box[2]), int(box[3]))
            text = "%s"%(classes[labels[i]-1][:3])
            # print(start, end, classes[labels[i] - 1], scores[i])
            img = cv2.rectangle(img, start, end, (255, 0, 0), 1)
            img = cv2.putText(img, text, start, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        
        plt.imshow(img)
        plt.show()

def to_txt_file(model, testset):
    """Write all annotions of boxes from model and data to file .txt
    """
    model.eval()

    for image, targets, difficulties, filenames in iter(testset):
        image = image.unsqueeze(0).to(device)
        targets = targets.to(device)

        locs, confs, def_boxes = model(image)

        det = Detect(min_score=0.2, top_k=200, max_overlap=0.5)
        boxes, labels, scores = det.forward(locs, confs, def_boxes)
        
        ground_truths_path =  'metrics//Object-Detection-Metrics//groundtruths//' + filenames[0][:-3] +'txt'
        detections_path =  'metrics//Object-Detection-Metrics//detections//' + filenames[0][:-3] +'txt'


        # write ground truths folder
        with open(ground_truths_path, 'w+') as f:
            for i in range(targets.size(0)):
                target = targets[i]
                box = target[:4] * 300
                box = box.int()
                content = '{} {} {} {} {}\n'.format(classes[target[4].int()], box[0] , box[1], box[2], box[3])
                f.write(content)

        
        with open(detections_path, 'w+') as f:
            for i in range(len(boxes[0])):
                label = labels[0][i].int()
                score = scores[0][i]
                box = boxes[0][i].int()
                if label == 0:
                    continue

                content = '{} {} {} {} {} {}\n'.format(classes[label-1], score, box[0], box[1], box[2], box[3])
                f.write(content)
        # show_pred(model, image)

if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    torch.backends.cudnn.benchmark = True

    classes = ["aeroplane", "bicycle", "bird",  "boat", "bottle", 
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]
    
    testset = VOC2007Detection(root='G:/VOC 2007/', classes=classes, transform=transform, image_set='test')
    testloader = DataLoader(dataset=testset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = SSD300(21).to(device)
    weights = torch.load('G:/VOC 2007/weights/ssd300_trainval_200.pth')
    model.load_state_dict(weights)

    image = cv2.imread('data/test.jpg')
    resize = Resize(300)
    totensor = transforms.ToTensor()
    
    image, _ = resize(image)
    image = totensor(image)

    image = image.unsqueeze(0).to(device)
    
    show_pred(model, image)

