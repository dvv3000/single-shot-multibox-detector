from cv2 import transform
from libs import *
from augment import SSDAugmentation


class VOC2007Detection(Dataset):
    """Get data from root path
        Args:
            root: root path that include trainval set and test set
            classes: list name of each class
            tranform:
            image_set: must be one of {'train', 'val', 'test'}

        Returns:
            image(tensor): (3, 300, 300) qua tranform de phu hop voi mo hinh ssd300
            targets(tensor): (num_boxes, 5): boundary coordinates, label(chua them background):
            difficulties(tensor of 0,1): (num_boxes, ): 1 is difficult, 0 is not 
            filenames(list of strings): (num_boxes, ): list of name for each image

    """
    def __init__(self, root, classes, transform=None, image_set='trainval', keep_difficult=False):
        self.root = root
        self.classes = classes
        self.transform = transform
        self.image_set = image_set  
        self.keep_difficult = keep_difficult

        self.ids = []

        assert self.image_set in {'train', 'val', 'test', 'trainval'}, 'image_set must be one of (train, val, test, trainval)'
        
        if self.image_set == 'test':
            id_path = os.path.join(root, 'VOCtest_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/', image_set+'.txt')

            self.anno_path = os.path.join(self.root, 'VOCtest_06-Nov-2007/VOCdevkit/VOC2007/Annotations/%s.xml')
            self.img_path = os.path.join(self.root, 'VOCtest_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/%s.jpg')
        else:
            id_path = os.path.join(root, 'VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/', image_set+'.txt')
            
            self.anno_path = os.path.join(self.root, 'VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations/%s.xml')
            self.img_path = os.path.join(self.root, 'VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages/%s.jpg')

        print(id_path)
        
        with open(id_path, 'r') as f:
            for line in f:
                self.ids.append(line.strip())

        
 
    def __getitem__(self, index):
        boxes, labels, difficulties, file_names = self.get_annotation(index)
        image = self.get_image(index)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        
        targets = torch.hstack((torch.tensor(boxes, dtype=float), torch.unsqueeze(torch.tensor(labels, dtype=float), dim=1)))
            
        return image, targets, difficulties, file_names

    def __len__(self):
        if self.image_set in {'train', 'test', 'trainval'}:
            return len(self.ids)
        else:
            return 500


    def get_annotation(self, index):
        # targets = []
        boxes = []
        labels = []
        difficulties = []
        file_names = []
        path = self.anno_path % self.ids[index]
        xml = ET.parse(path).getroot()

        for item in xml.iter('filename'):
            filename = item.text
            file_names.append(filename)
        
        for item in xml.iter('size'):
            width = int(item.find('width').text)
            height = int(item.find('height').text)
        
        for obj in xml.iter('object'):
            difficult = int(obj.find('difficult').text)
            # difficulties.append(difficult)
            

            if self.keep_difficult == False:
                if difficult == 1:
                    continue

         
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
        
            bndbox = []
            points = ['xmin', 'ymin', 'xmax', 'ymax']

            for item in points:
                pixel = int(bbox.find(item).text) - 1


                bndbox.append(pixel)

            label_id = self.classes.index(name) # without background
            labels.append(label_id)  #(xmin, ymin, xmax, ymax, label)

            boxes.append(bndbox)

        # targets = torch.tensor(targets, dtype=float)
        # boxes = torch.tensor(boxes, dtype=float)
        # labels = torch.tensor(labels, dtype=float)
        difficulties = torch.tensor(difficulties, dtype=torch.uint8)
        boxes = np.array(boxes, dtype='float32')
        labels = np.array(labels)
        return boxes, labels, difficulties, file_names
    
    def get_image(self, index):
        path = self.img_path % self.ids[index]
        image = cv2.imread(path) #(BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #RBG
        # image = Image.open(path, mode='r')
        # image = image.convert('RGB')
        
        # image = torch.from_numpy(image).permute(2, 0, 1)

        return image

def collate_fn(batch):
    image = []
    targets = []
    difficulties = []
    file_names = []

    for item in batch:
        image.append(item[0])
        targets.append(item[1])
        difficulties.append(item[2])
        file_names.append(item[3])
       
    image = torch.stack(image, dim=0) #(batch_size, 3, 300, 300)

    return image, targets, difficulties, file_names




# Transforms

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, boxes=None, labels=None):
        for t in self.transforms:
            image, boxes, labels = t(image, boxes, labels)

        return image, boxes, labels


class NormalizeCoords(object):
    """Normaize Boundary coordinates to [0, 1]"""
    def __call__(self, image, boxes=None, labels=None):
        height, weight, channel = image.shape
        if boxes is not None:
            boxes[:, 0] /= weight
            boxes[:, 1] /= height
            boxes[:, 2] /= weight
            boxes[:, 3] /= height
        
        return image, boxes, labels

class Resize(object):
    """Resize each image to 'size' by black two-side padding with non square image"""
    
    def __init__(self, size):
        self.size = size
    def __call__(self, image, boxes=None, labels=None):
        old_h, old_w, channel = image.shape

        padd_top = max((old_w - old_h) // 2, 0)
        padd_left = max((old_h - old_w) // 2, 0)

        image = cv2.copyMakeBorder(image, padd_top, padd_top, padd_left, padd_left, cv2.BORDER_CONSTANT, (0, 0, 0))

        image = cv2.resize(image, (self.size, self.size))
        # image = torch.from_numpy(image).permute(2, 0, 1)
        if boxes is not None:
            boxes[:, 0] = (boxes[:, 0] + padd_left) * self.size / max(old_h, old_w)
            boxes[:, 1] = (boxes[:, 1] + padd_top) * self.size / max(old_h, old_w)
            boxes[:, 2] = (boxes[:, 2] + padd_left) * self.size / max(old_h, old_w)
            boxes[:, 3] = (boxes[:, 3] + padd_top) * self.size / max(old_h, old_w)
        
        return image, boxes, labels

class ToTensor(object):
    def __call__(self, image, boxes=None, labels=None):
        to_tensor = transforms.ToTensor()
        return to_tensor(image), boxes, labels

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, image, boxes=None, labels=None):
        normalize = transforms.Normalize(mean = self.mean, std = self.std)

        return normalize(image), boxes, labels

if __name__ == "__main__":
    classes = ["aeroplane", "bicycle", "bird",  "boat", "bottle", 
               "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant",
               "sheep", "sofa", "train", "tvmonitor"]

    transform = Compose([Resize(300), NormalizeCoords(), ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
    
    dataset = VOC2007Detection('G:/VOC 2007/', classes, transform=SSDAugmentation())
    print('Length of dataset', len(dataset))
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    for (images, targets, _, _)  in iter(dataloader):
    # print(filenames[0])
        print(images.mean())
        image = images[0].permute(1, 2, 0).contiguous().numpy()
        target = targets[0]
        boxes = targets[0][:, :4] * 300
        labels = targets[0][:, 4].int()
        print(boxes, labels)
        for i in range(boxes.size(0)):  
            start = (int(boxes[i, 0]), int(boxes[i, 1]))
            end = (int(boxes[i, 2]), int(boxes[i, 3]))
            image = cv2.rectangle(image, start, end, (255, 0, 0), 1)
            label = classes[labels[i]]
            image = cv2.putText(image, label, start, cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
        plt.imshow(image)
        plt.show()

        break