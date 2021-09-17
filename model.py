from libs import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class Base(nn.Module):
    """Create first block base on VGG16 and get its pretrained weights
        Thiếu maxpool ở layer4
    """

    def __init__(self):
        super(Base, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), # (N, 64, 300, 300)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # (N, 64, 300, 300)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # (N, 64, 150, 150)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), #(N, 128, 150, 150)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), # (N, 128, 150, 150)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # (N, 128, 75, 75)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), # (N, 256, 75, 75)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # (N, 256, 75, 75)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # (N, 256, 75, 75)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) # (N, 256, 38, 38)
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), # (N, 512, 38, 38)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # (N, 512, 38, 38)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # (N, 512, 38, 38)
            nn.ReLU(inplace=True)
            # nn.MaxPool2d(kernel_size=2, stride=2) # (N, 512, 19, 19)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) #(N, 512, 19, 19)

        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # (N, 512, 19, 19)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # (N, 512, 19, 19)
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # (N, 512, 19, 19)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1) # (N, 512, 19, 19)
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6), # (N, 1024, 19, 19)
            nn.ReLU(inplace=True),
        )
        
        self.layer7 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1), # (N, 1024, 19, 19)
            nn.ReLU(inplace=True),
        )

        self.load_pretrained_params()

    def forward(self, input):
        #Base
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        conv4_3_feat = out
        # print(conv4_3_feat.shape)
        out = self.pool4(out)

        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        
        conv7_feat = out

        return conv4_3_feat, conv7_feat 


    def load_pretrained_params(self):

        state_dict = self.state_dict()
        params_keys = list(state_dict.keys())
        

        vgg = torchvision.models.vgg16(pretrained=True)
            
        pretrained_state_dict = vgg.state_dict()
        pretrained_params_keys = list(pretrained_state_dict.keys())

        for i, key in enumerate(params_keys[:-4]):
            state_dict[key] = pretrained_state_dict[pretrained_params_keys[i]]
        # print(params_keys)
        # print(pretrained_params_keys)
        #Convert fc6, fc7 to convolutional layers
        w_fc6 = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)
        b_fc6 = pretrained_state_dict['classifier.0.bias'] # (4096,)

        w_fc7 = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)
        b_fc7 = pretrained_state_dict['classifier.3.bias'] #(4096, )

        # Subsample parameters of fc6, fc7
        w_conv6 = torch.index_select(input=w_fc6, dim=0, index=torch.arange(0, 4096, step=4)) # (1024, 512, 7, 7)
        w_conv6 = torch.index_select(input=w_conv6, dim=2, index=torch.arange(0, 7, step=3)) # (1024, 512, 3, 7)
        w_conv6 = torch.index_select(input=w_conv6, dim=3, index=torch.arange(0, 7, step=3)) #(1024, 512, 3, 3)
        
        b_conv6 = torch.index_select(input=b_fc6, dim=0, index=torch.arange(0, 4096, step=4)) #(1024,)


        w_conv7 = torch.index_select(input=w_fc7, dim=0, index=torch.arange(0, 4096, step=4)) #(1024, 4096, 1, 1)
        w_conv7 = torch.index_select(input=w_conv7, dim=1, index=torch.arange(0, 4096, step=4)) #(1024, 1024, 1, 1)

        b_conv7 = torch.index_select(input=b_fc7, dim=0, index=torch.arange(0, 4096, step=4)) #(1024,)


        state_dict['layer6.0.weight'] = w_conv6
        state_dict['layer6.0.bias'] = b_conv6
        state_dict['layer7.0.weight'] = w_conv7
        state_dict['layer7.0.bias'] = b_conv7

        self.load_state_dict(state_dict)

        print('Loaded pretrained model VGG to Base.') 


class Extras(nn.Module):
    """Extra block with weight initialized by Xavier method"""
    def __init__(self):
        super(Extras, self).__init__()
        self.layer8 = nn.Sequential(
            nn.Conv2d(1024, 256, kernel_size=1, padding=0), #(N, 256, 19, 19)
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2), # (N, 512, 10, 10)
            nn.ReLU(inplace=True),
        )

        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, padding=0), # (N, 128, 10, 10)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # (N, 256, 5, 5)
            nn.ReLU(inplace=True),
        )

        self.layer10 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, padding=0), #(N, 128, 5, 5)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=0), # (N, 256, 3, 3)
            nn.ReLU(inplace=True),
        )
        
        self.layer11 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, padding=0), #(N, 128, 3, 3)
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=0), #(N, 256, 1, 1)
            nn.ReLU(inplace=True),
        )

        self.init_params()

    def init_params(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)
        
    def forward(self, input):
        #Extras
        out = self.layer8(input)
        conv8_2_feat = out
        out = self.layer9(out)
        conv9_2_feat = out
        out = self.layer10(out)
        conv10_2_feat = out
        conv11_2_feat = self.layer11(out)

        return conv8_2_feat, conv9_2_feat, conv10_2_feat, conv11_2_feat


class Predictions(nn.Module):
    """Predict block
        Returns:
            locs(tensor): (batch, 8732, 4): offsets of each boxes
            confs(tensor): (batch, 8732, 21): confidences of each boxes
    """
    def __init__(self, num_classes):
        super(Predictions, self).__init__()
        self.num_classes = num_classes
        num_boxes = {'conv4_3':4, 'conv7':6, 'conv8_2':6, 'conv9_2':6, 'conv10_2':4, 'conv11_2':4} #Number of default boxes for each feature

        #Location
        self.loc_conv4_3 = nn.Conv2d(512, num_boxes['conv4_3']*4, kernel_size=3, padding=1)
        self.loc_conv7 = nn.Conv2d(1024, num_boxes['conv7']*4, kernel_size=3, padding=1)
        self.loc_conv8_2 = nn.Conv2d(512, num_boxes['conv8_2']*4, kernel_size=3, padding=1)
        self.loc_conv9_2 = nn.Conv2d(256, num_boxes['conv9_2']*4, kernel_size=3, padding=1)
        self.loc_conv10_2 = nn.Conv2d(256, num_boxes['conv10_2']*4, kernel_size=3, padding=1)
        self.loc_conv11_2 = nn.Conv2d(256, num_boxes['conv11_2']*4, kernel_size=3, padding=1)

        #Classify
        self.cl_conv4_3 = nn.Conv2d(512, num_boxes['conv4_3']*self.num_classes, kernel_size=3, padding=1)
        self.cl_conv7 = nn.Conv2d(1024, num_boxes['conv7']*self.num_classes, kernel_size=3, padding=1)
        self.cl_conv8_2 = nn.Conv2d(512, num_boxes['conv8_2']*self.num_classes, kernel_size=3, padding=1)
        self.cl_conv9_2 = nn.Conv2d(256, num_boxes['conv9_2']*self.num_classes, kernel_size=3, padding=1)
        self.cl_conv10_2 = nn.Conv2d(256, num_boxes['conv10_2']*self.num_classes, kernel_size=3, padding=1)
        self.cl_conv11_2 = nn.Conv2d(256, num_boxes['conv11_2']*self.num_classes, kernel_size=3, padding=1)

        self.init_params()

    def init_params(self):
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)
    
    
    '''If you just want to reshape tensors, use torch.reshape.
       If you're also concerned about memory usage and want to ensure that the two tensors share the same data, use torch.view.'''
       
    def forward(self, conv4_3_feat, conv7_feat, conv8_2_feat, conv9_2_feat, conv10_2_feat, conv11_2_feat):
        batch_size = conv4_3_feat.shape[0]

        #Location
        loc_conv4_3 = self.loc_conv4_3(conv4_3_feat) # (N, 16, 38, 38)
        loc_conv4_3 = loc_conv4_3.permute(0, 2, 3, 1).contiguous() # (N, 38, 38, 16)
        loc_conv4_3 = loc_conv4_3.view(batch_size, -1, 4) #(N, 5776, 4)
        # print(loc_conv4_3.shape)
        
        loc_conv7 = self.loc_conv7(conv7_feat) #(N, 24, 19, 19)
        loc_conv7 = loc_conv7.permute(0, 2, 3, 1).contiguous() #(N, 19, 19, 24)
        loc_conv7 = loc_conv7.view(batch_size, -1, 4) #(N, 2166, 4)

        loc_conv8_2 = self.loc_conv8_2(conv8_2_feat)
        loc_conv8_2 = loc_conv8_2.permute(0, 2, 3, 1).contiguous()
        loc_conv8_2 = loc_conv8_2.view(batch_size, -1, 4)
        
        loc_conv9_2 = self.loc_conv9_2(conv9_2_feat)
        loc_conv9_2 = loc_conv9_2.permute(0, 2, 3, 1).contiguous()
        loc_conv9_2 = loc_conv9_2.view(batch_size, -1, 4)

        loc_conv10_2 = self.loc_conv10_2(conv10_2_feat)
        loc_conv10_2 = loc_conv10_2.permute(0, 2, 3, 1).contiguous()
        loc_conv10_2 = loc_conv10_2.view(batch_size, -1, 4)

        loc_conv11_2 = self.loc_conv11_2(conv11_2_feat)
        loc_conv11_2 = loc_conv11_2.permute(0, 2, 3, 1).contiguous()
        loc_conv11_2 = loc_conv11_2.view(batch_size, -1, 4)

        #Classification
        cl_conv4_3 = self.cl_conv4_3(conv4_3_feat)  #(N, classes*4, 38, 38)
        cl_conv4_3 = cl_conv4_3.permute(0, 2, 3, 1).contiguous() #(N, 38, 38, classes*4)
        cl_conv4_3 = cl_conv4_3.view(batch_size, -1, self.num_classes) # (N, 5776, classes)

        cl_conv7 = self.cl_conv7(conv7_feat) #(N, classes*6, 19, 19)
        cl_conv7 = cl_conv7.permute(0, 2, 3, 1).contiguous() # (N, 19, 19, classes*6)
        cl_conv7 = cl_conv7.view(batch_size, -1, self.num_classes) # (N, 2166, classes)

        cl_conv8_2 = self.cl_conv8_2(conv8_2_feat)
        cl_conv8_2 = cl_conv8_2.permute(0, 2, 3, 1).contiguous()
        cl_conv8_2 = cl_conv8_2.view(batch_size, -1, self.num_classes)

        cl_conv9_2 = self.cl_conv9_2(conv9_2_feat)
        cl_conv9_2 = cl_conv9_2.permute(0, 2, 3, 1).contiguous()
        cl_conv9_2 = cl_conv9_2.view(batch_size, -1, self.num_classes)

        cl_conv10_2 = self.cl_conv10_2(conv10_2_feat)
        cl_conv10_2 = cl_conv10_2.permute(0, 2, 3, 1).contiguous()
        cl_conv10_2 = cl_conv10_2.view(batch_size, -1, self.num_classes)

        cl_conv11_2 = self.cl_conv11_2(conv11_2_feat)
        cl_conv11_2 = cl_conv11_2.permute(0, 2, 3, 1).contiguous()
        cl_conv11_2 = cl_conv11_2.view(batch_size, -1, self.num_classes)

        
        locs = torch.cat((loc_conv4_3, loc_conv7, loc_conv8_2, loc_conv9_2, loc_conv10_2, loc_conv11_2), dim=1) # dim: the dimention over which the tensors are concatnated
        confs = torch.cat((cl_conv4_3, cl_conv7, cl_conv8_2, cl_conv9_2, cl_conv10_2, cl_conv11_2), dim=1) 

        return locs, confs


class SSD300(nn.Module):
    def __init__(self, num_classes):
        super(SSD300, self).__init__()

        self.num_classes = num_classes

        self.base = Base()
        
        for params in self.base.layer1.parameters():
            params.requires_grad = False

        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # for L2 norm
        nn.init.constant_(self.rescale_factors, 20)
            
        self.extras = Extras()
        self.predict = Predictions(num_classes)

        self.def_boxes = create_default_boxes()


    def forward(self, image):
        conv4_3_feat, conv7_feat = self.base(image) #(N, 512, 38, 38), (N, 1024, 19, 19)

        # L2 Norm 
        norm = conv4_3_feat.pow(2).sum(dim=1, keepdim=True) #(N, 1, 38, 38)
        norm = torch.sqrt(norm)
        conv4_3_feat = conv4_3_feat / norm #(N, 1, 38, 38)
        conv4_3_feat = conv4_3_feat * self.rescale_factors



        conv8_2_feat, conv9_2_feat, conv10_2_feat, conv11_2_feat = self.extras(conv7_feat)
        locs, confs = self.predict(conv4_3_feat, conv7_feat, conv8_2_feat, conv9_2_feat, conv10_2_feat, conv11_2_feat)


        output = (locs, confs, self.def_boxes)

        return output

def create_default_boxes():
    fmap_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

    obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

    aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}

    fmaps = list(fmap_dims.keys())

    def_boxes = []

    for k, fmap in enumerate(fmaps):
        for i in range(fmap_dims[fmap]):
            for j in range(fmap_dims[fmap]):
                cx = (j + 0.5) / fmap_dims[fmap]
                cy = (i + 0.5) / fmap_dims[fmap]

                for ratio in aspect_ratios[fmap]:
                    def_boxes.append([cx, cy, obj_scales[fmap] * np.sqrt(ratio), obj_scales[fmap] / np.sqrt(ratio)])

                    if ratio == 1.:
                        try:
                            additional_scale = np.sqrt(obj_scales[fmap] * obj_scales[fmaps[k + 1]])
                            
                        except IndexError:
                            additional_scale = 1.
                        def_boxes.append([cx, cy, additional_scale, additional_scale])

    def_boxes = torch.FloatTensor(def_boxes)  # (8732, 4)
    def_boxes.clamp_(0, 1)  # (8732, 4)

    return def_boxes.to(device)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0))
    model = SSD300(21)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(model)
    print('Trainable parameters =', trainable_params)
    print('Total parameters =', total_params)
