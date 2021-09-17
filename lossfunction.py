
from libs import *
from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiboxLoss(nn.Module):
    """ 
        1. localization loss
        2. confidence loss for predict class score

        Args:
            threshold: predicted boxes have smaller overlap with groundtruths box will be set to 0
            neg_pos_ratio: using in hard negative mining to decrease number of negative boxes
            alpha:

            predictions(tuple of tensors): output from model(locs, confs, def_boxes)
            targets(list of tensors, one tensor for each image): get from data(boundary coordinates and label)
        Returns:
            loss
    """

    def __init__(self, threshold=0.5, neg_pos_ratio=3, alpha=1):
        super(MultiboxLoss, self).__init__()

        self.threshold = threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.alpha = alpha
        self.L1Loss = nn.SmoothL1Loss(reduction='sum')
        self.CrossentropyLoss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, predictions, targets):
        locs, confs, def_boxes = predictions # (batch, 8732, 4), (batch, 8732, 21), (8732, 4)

        batch_size = confs.size(0)
        num_boxes = confs.size(1) #8732
        num_classes = confs.size(2) #21

        confs_t_labels = torch.LongTensor(batch_size, num_boxes).to(device)
        locs_t = torch.Tensor(batch_size, num_boxes, 4).to(device)
        # print(locs_t.shape)
        
        for i in range(batch_size):
            truths = targets[i][:, :-1].to(device) # xmin, ymin, xmax, ymax
            labels = targets[i][:, -1].to(device) # label
            def_boxes = def_boxes.to(device)
            match(self.threshold, truths, def_boxes, labels, locs_t, confs_t_labels, i)

    #Loc_loss
        pos_mask = confs_t_labels > 0  #(batch, 8732)
        n_positive = pos_mask.sum(dim=1) # (batch,)
        # print(n_positive.shape)
        # print(n_positive)
        pos_idx = pos_mask.unsqueeze(2).expand_as(locs_t) #(batch, 8732, 4)

        locs_pred = locs[pos_idx].view(-1, 4)
        locs_t = locs_t[pos_idx].view(-1, 4)

        loc_loss = self.L1Loss(locs_pred, locs_t)
        # print(loc_loss)

    # Conf_loss
        true_classes = confs_t_labels.view(-1).to(device) #(8732)
        predicted_scores = confs.view(-1, num_classes)

        conf_loss_all = self.CrossentropyLoss(predicted_scores, true_classes) # (batch * 8732)
        conf_loss_all = conf_loss_all.view(batch_size, -1) # (batch, 8732)

        # hard negative mining

        n_hard_negative = torch.clamp(n_positive*self.neg_pos_ratio, max=num_boxes) # (batch, )

        conf_loss_pos = conf_loss_all[pos_mask] #(N, )

        conf_loss_neg = conf_loss_all.clone()
        conf_loss_neg[pos_mask] = 0.  # chuyển loss của positive về 0

        conf_loss_neg, _ = torch.sort(conf_loss_neg, dim=1, descending=True)

        loss_rank = torch.LongTensor(range(num_boxes)).unsqueeze(0).expand_as(conf_loss_neg).to(device) # (batch, 8732)
        hard_negative = loss_rank < n_hard_negative.unsqueeze(1).expand_as(loss_rank)

        conf_loss_hard_neg = conf_loss_neg[hard_negative] 

        conf_loss = (conf_loss_pos.sum() + conf_loss_hard_neg.sum()) 
        # print(conf_loss_pos.sum(),  conf_loss_hard_neg.sum())

        loss = (conf_loss + self.alpha * loc_loss) / n_positive.sum().float()


        return loss
