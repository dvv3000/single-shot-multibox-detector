from libs import *


def decode(locs, def_boxes):
    """ Tinhs bouding boxes: offsets ---> center-size coords ---> boundary coords
        Args:
            locs: (8732, 4)         
            def_boxes: (8732, 4)
        Returns:
            boxes: (8732, 4) boundary coords
    """

    boxes = torch.cat((
        def_boxes[:, :2] + locs[:, :2] * def_boxes[:, 2:] * 0.1, #cx, cy
        def_boxes[:, 2:] * torch.exp(locs[:, 2:] * 0.2)), dim=1) #w, h

    boxes[:, :2] -= boxes[:, 2:] / 2  #xmin, ymin
    boxes[:, 2:] += boxes[:, :2]  # xmax, ymax
 
    return boxes


def encode(matches, def_boxes):
    """ Chuyen boxes ve dang offsets
        Args:
            matches: (8732, 4)      boundary coords ---> center-size coords ---> offset
            def_boxes: (8732, 4)
        Return:
            offset: (8732, 4) 
    """
    g_cxcy = (matches[:, :2] + matches[:, 2:]) / 2 #cxcy
    g_wh = matches[:, 2:] - matches[:, :2]
    
    g_hat_cxcy = (g_cxcy - def_boxes[:, :2]) / (def_boxes[:, 2:] * 0.1)
    g_hat_wh = torch.log(g_wh / def_boxes[:, 2:]) / 0.2 

    locs = torch.cat([g_hat_cxcy, g_hat_wh], dim=1)
    
    return locs


def intersect(box_a, box_b):
    """Args:
        box_a : tensor (num_boxes_a, 4)
        box_b : tensor (num_boxes_b, 4)

       Return:
        intersection area: tensor (num_boxes_A, num_boxes_B)
    """
    A = box_a.size(0)
    B = box_b.size(0)

    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2), box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2), box_b[:, 2:].unsqueeze(0).expand(A, B, 2))

    inter = torch.clamp((max_xy - min_xy), min=0)
    
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """
    """ 
    inter = intersect(box_a, box_b) # (num_boxes_a, num_boxes_b)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]) #(num_boxes_a, )
    area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1]) #(num_boxes_b, )

    area_a.unsqueeze_(1).expand_as(inter)
    area_b.unsqueeze_(0).expand_as(inter)

    union = area_a + area_b - inter

    return inter / union


def cxcy_to_xy(boxes):
    """ Convert center-size coords to boundary coords

    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax

                     
def match(threshhold, truths, def_boxes, labels, locs_t, confs_t, idx):

    """
        Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.

    """

    dbox_xy = cxcy_to_xy(def_boxes) # to xmin ymin xmax ymax (8732, 4)

    overlap = jaccard(truths, dbox_xy) # (num_truth_boxes, 8732)

    best_dbox_overlap, best_dbox_idx = torch.max(overlap, dim=1) # (num_truth_boxes, 1)
    best_truth_overlap, best_truth_idx = torch.max(overlap, dim=0) #(8732, 1)

    best_truth_overlap.index_fill_(0, best_dbox_idx, 2.)  # to ensure best dbox

    for j in range(best_dbox_idx.size(0)):
        best_truth_idx[best_dbox_idx[j]] = j

    matches = truths[best_truth_idx]  # (8732, 4)
    confs = labels[best_truth_idx] + 1 # set label from truth box to each dbox (8732, 1)
    confs[best_truth_overlap < threshhold] = 0 # set background to 0 #(8732, 1)

    # print(truths.shape)
    # print(matches.shape)

    locs = encode(matches, def_boxes)  # (8732, 4)
    locs_t[idx] = locs 
    confs_t[idx] = confs
