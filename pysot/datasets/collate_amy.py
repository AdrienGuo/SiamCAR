import torch


def collate_fn_new(batch):
    t_img,s_img,cls,box = zip(*batch)  
    for i, l in enumerate(box):
        
        l[:, 0] = i  # add target image index for build_targets()
    return {'template': torch.stack(t_img, 0), 'search': torch.stack(s_img, 0),\
            'label_cls': torch.stack(cls, 0),'bbox': torch.cat(box, 0)}


