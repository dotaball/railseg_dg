import torch


class DataPrefetcher(object):
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_rgb, self.next_gt, self.next_cla,_,_ = next(self.loader)
        except StopIteration:
            self.next_rgb = None
            self.next_gt = None
            self.next_cla = None
            return

        with torch.cuda.stream(self.stream):
            self.next_rgb = self.next_rgb.cuda(non_blocking=True).float()
            self.next_gt = self.next_gt.cuda(non_blocking=True).float()
            self.next_cla = self.next_cla.cuda(non_blocking=True).float()


    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        rgb = self.next_rgb
        gt = self.next_gt
        cla = self.next_cla
        self.preload()
        return rgb, gt, cla