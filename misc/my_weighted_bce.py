class my_weighted_bce():
    def __init__(self, weight: torch.tensor):
        self.w = weight

    def __call__(self, y_pred: torch.tensor, y_true: torch.tensor):
        # weight proportion of nMajority/nMin should be for dist in training only
        w_bce = torch.mean(-(self.w*(y_true*torch.log(y_pred))+(1-y_true)*torch.log(1-y_pred)))
        return w_bce