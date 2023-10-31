class CrossEntropyReweightLoss(CrossEntropyLoss):    
    def forward(self, input: Tensor, target: Tensor, reweight_coeff: Tensor) -> Tensor:
        return F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction,
                               label_smoothing=self.label_smoothing)