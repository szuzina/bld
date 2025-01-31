class EvaluationMetrics:
    """
    Storing the metrics data.
    """
    def __init__(self, msi, hausdorff, dice, jaccard):
        self.msi = msi
        self.hausdorff = hausdorff
        self.dice = dice
        self.jaccard = jaccard
