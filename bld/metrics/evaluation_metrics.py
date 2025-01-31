class EvaluationMetrics:
    """
    Storing the metrics data.
    """
    def __init__(self, msi: float, hausdorff: float, dice: float, jaccard: float):
        self.msi = msi
        self.hausdorff = hausdorff
        self.dice = dice
        self.jaccard = jaccard
