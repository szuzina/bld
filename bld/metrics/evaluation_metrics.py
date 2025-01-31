class EvaluationMetrics:
    """
    Storing the metrics data.
    """
    def __init__(self, msi: list, hausdorff: list, dice: list, jaccard: list):
        self.msi = msi
        self.hausdorff = hausdorff
        self.dice = dice
        self.jaccard = jaccard
