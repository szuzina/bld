import numpy as np
from scipy.stats import spearmanr, pearsonr


class CorrelationAnalyzer:
    def __init__(self, evaluation_metrics, manual_score):
        self.metrics = evaluation_metrics
        self.manual_score = manual_score
        self.median = []

        self.results = {}

    def calculate_median(self):
        """Calculate the median for each element in the msi list."""
        median = []
        for i in range(len(self.metrics.msi)):
            median.append(np.median(self.metrics.msi[i]))
        self.median=median
        return 0

    def find_spearman_correlation_with_metrics(self):
        """Find the Spearman correlation between the median and the other metrics."""
        self.results.update({
            "spearman_msi_hausdorff": spearmanr(self.median, self.metrics.hausdorff),
            "spearman_msi_dice": spearmanr(self.median, self.metrics.dice),
            "spearman_msi_jaccard": spearmanr(self.median, self.metrics.jaccard),
            "spearman_msi_manual": spearmanr(self.median, self.manual_score)
        })

    def find_pearson_correlation_with_metrics(self):
        """Find the Pearson correlation between the median and traditional metrics."""
        self.results.update({
            "pearson_msi_hausdorff": pearsonr(self.median, self.metrics.hausdorff),
            "pearson_msi_dice": pearsonr(self.median, self.metrics.dice),
            "pearson_msi_jaccard": pearsonr(self.median, self.metrics.jaccard),
            "pearson_msi_manual": pearsonr(self.median, self.manual_score)
        })

    def find_spearman_pearson_traditional_with_manual_score(self):
        """Find both Spearman and Pearson correlations between a traditional metric and manual score."""
        self.results.update({
            "spearman_hausdorff_manual": spearmanr(self.metrics.hausdorff, self.manual_score),
            "pearson_hausdorff_manual": pearsonr(self.metrics.hausdorff, self.manual_score),
            "spearman_dice_manual": spearmanr(self.metrics.dice, self.manual_score),
            "pearson_dice_manual": pearsonr(self.metrics.dice, self.manual_score),
            "spearman_jaccard_manual": spearmanr(self.metrics.jaccard, self.manual_score),
            "pearson_jaccard_manual": pearsonr(self.metrics.jaccard, self.manual_score)
        })

    def run(self):
        self.calculate_median()
        self.find_spearman_correlation_with_metrics()
        self.find_pearson_correlation_with_metrics()
        self.find_spearman_pearson_traditional_with_manual_score()
