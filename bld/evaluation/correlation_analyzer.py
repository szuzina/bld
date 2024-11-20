import numpy as np
from scipy.stats import spearmanr, pearsonr


class CorrelationAnalyzer:
    def __init__(self, evaluation_metrics, manual_score):
        self.metrics = evaluation_metrics
        self.manual_score = manual_score
        self.median = self.calculate_median()

        self.results = {}

    def calculate_median(self):
        """Calculate the median for each element in the msi list."""
        median = []
        for i in range(len(self.metrics.msi)):
            if len(self.metrics.msi[i]) > 1:
                median.append(np.median(self.metrics.msi[i]))
            else:
                median.append(self.metrics.msi[i][0])
        return median

    def find_spearman_correlation_with_metrics(self):
        """Find the Spearman correlation between the median and the other metrics."""
        correlation_1_2 = spearmanr(self.median, self.metrics.hausdorff)
        correlation_1_3 = spearmanr(self.median, self.metrics.dice)
        correlation_1_4 = spearmanr(self.median, self.metrics.jaccard)
        correlation_1_5 = spearmanr(self.median, self.manual_score)

        self.results["spearman_msi_hausdorff"] = correlation_1_2
        self.results["spearman_msi_dice"] = correlation_1_3
        self.results["spearman_msi_jaccard"] = correlation_1_4
        self.results["spearman_msi_manual"] = correlation_1_5

    def find_pearson_correlation_with_metrics(self):
        """Find the Pearson correlation between the median and traditional metrics."""
        correlation_1_2 = pearsonr(self.median, self.metrics.hausdorff)
        correlation_1_3 = pearsonr(self.median, self.metrics.dice)
        correlation_1_4 = pearsonr(self.median, self.metrics.jaccard)
        correlation_1_6 = pearsonr(self.median, self.manual_score)

        self.results["pearson_msi_hausdorff"] = correlation_1_2
        self.results["pearson_msi_dice"] = correlation_1_3
        self.results["pearson_msi_jaccard"] = correlation_1_4
        self.results["pearson_msi_manual"] = correlation_1_6

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
        self.find_spearman_correlation_with_metrics()
        self.find_pearson_correlation_with_metrics()
        self.find_spearman_pearson_traditional_with_manual_score()
