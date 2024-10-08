class CorrelationAnalyzer:
    def __init__(self, msi, hausdorff, dice, jaccard, manual_score):
        self.msi = msi
        self.hausdorff = hausdorff
        self.dice = dice
        self.jaccard = jaccard
        self.manual_score = manual_score
        self.median = self.calculate_median()

    def calculate_median(self):
        """Calculate the median for each element in the msi list."""
        median = []
        for i in range(len(self.msi)):
            if len(self.msi[i]) > 1:
                median.append(np.median(self.msi[i]))
            else:
                median.append(self.msi[i][0])
        return median

    def find_spearman_correlation_with_metrics(self):
        """Find the Spearman correlation between the median and the other metrics."""
        correlation_1_2, _ = spearmanr(self.median, self.hausdorff)
        correlation_1_3, _ = spearmanr(self.median, self.dice)
        correlation_1_4, _ = spearmanr(self.median, self.jaccard)
        print("Spearman MSI with Hausdorff:", correlation_1_2)
        print("Spearman MSI with Dice:", correlation_1_3)
        print("Spearman MSI with Jaccard:", correlation_1_4)
        return correlation_1_2, correlation_1_3, correlation_1_4

    def find_pearson_correlation_with_metrics(self):
        """Find the Pearson correlation between the median and traditional metrics."""
        correlation_1_2, _ = pearsonr(self.median, self.hausdorff)
        correlation_1_3, _ = pearsonr(self.median, self.dice)
        correlation_1_4, _ = pearsonr(self.median, self.jaccard)
        print("Pearson MSI with Hausdorff:", correlation_1_2)
        print("Pearson MSI with Dice:", correlation_1_3)
        print("Pearson MSI with Jaccard:", correlation_1_4)
        return correlation_1_2, correlation_1_3, correlation_1_4

    def find_spearman_pearson_with_manual_score(self):
        """Find both Spearman and Pearson correlations between the median MSI and manual score."""
        correlation_1_5, _ = spearmanr(self.median, self.manual_score)
        correlation_1_6, _ = pearsonr(self.median, self.manual_score)
        print("Spearman MSI and manual score:", correlation_1_5)
        print("Pearson MSI and manual score:", correlation_1_6)
        return correlation_1_5, correlation_1_6

    def find_spearman_pearson_traditional_with_manual_score(self, traditional_metric):
        """Find both Spearman and Pearson correlations between a traditional metric and manual score."""
        correlation_1_5, _ = spearmanr(traditional_metric, self.manual_score)
        correlation_1_6, _ = pearsonr(traditional_metric, self.manual_score)
        print("Spearman traditional metric and manual score:", correlation_1_5)
        print("Pearson traditional metric and manual score:", correlation_1_6)
        return correlation_1_5, correlation_1_6
