import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import binomtest
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.stats.diagnostic import het_white, acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera


class ResidualAnalysis:
    def __init__(self, residuals, exog_vars=None, alpha=0.05,lags=2):
        self.residuals = residuals
        self.exog_vars = exog_vars
        self.alpha = alpha
        self.lags=lags

    def test_mean_residuals(self):
        mean_resid = np.mean(self.residuals)
        t_stat, p_value = stats.ttest_1samp(self.residuals, 0)
        result = "Reject H0: Residuals' mean ≠ 0" if p_value < self.alpha else "Cannot reject H0: Residuals' mean = 0"

        return {
            "Mean of residuals": mean_resid,
            "t-statistic": t_stat,
            "p-value": p_value,
            "Result": result
        }

    def test_residuals_variance(self):
        if self.exog_vars is None:
            raise ValueError("Provide exog_vars for White's test.")

        self.exog_vars['const'] = self.exog_vars.get('const', 1)

        lm_stat, lm_pvalue, f_stat, f_pvalue = het_white(self.residuals, self.exog_vars)
        result = "Reject H0: Heteroskedastic residuals" if lm_pvalue < self.alpha else "Cannot reject H0: Possibly homoskedastic residuals"

        return {
            "LM Statistic": lm_stat,
            "LM p-value": lm_pvalue,
            "F-statistic": f_stat,
            "F p-value": f_pvalue,
            "Result": result
        }

    def test_normality_jarque_bera(self):
        jb_stat, jb_pvalue, skewness, kurtosis = jarque_bera(self.residuals)
        result = "Reject H0: Non-normal residuals" if jb_pvalue < self.alpha else "Cannot reject H0: Possibly normal residuals"

        return {
            "JB Statistic": jb_stat,
            "p-value": jb_pvalue,
            "Skewness": skewness,
            "Kurtosis": kurtosis,
            "Result": result
        }

    def test_residual_autocorrelation(self, lags=3):
        lb_results = acorr_ljungbox(self.residuals.values, lags=lags, return_df=True)
        worst_p_value = min(lb_results['lb_pvalue'].values)



        result = "Reject H0: Residuals show autocorrelation" if worst_p_value < self.alpha else "Cannot reject H0: No evident autocorrelation"

        return {
            "LB Statistic for max lag": lb_results['lb_stat'].iloc[-1],
            "Worst p-value": worst_p_value,
            "Result": result
        }

    def test_residual_independence_with_spearman(self):
        worst_p_value = 1.0  # initialize with a value that will always be replaced
        worst_coeff = None
        worst_col_name = None

        for col in self.exog_vars.columns:
            if col == 'const':
                continue  # Skip correlation with 'const'
            coeff, p_value = stats.spearmanr(self.residuals, self.exog_vars[col])

            # Check if the current p_value is worse than the previous worst
            if p_value < worst_p_value:
                worst_p_value = p_value
                worst_coeff = coeff
                worst_col_name = col

        result_string = (
            f"Reject the null hypothesis: Residuals might depend on {worst_col_name}."
            if worst_p_value < self.alpha else
            f"Fail to reject the null hypothesis: Residuals are likely independent of {worst_col_name}."
        )

        return {
            "Worst Variable": worst_col_name,
            "Spearman's rho": worst_coeff,
            "p-value": worst_p_value,
            "Result": result_string
        }

    def test_randomwalk(self, Y=None):
        if Y is None:
            Y = self.residuals

        def create_sequence(Y):
            return [1 if Y[i + 1] > Y[i] else 0 for i in range(len(Y) - 1)]

        sequence = create_sequence(Y)

        counts = {
            (0, 0): 0,
            (0, 1): 0,
            (1, 0): 0,
            (1, 1): 0
        }

        for i in range(len(sequence) - 1):
            counts[(sequence[i], sequence[i + 1])] += 1

        p_values = [binomtest(counts[key], counts[key] + counts[(key[0], 1 - key[1])], 0.5).pvalue for key in counts]
        worst_p_value = min(p_values)  # znajdowanie najgorszej wartości p

        result = "Reject H0: Not a random walk" if worst_p_value < self.alpha else "Cannot reject H0: Possibly a random walk"

        return {
            "Counts": counts,
            "Worst p-value": worst_p_value,  # zwracanie najgorszej wartości p
            "Result": result
        }

    def loss_function(self, alpha=0.05, weights=None):
        """
        Calculate a loss value based on the p-values from residual diagnostics.

        :param alpha: Significance level (default=0.05).
        :param weights: List of weights to give more or less importance to specific tests.
        :return: Total loss value, where lower values indicate better fitting models.
        """
        residual_results = {
            'Mean of residuals': self.test_mean_residuals(),
            'LM Statistic': self.test_residuals_variance(),
            'JB Statistic': self.test_normality_jarque_bera(),
            'LB Statistic for max lag': self.test_residual_autocorrelation(lags=2),
            'Worst Variable': self.test_residual_independence_with_spearman(),
            'Counts': self.test_randomwalk()
        }

        # List of p-values from the results
        p_values = [
            residual_results['Mean of residuals']['p-value'],
            residual_results['LM Statistic']['LM p-value'],
            residual_results['JB Statistic']['p-value'],
            residual_results['LB Statistic for max lag']['Worst p-value'],
            residual_results["Worst Variable"]['p-value'],
            residual_results["Counts"]['Worst p-value']
        ]

        # Default weights for tests (all have the same weight if none are provided)
        if weights is None:
            weights = [1] * len(p_values)

        # Calculate the loss function as the sum of (alpha - p_value)^2 for p_value < alpha.
        # If p_value > alpha, it's considered 0.
        loss_values = [((alpha - p) ** 2) * w if p < alpha else 0 for p, w in zip(p_values, weights)]

        total_loss = sum(loss_values)

        return total_loss

    def make_tests(self):
        print("Test Mean of Residuals:")
        print(self.test_mean_residuals())
        print("\nTest Residuals Variance:")
        print(self.test_residuals_variance())
        print("\nTest Normality (Jarque-Bera):")
        print(self.test_normality_jarque_bera())
        print("\nTest Residual Autocorrelation:")
        print(self.test_residual_autocorrelation(lags=self.lags))
        plot_pacf(
            self.residuals,
            lags=self.lags,
            method="yw",
            alpha=self.alpha,
            use_vlines=True,
            title='Partial Autocorrelation',
            zero=False
        )
        plt.show()
        print("\nTest Residual Independence with Spearman:")
        print(self.test_residual_independence_with_spearman())
        print("\nTest Random Walk:")
        print(self.test_randomwalk())
        print("\nLoss Function:")
        loss_val = self.loss_function()
        print(f"Loss Function Value: {loss_val}")

if __name__ == "__main__":
    df = pd.DataFrame({
        'resid': [1.2, 0.5, -0.5, -1.2, 0.3, 0.4, -0.2],
        'x1': [2, 3, 1, 5, 4, 2, 3],
        'x2': [5, 6, 5, 7, 6, 5, 6]
    })

    res = ResidualAnalysis(df['resid'], exog_vars=df[['x1', 'x2']],alpha=0.05,lags=2)
    res.make_tests()
    print(res.loss_function())
