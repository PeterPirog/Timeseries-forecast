import matplotlib.pyplot as plt
import numpy as np
import math
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

    def loss_function_old(self, alpha=0.05, weights=None):
        residual_results = {
            'Mean of residuals': self.test_mean_residuals(),
            'LM Statistic': self.test_residuals_variance(),
            'JB Statistic': self.test_normality_jarque_bera(),
            'LB Statistic for max lag': self.test_residual_autocorrelation(lags=2),
            'Worst Variable': self.test_residual_independence_with_spearman(),
        }

        p_values = [
            residual_results['Mean of residuals']['p-value'],
            residual_results['LM Statistic']['LM p-value'],
            residual_results['JB Statistic']['p-value'],
            residual_results['LB Statistic for max lag']['Worst p-value'],
            residual_results["Worst Variable"]['p-value'],
        ]

        if weights is None:
            weights = [1] * len(p_values)

        loss_values = [((alpha - p) ** 2) * w if p < alpha else 0 for p, w in zip(p_values, weights)]

        total_loss = sum(loss_values)

        return total_loss

    def loss_function(self, alpha=0.05, weights=None):
        residual_results = {
            'Mean of residuals': self.test_mean_residuals(),
            'LM Statistic': self.test_residuals_variance(),
            'JB Statistic': self.test_normality_jarque_bera(),
            'LB Statistic for max lag': self.test_residual_autocorrelation(lags=2),
            'Worst Variable': self.test_residual_independence_with_spearman(),
        }

        p_values = [
            residual_results['Mean of residuals']['p-value'],
            residual_results['LM Statistic']['LM p-value'],
            residual_results['JB Statistic']['p-value'],
            residual_results['LB Statistic for max lag']['Worst p-value'],
            residual_results["Worst Variable"]['p-value'],
        ]

        if weights is None:
            weights = [1] * len(p_values)

        # Modified loss function
        def compute_loss(p, alpha=0.05, w=1):
            m1 = (1 - 100) / (alpha - 0)
            c1 = 100

            m2 = (0 - 1) / (1 - alpha)
            c2 = 1 - m2 * alpha

            if p <= alpha:
                return w * (m1 * p + c1)
            else:
                return w * (m2 * p + c2)

        loss_values = [compute_loss(p, alpha, w) for p, w in zip(p_values, weights)]
        total_loss = sum(loss_values)

        return total_loss

    def make_tests(self):
        results = {}

        results["Test Mean of Residuals"] = self.test_mean_residuals()
        results["Test Residuals Variance"] = self.test_residuals_variance()
        results["Test Normality (Jarque-Bera)"] = self.test_normality_jarque_bera()
        results["Test Residual Autocorrelation"] = self.test_residual_autocorrelation(lags=self.lags)
        results["Test Residual Independence with Spearman"] = self.test_residual_independence_with_spearman()

        details_keys = {
            "Test Mean of Residuals": "Mean of residuals",
            "Test Residuals Variance": "LM Statistic",
            "Test Normality (Jarque-Bera)": "Skewness, Kurtosis",
            "Test Residual Autocorrelation": "LB Statistic for max lag",
            "Test Residual Independence with Spearman": "Spearman's rho",
        }

        p_value_keys = {
            "Test Mean of Residuals": "p-value",
            "Test Residuals Variance": "LM p-value",
            "Test Normality (Jarque-Bera)": "p-value",
            "Test Residual Autocorrelation": "Worst p-value",
            "Test Residual Independence with Spearman": "p-value",
        }

        # Drawing the table
        header = "| {:<40} | {:<15} | {:<25} | {:<30} |".format("Test Name", "p-value", "Result", "Details")
        separator = "+" + "-" * 42 + "+" + "-" * 17 + "+" + "-" * 27 + "+" + "-" * 32 + "+"

        print(separator)
        print(header)
        print(separator)

        for test, value in results.items():
            p_value_key = p_value_keys[test]
            p_value = value[p_value_key]
            if p_value < self.alpha:
                result_text = "Reject H0"
            else:
                result_text = "Cannot reject H0"

            details_key = details_keys[test]
            if ',' in details_key:
                keys = details_key.split(', ')
                details_text = ', '.join([f"{k}: {value[k]:.2f}" for k in keys])
            else:
                details_text = f"{details_key}: {value[details_key]:.2f}"

            print("| {:<40} | {:<15.5f} | {:<25} | {:<30} |".format(test, p_value, result_text, details_text))
            print(separator)

        # Drawing PACF for residuals
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

        # Print the loss function value
        loss_val = self.loss_function()
        print(f"\nLoss Function Value: {loss_val}")

def return_dataframe(N=1000, r=0.7, random_walk=True):
    # Generowanie danych dla x1 i x2
    mean = [0, 0]
    cov = [[1, r], [r, 1]]  # współczynnik korelacji r
    x1, x2 = np.random.multivariate_normal(mean, cov, N).T

    # Tworzenie randomwalk lub szumu białego dla resid
    if random_walk:
        resid = np.cumsum(np.random.normal(0, 1, N))
    else:
        resid = np.random.normal(0, 1, N)

    df = pd.DataFrame({
        'resid': resid,
        'x1': x1,
        'x2': x2
    })

    return df


# Przykładowe użycie:
df = return_dataframe()

if __name__ == "__main__":
    """
    df = pd.DataFrame({
        'resid': [1.2, 0.5, -0.5, -1.2, 0.3, 0.4, -0.2],
        'x1': [2, 3, 1, 5, 4, 2, 3],
        'x2': [5, 6, 5, 7, 6, 5, 6]
    })
    """
    df=return_dataframe(N=1000, r=0.7,random_walk=False)


    res = ResidualAnalysis(df['resid'], exog_vars=df[['x1', 'x2']],alpha=0.05,lags=2)
    res.make_tests()

