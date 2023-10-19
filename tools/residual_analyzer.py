import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_white, acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt

class ResidualAnalysis:
    def __init__(self, residuals, exog_vars=None, alpha=0.05):
        self.residuals = residuals
        self.exog_vars = exog_vars
        self.alpha = alpha

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

        plot_pacf(
            self.residuals,
            lags=lags,
            method="yw",
            alpha=self.alpha,
            use_vlines=True,
            title='Partial Autocorrelation',
            zero=False
        )
        plt.show()

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

if __name__ == "__main__":
    df = pd.DataFrame({
        'resid': [1.2, 0.5, -0.5, -1.2, 0.3, 0.4, -0.2],
        'x1': [2, 3, 1, 5, 4, 2, 3],
        'x2': [5, 6, 5, 7, 6, 5, 6]
    })

    res = ResidualAnalysis(df['resid'], exog_vars=df[['x1', 'x2']])
    print(res.test_mean_residuals())
    print(res.test_residuals_variance())
    print(res.test_normality_jarque_bera())
    print(res.test_residual_autocorrelation(lags=2))
    print(res.test_residual_independence_with_spearman())