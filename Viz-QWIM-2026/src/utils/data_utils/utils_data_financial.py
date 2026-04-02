"""Financial data utilities module.

This module contains utilities for handling and processing financial data,
including enumerations for expected returns estimation methods.
"""

from enum import Enum


class expected_returns_estimator_type(Enum):
    r"""Enumeration of expected returns estimator types for portfolio optimization.

    This enum defines various methods for estimating expected returns of assets,
    which are fundamental inputs for mean-variance optimization and other
    portfolio construction techniques.

    **Historical Methods**: Backward-looking estimates from past returns
        - Empirical mean, exponentially weighted mean

    **Model-Based Methods**: Forward-looking estimates from economic models
        - Equilibrium (Black-Litterman, CAPM), shrinkage estimators

    **Distribution-Based**: Estimates derived from assumed return distributions
        - Parametric distributions, scenario-based approaches

    Attributes
    ----------
    EMPIRICAL : str
        Simple arithmetic mean of historical returns.
        $\\mu_i = \\frac{1}{T}\\sum_{t=1}^{T} r_{i,t}$
    EXPONENTIALLY_WEIGHTED : str
        Exponentially weighted moving average (EWMA) of returns.
        Gives more weight to recent observations.
        $\\mu_i = \\sum_{t=1}^{T} w_t r_{i,t}$ where $w_t = \\frac{(1-\\lambda)\\lambda^{T-t}}{1-\\lambda^T}$
    EQUILIBRIUM : str
        Market equilibrium-based estimates (e.g., Black-Litterman, CAPM).
        Uses market capitalization weights and risk aversion to derive implied returns.
        $\\Pi = \\delta \\Sigma w_{mkt}$ (reverse optimization)
    SHRINKAGE : str
        Shrinkage estimator combining sample mean with prior/target.
        $\\mu_{shrink} = \\alpha \\mu_{sample} + (1-\\alpha) \\mu_{target}$
        Reduces estimation error, especially for large portfolios.
    FROM_DISTRIBUTION : str
        Expected returns derived from assumed parametric distribution.
        Uses fitted distribution parameters (e.g., normal, Student-t, skewed distributions).

    Notes
    -----
    The choice of expected returns estimator significantly impacts portfolio
    performance:

    - **EMPIRICAL**: Simple but sensitive to outliers and non-stationarity.
      Works best with long, stable return histories.

    - **EXPONENTIALLY_WEIGHTED**: Adapts to regime changes faster than empirical.
      Decay parameter $\\lambda$ controls memory (typical values: 0.94-0.97 for daily data).

    - **EQUILIBRIUM**: Incorporates market-wide information and economic theory.
      Black-Litterman allows blending equilibrium with subjective views.

    - **SHRINKAGE**: Reduces estimation error by pulling extreme estimates toward
      a sensible target (e.g., grand mean, zero, equal returns).

    - **FROM_DISTRIBUTION**: Useful when parametric assumptions are justified
      or when using Monte Carlo simulation.

    Mathematical Formulations:

    **Empirical Mean**:
    $$
    \\mu_i = \\frac{1}{T}\\sum_{t=1}^{T} r_{i,t}
    $$

    **EWMA** (with decay $\\lambda$):
    $$
    \\mu_i(t) = (1-\\lambda)r_{i,t} + \\lambda \\mu_i(t-1)
    $$

    **Equilibrium (Reverse Optimization)**:
    $$
    \\Pi = \\delta \\Sigma w_{mkt}
    $$
    where $\\delta$ is risk aversion, $\\Sigma$ is covariance, $w_{mkt}$ is market cap weights.

    **Shrinkage (Ledoit-Wolf style)**:
    $$
    \\mu^* = \\alpha \\mu_{sample} + (1-\\alpha) \\mu_{target}
    $$

    Examples
    --------
    >>> from src.data_utils.utils_data_financial import expected_returns_estimator_type
    >>> estimator = expected_returns_estimator_type.EMPIRICAL
    >>> print(estimator.value)
    'EMPIRICAL'

    >>> # Get all historical methods
    >>> historical = expected_returns_estimator_type.get_historical_methods()
    >>> print([m.name for m in historical])
    ['EMPIRICAL', 'EXPONENTIALLY_WEIGHTED']

    >>> # Check if estimator is model-based
    >>> estimator = expected_returns_estimator_type.EQUILIBRIUM
    >>> is_model = estimator in expected_returns_estimator_type.get_model_based_methods()
    >>> print(is_model)
    True

    References
    ----------
    - Black, F., & Litterman, R. (1992). "Global Portfolio Optimization".
      Financial Analysts Journal.
    - Ledoit, O., & Wolf, M. (2003). "Improved estimation of the covariance
      matrix of stock returns with an application to portfolio selection".
      Journal of Empirical Finance.
    - Meucci, A. (2005). "Risk and Asset Allocation". Springer.
    - RiskMetrics (1996). "Technical Document". J.P. Morgan.
    """

    # Historical Methods
    EMPIRICAL = "EMPIRICAL"
    EXPONENTIALLY_WEIGHTED = "EXPONENTIALLY_WEIGHTED"

    # Model-Based Methods
    EQUILIBRIUM = "EQUILIBRIUM"
    SHRINKAGE = "SHRINKAGE"

    # Distribution-Based Methods
    FROM_DISTRIBUTION = "FROM_DISTRIBUTION"

    @classmethod
    def get_historical_methods(cls) -> list["expected_returns_estimator_type"]:
        """Return all historical (backward-looking) estimation methods.

        These methods use only past return data without additional modeling
        assumptions or economic theory.

        Returns
        -------
        list[expected_returns_estimator_type]
            List of historical expected returns estimators.
        """
        return [
            cls.EMPIRICAL,
            cls.EXPONENTIALLY_WEIGHTED,
        ]

    @classmethod
    def get_model_based_methods(cls) -> list["expected_returns_estimator_type"]:
        """Return all model-based (forward-looking) estimation methods.

        These methods incorporate economic theory, equilibrium assumptions,
        or statistical regularization.

        Returns
        -------
        list[expected_returns_estimator_type]
            List of model-based expected returns estimators.
        """
        return [
            cls.EQUILIBRIUM,
            cls.SHRINKAGE,
        ]

    @classmethod
    def get_distribution_based_methods(cls) -> list["expected_returns_estimator_type"]:
        """Return all distribution-based estimation methods.

        These methods derive expected returns from assumed parametric
        return distributions.

        Returns
        -------
        list[expected_returns_estimator_type]
            List of distribution-based expected returns estimators.
        """
        return [
            cls.FROM_DISTRIBUTION,
        ]

    @classmethod
    def get_robust_methods(cls) -> list["expected_returns_estimator_type"]:
        """Return robust estimation methods that reduce estimation error.

        These methods are less sensitive to outliers or extreme sample
        estimates, making them more stable for optimization.

        Returns
        -------
        list[expected_returns_estimator_type]
            List of robust expected returns estimators.

        Notes
        -----
        Robust methods include:
        - EXPONENTIALLY_WEIGHTED: Dampens impact of old outliers
        - SHRINKAGE: Pulls extreme estimates toward sensible target
        - EQUILIBRIUM: Grounded in market-wide equilibrium theory
        """
        return [
            cls.EXPONENTIALLY_WEIGHTED,
            cls.SHRINKAGE,
            cls.EQUILIBRIUM,
        ]


class prior_estimator_type(Enum):
    r"""Enumeration of prior estimator types for Bayesian portfolio optimization.

    This enum defines various methods for specifying prior beliefs about expected
    returns in Bayesian frameworks, particularly useful for Black-Litterman models,
    stress testing, and scenario analysis.

    **Classical Priors**: Data-driven prior distributions
        - Empirical (historical mean as prior)

    **Equilibrium Priors**: Market-implied expected returns
        - Black-Litterman (reverse optimization from market weights)

    **Factor-Based Priors**: Returns derived from factor models
        - Factor models (Fama-French, CAPM, APT)

    **Scenario-Based Priors**: Synthetic data for stress testing
        - Stress test scenarios, factor stress tests

    **Probabilistic Priors**: Combining multiple probability assessments
        - Entropy pooling (Meucci), opinion pooling (aggregating views)

    Attributes
    ----------
    EMPIRICAL : str
        Historical mean returns used as prior beliefs.
        Simple but assumes past returns predict future performance.
        $\\mu_{prior} = \\frac{1}{T}\\sum_{t=1}^{T} r_t$
    BLACK_LITTERMAN : str
        Market equilibrium returns from reverse optimization.
        Blends equilibrium with investor views using Bayesian updating.
        $\\Pi = \\delta \\Sigma w_{mkt}$ (equilibrium prior)
        $E[R] = [(\\tau\\Sigma)^{-1} + P^T\\Omega^{-1}P]^{-1}[(\\tau\\Sigma)^{-1}\\Pi + P^T\\Omega^{-1}Q]$
    FACTOR_MODEL : str
        Expected returns derived from factor exposures and factor premia.
        $E[R_i] = \\alpha_i + \\sum_{k=1}^{K} \\beta_{ik} \\lambda_k$
        where $\\beta$ are factor loadings and $\\lambda$ are risk premia.
    SYNTHETIC_DATA_STRESS_TEST : str
        Synthetic return scenarios for stress testing.
        Generates adverse/favorable scenarios to test portfolio resilience.
        Useful for tail risk assessment and regulatory stress tests.
    SYNTHETIC_DATA_FACTOR_STRESS_TEST : str
        Factor-based synthetic scenarios for stress testing.
        Stresses specific factor exposures (e.g., equity beta, credit spread).
        Allows targeted "what-if" analysis on factor risk drivers.
    ENTROPY_POOLING : str
        Meucci's entropy pooling framework for incorporating views.
        Minimizes relative entropy subject to view constraints.
        $\\min_{p} \\sum_j p_j \\log\\frac{p_j}{p_j^{prior}}$ subject to $Fp = f$ (views)
        Non-parametric, handles complex views naturally.
    OPINION_POOLING : str
        Aggregation of multiple expert opinions into consensus prior.
        Combines forecasts from different analysts/models.
        Methods: linear pooling, logarithmic pooling, Bayesian Model Averaging.

    Notes
    -----
    Prior selection in Bayesian portfolio optimization:

    - **EMPIRICAL**: Simplest prior, but ignores market structure and theory.
      May work well in stable markets with long histories.

    - **BLACK_LITTERMAN**: Gold standard for institutional investors.
      Equilibrium prior is neutral, then adjusted with investor views.
      Confidence in views controlled via $\\Omega$ (uncertainty matrix).

    - **FACTOR_MODEL**: Grounded in asset pricing theory.
      Works well when factor structure is stable and well-estimated.
      Examples: Fama-French 5-factor, Carhart 4-factor.

    - **SYNTHETIC_DATA_STRESS_TEST**: Critical for risk management.
      Tests portfolio under extreme but plausible scenarios.
      Required for regulatory compliance (CCAR, EBA stress tests).

    - **SYNTHETIC_DATA_FACTOR_STRESS_TEST**: More targeted than general stress tests.
      Isolates impact of specific risk factors (credit, equity, rates).

    - **ENTROPY_POOLING**: Most flexible Bayesian approach.
      Can incorporate inequality views, tail constraints, confidence bands.
      Computationally intensive but theoretically elegant.

    - **OPINION_POOLING**: Useful when multiple forecasters available.
      Reduces individual forecast error through diversification.
      Weight schemes: equal, performance-based, Bayesian Model Averaging.

    Mathematical Formulations:

    **Black-Litterman Posterior**:
    $$
    E[R] = \\left[(\\tau\\Sigma)^{-1} + P^T\\Omega^{-1}P\\right]^{-1}
           \\left[(\\tau\\Sigma)^{-1}\\Pi + P^T\\Omega^{-1}Q\\right]
    $$
    where:
    - $\\Pi$ = equilibrium returns
    - $P$ = view matrix (links assets to views)
    - $Q$ = view values
    - $\\Omega$ = uncertainty in views
    - $\\tau$ = scaling factor (typically 0.01-0.05)

    **Factor Model Prior**:
    $$
    E[R_i] = r_f + \\sum_{k=1}^{K} \\beta_{ik} (E[F_k] - r_f)
    $$

    **Entropy Pooling Objective**:
    $$
    \\min_{p} \\sum_{j=1}^{J} p_j \\log\\frac{p_j}{p_j^{prior}}
    $$
    subject to:
    $$
    \\sum_j p_j = 1, \\quad Fp = f \\quad (\\text{views}), \\quad p_j \\geq 0
    $$

    Examples
    --------
    >>> from src.data_utils.utils_data_financial import prior_estimator_type
    >>> prior = prior_estimator_type.BLACK_LITTERMAN
    >>> print(prior.value)
    'BLACK_LITTERMAN'

    >>> # Get all scenario-based priors for stress testing
    >>> scenario_priors = prior_estimator_type.get_scenario_based()
    >>> print([p.name for p in scenario_priors])
    ['SYNTHETIC_DATA_STRESS_TEST', 'SYNTHETIC_DATA_FACTOR_STRESS_TEST']

    >>> # Check if prior is Bayesian
    >>> prior = prior_estimator_type.ENTROPY_POOLING
    >>> is_bayesian = prior in prior_estimator_type.get_bayesian_methods()
    >>> print(is_bayesian)
    True

    References
    ----------
    - Black, F., & Litterman, R. (1992). "Global Portfolio Optimization".
      Financial Analysts Journal.
    - Meucci, A. (2008). "Fully Flexible Views: Theory and Practice".
      Risk, 21(10), 97-102.
    - Fama, E. F., & French, K. R. (2015). "A five-factor asset pricing model".
      Journal of Financial Economics.
    - He, G., & Litterman, R. (1999). "The Intuition Behind Black-Litterman
      Model Portfolios". Goldman Sachs Asset Management.
    """

    # Classical Priors
    EMPIRICAL = "EMPIRICAL"

    # Equilibrium Priors
    BLACK_LITTERMAN = "BLACK_LITTERMAN"

    # Factor-Based Priors
    FACTOR_MODEL = "FACTOR_MODEL"

    # Scenario-Based Priors
    SYNTHETIC_DATA_STRESS_TEST = "SYNTHETIC_DATA_STRESS_TEST"
    SYNTHETIC_DATA_FACTOR_STRESS_TEST = "SYNTHETIC_DATA_FACTOR_STRESS_TEST"

    # Probabilistic Priors
    ENTROPY_POOLING = "ENTROPY_POOLING"
    OPINION_POOLING = "OPINION_POOLING"

    @classmethod
    def get_data_driven(cls) -> list["prior_estimator_type"]:
        """Return all data-driven prior estimators.

        These priors are derived directly from historical data without
        additional modeling or expert input.

        Returns
        -------
        list[prior_estimator_type]
            List of data-driven prior estimator types.
        """
        return [
            cls.EMPIRICAL,
        ]

    @classmethod
    def get_equilibrium_based(cls) -> list["prior_estimator_type"]:
        """Return all equilibrium-based prior estimators.

        These priors use market equilibrium theory to derive expected returns
        from observable market data (prices, weights).

        Returns
        -------
        list[prior_estimator_type]
            List of equilibrium-based prior estimator types.
        """
        return [
            cls.BLACK_LITTERMAN,
        ]

    @classmethod
    def get_factor_based(cls) -> list["prior_estimator_type"]:
        """Return all factor model-based prior estimators.

        These priors derive expected returns from factor exposures
        and factor risk premia.

        Returns
        -------
        list[prior_estimator_type]
            List of factor-based prior estimator types.
        """
        return [
            cls.FACTOR_MODEL,
        ]

    @classmethod
    def get_scenario_based(cls) -> list["prior_estimator_type"]:
        """Return all scenario-based prior estimators for stress testing.

        These priors use synthetic data or hypothetical scenarios
        to assess portfolio resilience under stress conditions.

        Returns
        -------
        list[prior_estimator_type]
            List of scenario-based prior estimator types.
        """
        return [
            cls.SYNTHETIC_DATA_STRESS_TEST,
            cls.SYNTHETIC_DATA_FACTOR_STRESS_TEST,
        ]

    @classmethod
    def get_probabilistic(cls) -> list["prior_estimator_type"]:
        """Return all probabilistic prior estimators.

        These priors combine multiple probability assessments or
        incorporate views using Bayesian/information-theoretic methods.

        Returns
        -------
        list[prior_estimator_type]
            List of probabilistic prior estimator types.
        """
        return [
            cls.ENTROPY_POOLING,
            cls.OPINION_POOLING,
        ]

    @classmethod
    def get_bayesian_methods(cls) -> list["prior_estimator_type"]:
        """Return all Bayesian prior estimation methods.

        These methods explicitly use Bayesian updating to combine
        prior beliefs with data or views.

        Returns
        -------
        list[prior_estimator_type]
            List of Bayesian prior estimator types.

        Notes
        -----
        Bayesian methods include:
        - BLACK_LITTERMAN: Bayesian updating of equilibrium with views
        - ENTROPY_POOLING: Minimizes relative entropy (Bayesian twist)
        - OPINION_POOLING: Can use Bayesian Model Averaging
        """
        return [
            cls.BLACK_LITTERMAN,
            cls.ENTROPY_POOLING,
            cls.OPINION_POOLING,
        ]

    @classmethod
    def get_view_incorporation_methods(cls) -> list["prior_estimator_type"]:
        """Return prior methods that can incorporate subjective views.

        These methods allow portfolio managers to express opinions
        and blend them with statistical/equilibrium priors.

        Returns
        -------
        list[prior_estimator_type]
            List of prior estimators supporting view incorporation.
        """
        return [
            cls.BLACK_LITTERMAN,
            cls.ENTROPY_POOLING,
            cls.OPINION_POOLING,
        ]


class distribution_estimator_type(Enum):
    r"""Enumeration of distribution estimator types for modeling asset returns.

    This enum defines various parametric distributions and copula structures
    used to model the joint distribution of asset returns, essential for
    portfolio optimization, risk management, and Monte Carlo simulation.

    **Univariate Distributions**: Single-asset return distributions
        - Gaussian (normal), Student-t, Johnson SU, Normal Inverse Gaussian

    **Bivariate Copulas**: Two-asset dependence structures
        - Gaussian, Student-t, Archimedean copulas (Clayton, Gumbel, Joe), Independent

    **Multivariate Vine Copulas**: High-dimensional dependence modeling
        - Regular vine (R-vine), Centered vine (C-vine), Clustered vine, Conditional sampling

    Attributes
    ----------
    UNIVARIATE_GAUSSIAN : str
        Normal distribution with mean and variance parameters.
        $f(x) = \\frac{1}{\\sqrt{2\\pi\\sigma^2}} e^{-\\frac{(x-\\mu)^2}{2\\sigma^2}}$
        Simple but assumes symmetry and light tails.
    UNIVARIATE_STUDENT_T : str
        Student's t-distribution with degrees of freedom parameter.
        $f(x) = \\frac{\\Gamma(\\frac{\\nu+1}{2})}{\\sqrt{\\nu\\pi}\\Gamma(\\frac{\\nu}{2})} \\left(1+\\frac{x^2}{\\nu}\\right)^{-\\frac{\\nu+1}{2}}$
        Captures fat tails, models extreme events better than Gaussian.
    UNIVARIATE_JOHNSON_SU : str
        Johnson SU (unbounded support) distribution.
        Flexible four-parameter family matching any skewness and kurtosis.
        $Z = \\gamma + \\delta \\sinh^{-1}\\left(\\frac{X-\\xi}{\\lambda}\\right)$ where $Z \\sim N(0,1)$
    UNIVARIATE_NORMAL_INVERSE_GAUSSIAN : str
        Normal Inverse Gaussian (NIG) distribution.
        $f(x) = \\frac{\\alpha\\delta K_1(\\alpha\\sqrt{\\delta^2+(x-\\mu)^2})}{\\pi\\sqrt{\\delta^2+(x-\\mu)^2}} e^{\\delta\\gamma+\\beta(x-\\mu)}$
        Semi-heavy tails, captures asymmetry, popular in finance.
    COPULA_BIVARIATE_GAUSSIAN : str
        Gaussian copula for two assets.
        $C(u_1,u_2) = \\Phi_\\rho(\\Phi^{-1}(u_1), \\Phi^{-1}(u_2))$
        Symmetric dependence, no tail dependence.
    COPULA_BIVARIATE_STUDENT_T : str
        Student-t copula for two assets.
        Exhibits symmetric tail dependence, captures joint extremes.
        $C(u_1,u_2) = t_{\\nu,\\rho}(t_\\nu^{-1}(u_1), t_\\nu^{-1}(u_2))$
    COPULA_BIVARIATE_CLAYTON : str
        Clayton copula (Archimedean family).
        $C(u_1,u_2) = (u_1^{-\\theta} + u_2^{-\\theta} - 1)^{-1/\\theta}$
        Lower tail dependence, asymmetric, models joint crashes.
    COPULA_BIVARIATE_GUMBEL : str
        Gumbel copula (Archimedean family).
        $C(u_1,u_2) = \\exp\\left(-[(- \\ln u_1)^\\theta + (- \\ln u_2)^\\theta]^{1/\\theta}\\right)$
        Upper tail dependence, asymmetric, models joint booms.
    COPULA_BIVARIATE_JOE : str
        Joe copula (Archimedean family).
        Upper tail dependence, alternative to Gumbel with different tail behavior.
    COPULA_BIVARIATE_INDEPENDENT : str
        Independence copula (product copula).
        $C(u_1,u_2) = u_1 \\cdot u_2$
        No dependence between assets, benchmark copula.
    COPULA_MULTIVARIATE_VINE_REGULAR : str
        Regular vine (R-vine) copula for high dimensions.
        Most flexible vine structure, pair-copulas arranged in tree sequence.
        No specific ordering constraint, maximizes modeling flexibility.
    COPULA_MULTIVARIATE_VINE_CENTERED : str
        Centered vine (C-vine) copula.
        Tree structure with central/root variables.
        Useful when one asset (e.g., market index) drives others.
    COPULA_MULTIVARIATE_VINE_CLUSTERED : str
        Clustered vine copula structure.
        Groups assets into clusters with intra-cluster and inter-cluster copulas.
        Incorporates hierarchical dependence patterns.
    COPULA_MULTIVARIATE_VINE_CONDITIONAL_SAMPLING : str
        Vine copula with conditional sampling capability.
        Enables sequential conditional sampling for Monte Carlo simulation.
        Essential for scenario generation and stress testing.

    Notes
    -----
    **Univariate Distribution Selection**:

    - **GAUSSIAN**: Use when returns are symmetric, no extreme outliers.
      Computationally simple but often inadequate for financial data.

    - **STUDENT_T**: Captures fat tails (leptokurtosis) common in financial returns.
      Degrees of freedom $\\nu$ controls tail heaviness ($\\nu < 10$ for heavy tails).

    - **JOHNSON_SU**: Flexible distribution matching target skewness and kurtosis.
      Four parameters allow precise moment matching.

    - **NORMAL_INVERSE_GAUSSIAN**: Semi-heavy tails with asymmetry.
      Popular in derivatives pricing and risk management.

    **Copula Selection for Dependence**:

    - **GAUSSIAN/STUDENT_T COPULAS**:
      - Gaussian: No tail dependence, symmetric
      - Student-t: Symmetric tail dependence, models joint crashes and rallies

    - **ARCHIMEDEAN COPULAS (Clayton, Gumbel, Joe)**:
      - Clayton: Lower tail dependence → joint downside risk
      - Gumbel/Joe: Upper tail dependence → joint upside movements
      - Asymmetric, parsimonious (one parameter)

    - **VINE COPULAS**:
      - Decompose multivariate dependence into bivariate building blocks
      - R-vine: Maximum flexibility, no structure imposed
      - C-vine: Star structure, useful with dominant variable
      - Clustered: Hierarchical dependence with asset groups

    Mathematical Formulations:

    **Sklar's Theorem** (copula decomposition):
    $$
    F(x_1,...,x_n) = C(F_1(x_1),...,F_n(x_n))
    $$
    where $C$ is the copula and $F_i$ are marginal distributions.

    **Tail Dependence Coefficient**:
    $$
    \\lambda_L = \\lim_{u \\to 0^+} P(U_2 \\leq u | U_1 \\leq u) \\quad \\text{(lower)}
    $$
    $$
    \\lambda_U = \\lim_{u \\to 1^-} P(U_2 > u | U_1 > u) \\quad \\text{(upper)}
    $$

    **Vine Copula Density** (R-vine example):
    $$
    f(x_1,...,x_n) = \\prod_{i=1}^n f_i(x_i) \\cdot \\prod_{j=1}^{n-1} \\prod_{e \\in E_j} c_{e|D(e)}
    $$
    where $c_{e|D(e)}$ are conditional pair-copula densities.

    Examples
    --------
    >>> from src.data_utils.utils_data_financial import distribution_estimator_type
    >>> dist_type = distribution_estimator_type.UNIVARIATE_STUDENT_T
    >>> print(dist_type.value)
    'UNIVARIATE_STUDENT_T'

    >>> # Get all univariate distributions
    >>> univariate = distribution_estimator_type.get_univariate()
    >>> print([d.name for d in univariate])
    ['UNIVARIATE_GAUSSIAN', 'UNIVARIATE_STUDENT_T', 'UNIVARIATE_JOHNSON_SU', 'UNIVARIATE_NORMAL_INVERSE_GAUSSIAN']

    >>> # Get copulas with tail dependence
    >>> tail_dep = distribution_estimator_type.get_tail_dependent_copulas()
    >>> print([c.name for c in tail_dep])
    ['COPULA_BIVARIATE_STUDENT_T', 'COPULA_BIVARIATE_CLAYTON', 'COPULA_BIVARIATE_GUMBEL', 'COPULA_BIVARIATE_JOE']

    References
    ----------
    - Sklar, A. (1959). "Fonctions de répartition à n dimensions et leurs marges".
      Publications de l'Institut de Statistique de l'Université de Paris.
    - Nelsen, R. B. (2006). "An Introduction to Copulas". Springer.
    - Aas, K., Czado, C., Frigessi, A., & Bakken, H. (2009). "Pair-copula
      constructions of multiple dependence". Insurance: Mathematics and Economics.
    - Barndorff-Nielsen, O. E. (1997). "Normal Inverse Gaussian Distributions
      and Stochastic Volatility Modelling". Scandinavian Journal of Statistics.
    - Johnson, N. L. (1949). "Systems of frequency curves generated by methods
      of translation". Biometrika.
    """

    # Univariate Distributions
    UNIVARIATE_GAUSSIAN = "UNIVARIATE_GAUSSIAN"
    UNIVARIATE_STUDENT_T = "UNIVARIATE_STUDENT_T"
    UNIVARIATE_JOHNSON_SU = "UNIVARIATE_JOHNSON_SU"
    UNIVARIATE_NORMAL_INVERSE_GAUSSIAN = "UNIVARIATE_NORMAL_INVERSE_GAUSSIAN"

    # Bivariate Copulas
    COPULA_BIVARIATE_GAUSSIAN = "COPULA_BIVARIATE_GAUSSIAN"
    COPULA_BIVARIATE_STUDENT_T = "COPULA_BIVARIATE_STUDENT_T"
    COPULA_BIVARIATE_CLAYTON = "COPULA_BIVARIATE_CLAYTON"
    COPULA_BIVARIATE_GUMBEL = "COPULA_BIVARIATE_GUMBEL"
    COPULA_BIVARIATE_JOE = "COPULA_BIVARIATE_JOE"
    COPULA_BIVARIATE_INDEPENDENT = "COPULA_BIVARIATE_INDEPENDENT"

    # Multivariate Vine Copulas
    COPULA_MULTIVARIATE_VINE_REGULAR = "COPULA_MULTIVARIATE_VINE_REGULAR"
    COPULA_MULTIVARIATE_VINE_CENTERED = "COPULA_MULTIVARIATE_VINE_CENTERED"
    COPULA_MULTIVARIATE_VINE_CLUSTERED = "COPULA_MULTIVARIATE_VINE_CLUSTERED"
    COPULA_MULTIVARIATE_VINE_CONDITIONAL_SAMPLING = "COPULA_MULTIVARIATE_VINE_CONDITIONAL_SAMPLING"

    @classmethod
    def get_univariate(cls) -> list["distribution_estimator_type"]:
        """Return all univariate distribution estimators.

        Returns
        -------
        list[distribution_estimator_type]
            List of univariate distribution types for single-asset modeling.
        """
        return [
            cls.UNIVARIATE_GAUSSIAN,
            cls.UNIVARIATE_STUDENT_T,
            cls.UNIVARIATE_JOHNSON_SU,
            cls.UNIVARIATE_NORMAL_INVERSE_GAUSSIAN,
        ]

    @classmethod
    def get_bivariate_copulas(cls) -> list["distribution_estimator_type"]:
        """Return all bivariate copula estimators.

        Returns
        -------
        list[distribution_estimator_type]
            List of bivariate copula types for two-asset dependence modeling.
        """
        return [
            cls.COPULA_BIVARIATE_GAUSSIAN,
            cls.COPULA_BIVARIATE_STUDENT_T,
            cls.COPULA_BIVARIATE_CLAYTON,
            cls.COPULA_BIVARIATE_GUMBEL,
            cls.COPULA_BIVARIATE_JOE,
            cls.COPULA_BIVARIATE_INDEPENDENT,
        ]

    @classmethod
    def get_multivariate_copulas(cls) -> list["distribution_estimator_type"]:
        """Return all multivariate vine copula estimators.

        Returns
        -------
        list[distribution_estimator_type]
            List of multivariate vine copula types for high-dimensional modeling.
        """
        return [
            cls.COPULA_MULTIVARIATE_VINE_REGULAR,
            cls.COPULA_MULTIVARIATE_VINE_CENTERED,
            cls.COPULA_MULTIVARIATE_VINE_CLUSTERED,
            cls.COPULA_MULTIVARIATE_VINE_CONDITIONAL_SAMPLING,
        ]

    @classmethod
    def get_all_copulas(cls) -> list["distribution_estimator_type"]:
        """Return all copula estimators (bivariate and multivariate).

        Returns
        -------
        list[distribution_estimator_type]
            List of all copula types.
        """
        return cls.get_bivariate_copulas() + cls.get_multivariate_copulas()

    @classmethod
    def get_fat_tailed_distributions(cls) -> list["distribution_estimator_type"]:
        """Return distributions with heavy tails (excess kurtosis).

        These distributions better capture extreme events and tail risk
        common in financial returns.

        Returns
        -------
        list[distribution_estimator_type]
            List of fat-tailed distribution types.
        """
        return [
            cls.UNIVARIATE_STUDENT_T,
            cls.UNIVARIATE_NORMAL_INVERSE_GAUSSIAN,
        ]

    @classmethod
    def get_flexible_distributions(cls) -> list["distribution_estimator_type"]:
        """Return flexible distributions that can match various moments.

        These distributions have enough parameters to fit skewness,
        kurtosis, and other higher moments.

        Returns
        -------
        list[distribution_estimator_type]
            List of flexible distribution types.
        """
        return [
            cls.UNIVARIATE_JOHNSON_SU,
            cls.UNIVARIATE_NORMAL_INVERSE_GAUSSIAN,
        ]

    @classmethod
    def get_tail_dependent_copulas(cls) -> list["distribution_estimator_type"]:
        """Return copulas exhibiting tail dependence.

        Tail dependence measures the probability of joint extreme events.
        Critical for modeling systemic risk and portfolio crashes.

        Returns
        -------
        list[distribution_estimator_type]
            List of copulas with tail dependence (lower and/or upper).

        Notes
        -----
        - Student-t: Symmetric tail dependence (both upper and lower)
        - Clayton: Lower tail dependence (joint crashes)
        - Gumbel: Upper tail dependence (joint rallies)
        - Joe: Upper tail dependence (alternative parameterization)
        """
        return [
            cls.COPULA_BIVARIATE_STUDENT_T,
            cls.COPULA_BIVARIATE_CLAYTON,
            cls.COPULA_BIVARIATE_GUMBEL,
            cls.COPULA_BIVARIATE_JOE,
        ]

    @classmethod
    def get_archimedean_copulas(cls) -> list["distribution_estimator_type"]:
        """Return Archimedean copula family members.

        Archimedean copulas are characterized by a generator function
        and typically have one parameter, making them parsimonious.

        Returns
        -------
        list[distribution_estimator_type]
            List of Archimedean bivariate copula types.
        """
        return [
            cls.COPULA_BIVARIATE_CLAYTON,
            cls.COPULA_BIVARIATE_GUMBEL,
            cls.COPULA_BIVARIATE_JOE,
        ]

    @classmethod
    def get_simulation_ready(cls) -> list["distribution_estimator_type"]:
        """Return distributions/copulas suitable for Monte Carlo simulation.

        These have efficient sampling algorithms implemented.

        Returns
        -------
        list[distribution_estimator_type]
            List of types with efficient sampling methods.
        """
        return [
            cls.UNIVARIATE_GAUSSIAN,
            cls.UNIVARIATE_STUDENT_T,
            cls.COPULA_BIVARIATE_GAUSSIAN,
            cls.COPULA_BIVARIATE_STUDENT_T,
            cls.COPULA_MULTIVARIATE_VINE_CONDITIONAL_SAMPLING,
        ]
