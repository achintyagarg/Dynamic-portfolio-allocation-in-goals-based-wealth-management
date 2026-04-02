"""Utilities for portfolio optimization.

This module contains utilities for portfolio optimization.
"""

from enum import Enum


class portfolio_optimization_type(Enum):
    """Enumeration of portfolio optimization strategy types.

    This enum categorizes optimization approaches into four main families:

    **Basic Methods**: Simple, rule-based allocation strategies
        - Equal weighting, inverse volatility, random sampling

    **Convex Optimization**: Mathematical optimization with convex objectives
        - Mean-risk, risk budgeting, maximum diversification, robust methods

    **Clustering Methods**: Hierarchical and graph-based allocation
        - HRP, HERC, nested clustering, Schur complement methods

    **Ensemble Methods**: Combining multiple optimization strategies
        - Stacking optimizers for improved robustness

    Attributes
    ----------
        BASIC_EQUAL_WEIGHTED: Equal weight allocation (1/N portfolio)
        BASIC_INVERSE_VOLATILITY: Weights inversely proportional to asset volatility
        BASIC_RANDOM_DIRICHLET: Random weights sampled from Dirichlet distribution
        CONVEX_MEAN_RISK: Mean-variance and mean-risk optimization (Markowitz)
        CONVEX_RISK_BUDGETING: Risk parity and risk budgeting optimization
        CONVEX_MAXIMUM_DIVERSIFICATION: Maximize diversification ratio
        CONVEX_DISTRIBUTIONALLY_ROBUST_CVAR: Robust CVaR with distributional uncertainty
        CONVEX_BENCHMARK_TRACKING: Minimize tracking error to benchmark
        CLUSTERING_HIERARCHICAL_RISK_PARITY: HRP using hierarchical clustering
        CLUSTERING_HIERARCHICAL_EQUAL_RISK_CONTRIBUTION: HERC method
        CLUSTERING_SCHUR_COMPLEMENTARY_ALLOCATION: Schur complement-based allocation
        CLUSTERING_NESTED: Nested clustering optimization
        ENSEMBLE_STACKING: Stacked ensemble of multiple optimizers

    Examples
    --------
        >>> opt_type = portfolio_optimization_type.CONVEX_MEAN_RISK
        >>> print(opt_type.value)
        'CONVEX_MEAN_RISK'

        >>> # Check if optimization is clustering-based
        >>> opt_type = portfolio_optimization_type.CLUSTERING_HIERARCHICAL_RISK_PARITY
        >>> is_clustering = opt_type.name.startswith("CLUSTERING")
        >>> print(is_clustering)
        True

    References
    ----------
        - Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
        - Maillard, S., Roncalli, T., & Teïletche, J. (2010). The Properties of
          Equally Weighted Risk Contribution Portfolios. Journal of Portfolio Management.
        - López de Prado, M. (2016). Building Diversified Portfolios that Outperform
          Out-of-Sample. Journal of Portfolio Management.
    """

    # Basic Methods - Simple rule-based allocation
    BASIC_EQUAL_WEIGHTED = "BASIC_EQUAL_WEIGHTED"
    BASIC_INVERSE_VOLATILITY = "BASIC_INVERSE_VOLATILITY"
    BASIC_RANDOM_DIRICHLET = "BASIC_RANDOM_DIRICHLET"

    # Convex Optimization Methods
    CONVEX_MEAN_RISK = "CONVEX_MEAN_RISK"
    CONVEX_RISK_BUDGETING = "CONVEX_RISK_BUDGETING"
    CONVEX_MAXIMUM_DIVERSIFICATION = "CONVEX_MAXIMUM_DIVERSIFICATION"
    CONVEX_DISTRIBUTIONALLY_ROBUST_CVAR = "CONVEX_DISTRIBUTIONALLY_ROBUST_CVAR"
    CONVEX_BENCHMARK_TRACKING = "CONVEX_BENCHMARK_TRACKING"

    # Clustering-Based Methods
    CLUSTERING_HIERARCHICAL_RISK_PARITY = "CLUSTERING_HIERARCHICAL_RISK_PARITY"
    CLUSTERING_HIERARCHICAL_EQUAL_RISK_CONTRIBUTION = (
        "CLUSTERING_HIERARCHICAL_EQUAL_RISK_CONTRIBUTION"
    )
    CLUSTERING_SCHUR_COMPLEMENTARY_ALLOCATION = "CLUSTERING_SCHUR_COMPLEMENTARY_ALLOCATION"
    CLUSTERING_NESTED = "CLUSTERING_NESTED"

    # Ensemble Methods
    ENSEMBLE_STACKING = "ENSEMBLE_STACKING"

    @classmethod
    def get_basic_methods(cls) -> list["portfolio_optimization_type"]:
        """Return all basic optimization methods.

        Returns
        -------
            List of basic portfolio optimization types.
        """
        return [member for member in cls if member.name.startswith("BASIC")]

    @classmethod
    def get_convex_methods(cls) -> list["portfolio_optimization_type"]:
        """Return all convex optimization methods.

        Returns
        -------
            List of convex portfolio optimization types.
        """
        return [member for member in cls if member.name.startswith("CONVEX")]

    @classmethod
    def get_clustering_methods(cls) -> list["portfolio_optimization_type"]:
        """Return all clustering-based optimization methods.

        Returns
        -------
            List of clustering portfolio optimization types.
        """
        return [member for member in cls if member.name.startswith("CLUSTERING")]

    @classmethod
    def get_ensemble_methods(cls) -> list["portfolio_optimization_type"]:
        """Return all ensemble optimization methods.

        Returns
        -------
            List of ensemble portfolio optimization types.
        """
        return [member for member in cls if member.name.startswith("ENSEMBLE")]


class portfolio_optimization_feature_type(Enum):
    r"""Enumeration of portfolio optimization feature types.

    This enum categorizes optimization features, constraints, and objectives
    used in portfolio construction and risk management.

    **Objectives**: Primary goals of optimization
        - Minimize risk, maximize returns, maximize utility, maximize ratio

    **Costs**: Transaction costs and management fees
        - Transaction costs, management fees

    **Regularization**: Penalty terms for sparsity and stability
        - L1 (Lasso) regularization, L2 (Ridge) regularization

    **Weight Constraints**: Restrictions on portfolio weights
        - Weight bounds, group constraints, budget constraint, long/short thresholds

    **Portfolio Constraints**: Portfolio-level restrictions
        - Tracking error, turnover, cardinality, risk measure constraints

    **Advanced Constraints**: Sophisticated modeling features
        - Custom constraints, expected return constraints, robust optimization

    **Custom Features**: User-defined objectives and priors
        - Custom objectives, prior estimators

    Attributes
    ----------
    MINIMIZE_RISK : str
        Objective to minimize portfolio risk (variance, CVaR, etc.).
        $\\min_w \\; w^T \\Sigma w$ (variance minimization)
    MAXIMIZE_RETURNS : str
        Objective to maximize expected portfolio returns.
        $\\max_w \\; \\mu^T w$ where $\\mu$ is expected return vector
    MAXIMIZE_UTILITY : str
        Objective to maximize risk-adjusted utility function.
        $\\max_w \\; \\mu^T w - \\frac{\\lambda}{2} w^T \\Sigma w$ (mean-variance utility)
    MAXIMIZE_RATIO : str
        Objective to maximize risk-reward ratio (Sharpe, Sortino, Calmar).
        $\\max_w \\; \\frac{\\mu^T w - r_f}{\\sqrt{w^T \\Sigma w}}$ (Sharpe ratio)
    COSTS_TRANSACTION : str
        Transaction cost modeling in optimization.
        $c_{trans} = \\sum_i c_i |w_i^{new} - w_i^{old}|$ (linear cost)
    FEES_MANAGEMENT : str
        Management fee consideration in optimization.
        Annual fees reduce expected returns: $\\mu_{net} = \\mu_{gross} - fees$
    REGULARIZATION_L1 : str
        L1 (Lasso) regularization for sparse portfolios.
        $\\lambda_1 \\|w\\|_1 = \\lambda_1 \\sum_i |w_i|$ promotes sparsity
    REGULARIZATION_L2 : str
        L2 (Ridge) regularization for weight stability.
        $\\lambda_2 \\|w\\|_2^2 = \\lambda_2 \\sum_i w_i^2$ shrinks weights
    CONSTRAINTS_WEIGHT : str
        Weight constraints (box constraints).
        $w_{min,i} \\leq w_i \\leq w_{max,i}$ for each asset $i$
    CONSTRAINTS_GROUP : str
        Group-level weight constraints.
        $\\sum_{i \\in G_j} w_i \\leq \\theta_j$ for asset group $G_j$
    CONSTRAINTS_BUDGET : str
        Budget constraint (weights sum to one).
        $\\sum_i w_i = 1$ (fully invested) or $\\sum_i w_i \\leq 1$ (cash allowed)
    CONSTRAINTS_TRACKING_ERROR : str
        Tracking error constraint relative to benchmark.
        $\\sqrt{(w-w_b)^T \\Sigma (w-w_b)} \\leq TE_{max}$ where $w_b$ is benchmark
    CONSTRAINTS_TURNOVER : str
        Turnover constraint limiting rebalancing.
        $\\sum_i |w_i^{new} - w_i^{old}| \\leq \\tau_{max}$
    CONSTRAINTS_CARDINALITY : str
        Cardinality constraint (number of assets).
        $\\|w\\|_0 \\leq K$ (at most $K$ non-zero positions)
    CONSTRAINTS_CARDINALITY_GROUP : str
        Group-level cardinality constraints.
        $|\\{i \\in G_j : w_i > 0\\}| \\leq K_j$ for group $G_j$
    CONSTRAINTS_THRESHOLD_LONG : str
        Minimum long position threshold.
        $w_i \\geq w_{min,long}$ or $w_i = 0$ (minimum buy-in)
    CONSTRAINTS_THRESHOLD_SHORT : str
        Minimum short position threshold.
        $w_i \\leq -w_{min,short}$ or $w_i \\geq 0$ (minimum short size)
    CONSTRAINTS_CUSTOM : str
        User-defined custom constraints.
        $g(w) \\leq 0$ for arbitrary constraint function $g$
    CONSTRAINTS_RISK_MEASURE : str
        Risk measure constraint (CVaR, VaR, drawdown, etc.).
        $Risk(w) \\leq \\rho_{max}$ for chosen risk measure
    OBJECTIVE_CUSTOM : str
        User-defined custom objective function.
        $\\max_w \\; f(w)$ or $\\min_w \\; f(w)$ for arbitrary $f$
    ESTIMATOR_PRIOR : str
        Prior estimator feature (Black-Litterman, Bayesian).
        Incorporates prior beliefs into expected returns/covariance
    CONSTRAINTS_EXPECTED_RETURN : str
        Minimum expected return constraint.
        $\\mu^T w \\geq r_{min}$ (return target)
    UNCERTAINTY_SET_ON_EXPECTED_RETURNS : str
        Robust optimization with uncertain expected returns.
        $\\min_w \\max_{\\mu \\in \\mathcal{U}_\\mu} \\; objective(w, \\mu)$
    UNCERTAINTY_SET_ON_COVARIANCE : str
        Robust optimization with uncertain covariance matrix.
        $\\min_w \\max_{\\Sigma \\in \\mathcal{U}_\\Sigma} \\; objective(w, \\Sigma)$

    Notes
    -----
    **Objective Selection**:

    - **MINIMIZE_RISK**: Minimum variance portfolio (defensive).
      Ignores expected returns, purely risk-focused.

    - **MAXIMIZE_RETURNS**: Maximum return portfolio (aggressive).
      Ignores risk, typically requires constraints to prevent concentration.

    - **MAXIMIZE_UTILITY**: Balances risk and return via risk aversion $\\lambda$.
      $\\lambda$ large → risk-averse, $\\lambda$ small → risk-seeking.

    - **MAXIMIZE_RATIO**: Optimal Sharpe ratio (tangency portfolio).
      Most efficient risk-adjusted returns.

    **Regularization Benefits**:

    - **L1 (Lasso)**: Promotes sparse portfolios (few assets).
      Useful for reducing trading complexity and focus.

    - **L2 (Ridge)**: Stabilizes weights, reduces turnover.
      Useful for controlling estimation error and rebalancing costs.

    **Constraint Types**:

    - **Hard Constraints**: Must be satisfied exactly (budget, weight bounds).
    - **Soft Constraints**: Can be violated with penalty (via regularization).
    - **Integer Constraints**: Cardinality (combinatorial, harder to solve).

    **Robust Optimization**:

    Accounts for estimation uncertainty in inputs ($\\mu$, $\\Sigma$):
    - Worst-case optimization over uncertainty sets
    - Improves out-of-sample stability
    - Reduces sensitivity to input estimation errors

    Mathematical Formulations:

    **Mean-Variance Utility**:
    $$
    \\max_w \\; \\mu^T w - \\frac{\\lambda}{2} w^T \\Sigma w
    $$

    **Sharpe Ratio** (equivalent to tangency portfolio):
    $$
    \\max_w \\; \\frac{\\mu^T w - r_f}{\\sqrt{w^T \\Sigma w}}
    $$

    **Cardinality Constraint**:
    $$
    \\sum_i \\mathbb{1}_{\\{w_i \\neq 0\\}} \\leq K
    $$

    **Robust Mean-Variance** (worst-case expected return):
    $$
    \\max_w \\; \\min_{\\mu \\in \\mathcal{U}_\\mu} \\left( \\mu^T w - \\frac{\\lambda}{2} w^T \\Sigma w \\right)
    $$
    where $\\mathcal{U}_\\mu = \\{\\mu : \\|\\mu - \\hat{\\mu}\\| \\leq \\epsilon\\}$

    Examples
    --------
    >>> from src.models.portfolio_optimization.utils_portfolio_optimization import (
    ...     portfolio_optimization_feature_type,
    ... )
    >>> feature = portfolio_optimization_feature_type.MAXIMIZE_UTILITY
    >>> print(feature.value)
    'MAXIMIZE_UTILITY'

    >>> # Get all objective features
    >>> objectives = portfolio_optimization_feature_type.get_objectives()
    >>> print([obj.name for obj in objectives])
    ['MINIMIZE_RISK', 'MAXIMIZE_RETURNS', 'MAXIMIZE_UTILITY', 'MAXIMIZE_RATIO']

    >>> # Get all constraint features
    >>> constraints = portfolio_optimization_feature_type.get_all_constraints()
    >>> print(len(constraints))
    13

    >>> # Check if feature is a regularization method
    >>> feature = portfolio_optimization_feature_type.REGULARIZATION_L1
    >>> is_reg = feature in portfolio_optimization_feature_type.get_regularization()
    >>> print(is_reg)
    True

    References
    ----------
    - Markowitz, H. (1952). "Portfolio Selection". Journal of Finance.
    - Tibshirani, R. (1996). "Regression Shrinkage and Selection via the Lasso".
      Journal of the Royal Statistical Society: Series B.
    - Ben-Tal, A., & Nemirovski, A. (1998). "Robust Convex Optimization".
      Mathematics of Operations Research.
    - Brodie, J., Daubechies, I., De Mol, C., Giannone, D., & Loris, I. (2009).
      "Sparse and stable Markowitz portfolios". PNAS.
    - Bertsimas, D., & Shioda, R. (2009). "Algorithm for cardinality-constrained
      quadratic optimization". Computational Optimization and Applications.
    """

    # Objectives
    MINIMIZE_RISK = "MINIMIZE_RISK"
    MAXIMIZE_RETURNS = "MAXIMIZE_RETURNS"
    MAXIMIZE_UTILITY = "MAXIMIZE_UTILITY"
    MAXIMIZE_RATIO = "MAXIMIZE_RATIO"

    # Costs
    COSTS_TRANSACTION = "COSTS_TRANSACTION"
    FEES_MANAGEMENT = "FEES_MANAGEMENT"

    # Regularization
    REGULARIZATION_L1 = "REGULARIZATION_L1"
    REGULARIZATION_L2 = "REGULARIZATION_L2"

    # Weight Constraints
    CONSTRAINTS_WEIGHT = "CONSTRAINTS_WEIGHT"
    CONSTRAINTS_GROUP = "CONSTRAINTS_GROUP"
    CONSTRAINTS_BUDGET = "CONSTRAINTS_BUDGET"
    CONSTRAINTS_THRESHOLD_LONG = "CONSTRAINTS_THRESHOLD_LONG"
    CONSTRAINTS_THRESHOLD_SHORT = "CONSTRAINTS_THRESHOLD_SHORT"

    # Portfolio Constraints
    CONSTRAINTS_TRACKING_ERROR = "CONSTRAINTS_TRACKING_ERROR"
    CONSTRAINTS_TURNOVER = "CONSTRAINTS_TURNOVER"
    CONSTRAINTS_CARDINALITY = "CONSTRAINTS_CARDINALITY"
    CONSTRAINTS_CARDINALITY_GROUP = "CONSTRAINTS_CARDINALITY_GROUP"
    CONSTRAINTS_RISK_MEASURE = "CONSTRAINTS_RISK_MEASURE"
    CONSTRAINTS_EXPECTED_RETURN = "CONSTRAINTS_EXPECTED_RETURN"

    # Advanced Constraints
    CONSTRAINTS_CUSTOM = "CONSTRAINTS_CUSTOM"

    # Robust Optimization
    UNCERTAINTY_SET_ON_EXPECTED_RETURNS = "UNCERTAINTY_SET_ON_EXPECTED_RETURNS"
    UNCERTAINTY_SET_ON_COVARIANCE = "UNCERTAINTY_SET_ON_COVARIANCE"

    # Custom Features
    OBJECTIVE_CUSTOM = "OBJECTIVE_CUSTOM"
    ESTIMATOR_PRIOR = "ESTIMATOR_PRIOR"

    @classmethod
    def get_objectives(cls) -> list["portfolio_optimization_feature_type"]:
        """Return all objective function types.

        Returns
        -------
        list[portfolio_optimization_feature_type]
            List of optimization objective features.
        """
        return [
            cls.MINIMIZE_RISK,
            cls.MAXIMIZE_RETURNS,
            cls.MAXIMIZE_UTILITY,
            cls.MAXIMIZE_RATIO,
        ]

    @classmethod
    def get_costs(cls) -> list["portfolio_optimization_feature_type"]:
        """Return all cost-related features.

        Returns
        -------
        list[portfolio_optimization_feature_type]
            List of cost modeling features.
        """
        return [
            cls.COSTS_TRANSACTION,
            cls.FEES_MANAGEMENT,
        ]

    @classmethod
    def get_regularization(cls) -> list["portfolio_optimization_feature_type"]:
        """Return all regularization features.

        Returns
        -------
        list[portfolio_optimization_feature_type]
            List of regularization types (L1 and L2).
        """
        return [
            cls.REGULARIZATION_L1,
            cls.REGULARIZATION_L2,
        ]

    @classmethod
    def get_weight_constraints(cls) -> list["portfolio_optimization_feature_type"]:
        """Return all weight-related constraint features.

        Returns
        -------
        list[portfolio_optimization_feature_type]
            List of weight constraint types.
        """
        return [
            cls.CONSTRAINTS_WEIGHT,
            cls.CONSTRAINTS_GROUP,
            cls.CONSTRAINTS_BUDGET,
            cls.CONSTRAINTS_THRESHOLD_LONG,
            cls.CONSTRAINTS_THRESHOLD_SHORT,
        ]

    @classmethod
    def get_portfolio_constraints(cls) -> list["portfolio_optimization_feature_type"]:
        """Return all portfolio-level constraint features.

        Returns
        -------
        list[portfolio_optimization_feature_type]
            List of portfolio constraint types.
        """
        return [
            cls.CONSTRAINTS_TRACKING_ERROR,
            cls.CONSTRAINTS_TURNOVER,
            cls.CONSTRAINTS_CARDINALITY,
            cls.CONSTRAINTS_CARDINALITY_GROUP,
            cls.CONSTRAINTS_RISK_MEASURE,
            cls.CONSTRAINTS_EXPECTED_RETURN,
        ]

    @classmethod
    def get_all_constraints(cls) -> list["portfolio_optimization_feature_type"]:
        """Return all constraint features.

        Returns
        -------
        list[portfolio_optimization_feature_type]
            Combined list of all constraint types.
        """
        return (
            cls.get_weight_constraints()
            + cls.get_portfolio_constraints()
            + [cls.CONSTRAINTS_CUSTOM]
        )

    @classmethod
    def get_robust_optimization_features(cls) -> list["portfolio_optimization_feature_type"]:
        """Return all robust optimization features.

        These features account for uncertainty in optimization inputs.

        Returns
        -------
        list[portfolio_optimization_feature_type]
            List of robust optimization features.
        """
        return [
            cls.UNCERTAINTY_SET_ON_EXPECTED_RETURNS,
            cls.UNCERTAINTY_SET_ON_COVARIANCE,
        ]

    @classmethod
    def get_custom_features(cls) -> list["portfolio_optimization_feature_type"]:
        """Return all custom/user-defined features.

        Returns
        -------
        list[portfolio_optimization_feature_type]
            List of custom feature types.
        """
        return [
            cls.OBJECTIVE_CUSTOM,
            cls.CONSTRAINTS_CUSTOM,
            cls.ESTIMATOR_PRIOR,
        ]

    @classmethod
    def get_integer_features(cls) -> list["portfolio_optimization_feature_type"]:
        r"""Return features requiring integer/combinatorial optimization.

        These features make the optimization problem non-convex
        and computationally harder (NP-hard).

        Returns
        -------
        list[portfolio_optimization_feature_type]
            List of features requiring integer programming.

        Notes
        -----
        Cardinality constraints require mixed-integer programming (MIP):
        - Binary variables: $z_i \\in \\{0,1\\}$ indicates asset inclusion
        - Big-M constraints: $w_{min} z_i \\leq w_i \\leq w_{max} z_i$
        - Cardinality: $\\sum_i z_i \\leq K$
        """
        return [
            cls.CONSTRAINTS_CARDINALITY,
            cls.CONSTRAINTS_CARDINALITY_GROUP,
            cls.CONSTRAINTS_THRESHOLD_LONG,
            cls.CONSTRAINTS_THRESHOLD_SHORT,
        ]

    @classmethod
    def get_convex_features(cls) -> list["portfolio_optimization_feature_type"]:
        """Return features maintaining convexity of optimization problem.

        Convex problems can be solved efficiently with guaranteed
        global optimality.

        Returns
        -------
        list[portfolio_optimization_feature_type]
            List of features preserving convexity.
        """
        return [
            cls.MINIMIZE_RISK,
            cls.MAXIMIZE_RETURNS,
            cls.MAXIMIZE_UTILITY,
            cls.REGULARIZATION_L1,
            cls.REGULARIZATION_L2,
            cls.CONSTRAINTS_WEIGHT,
            cls.CONSTRAINTS_GROUP,
            cls.CONSTRAINTS_BUDGET,
            cls.CONSTRAINTS_TRACKING_ERROR,
            cls.CONSTRAINTS_TURNOVER,
            cls.CONSTRAINTS_EXPECTED_RETURN,
            cls.UNCERTAINTY_SET_ON_EXPECTED_RETURNS,
            cls.UNCERTAINTY_SET_ON_COVARIANCE,
        ]
