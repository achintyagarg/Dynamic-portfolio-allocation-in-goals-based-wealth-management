"""
Utilities for risks
===============

This module contains utilities for risks including risk measure enumerations
and helper functions for risk calculations.
"""

from aenum import Enum


class risk_measure_type(Enum):
    """Enumeration of risk measure types used in portfolio optimization and risk management.
    
    This enum categorizes risk measures into five main families based on their
    mathematical properties and the aspects of return distributions they capture:
    
    **Variance-Based Measures**: Traditional dispersion measures
        - Variance, semi-variance, mean absolute deviation
    
    **Value-at-Risk (VaR) Family**: Quantile-based downside risk measures
        - VaR, CVaR (Expected Shortfall), EVaR, Worst Realization
    
    **Drawdown Measures**: Peak-to-trough decline metrics
        - Maximum drawdown, average drawdown, CDaR, DaR, EDaR, Ulcer Index
    
    **Higher Moment Measures**: Skewness and kurtosis-based measures
        - Fourth central moment, fourth lower partial moment, skew, kurtosis
    
    **Other Risk Measures**: Specialized risk metrics
        - Gini mean difference, entropic risk measure
    
    Attributes:
        VARIANCE: Standard variance of returns ($\\sigma^2$)
        SEMI_VARIANCE: Variance of returns below the mean (downside variance)
        MEAN_ABSOLUTE_DEVIATION: Average absolute deviation from mean (MAD)
        FIRST_LOWER_PARTIAL_MOMENT: Expected shortfall below target (LPM1)
        CONDITIONAL_VALUE_AT_RISK: Expected loss beyond VaR (CVaR/ES)
        ENTROPIC_VALUE_AT_RISK: Coherent risk measure using exponential utility (EVaR)
        WORST_REALIZATION: Maximum observed loss (worst-case scenario)
        CONDITIONAL_DRAWDOWN_AT_RISK: Expected drawdown beyond DaR threshold (CDaR)
        MAXIMUM_DRAWDOWN: Largest peak-to-trough decline (MDD)
        AVERAGE_DRAWDOWN: Mean of all drawdowns over the period
        ENTROPIC_DRAWDOWN_AT_RISK: Entropic version of drawdown risk (EDaR)
        ULCER_INDEX: Root mean square of percentage drawdowns
        GINI_MEAN_DIFFERENCE: Expected absolute difference between random pairs
        VALUE_AT_RISK: Loss threshold at given confidence level (VaR)
        DRAWDOWN_AT_RISK: Drawdown threshold at given confidence level (DaR)
        ENTROPIC_RISK_MEASURE: Convex risk measure based on relative entropy
        FOURTH_CENTRAL_MOMENT: Measure of tail heaviness (kurtosis-related)
        FOURTH_LOWER_PARTIAL_MOMENT: Fourth moment of returns below target (LPM4)
        SKEW: Asymmetry of the return distribution (third standardized moment)
        KURTOSIS: Tail heaviness of the return distribution (fourth standardized moment)
    
    Notes
    -----
    Risk measures can be categorized by their mathematical properties:
    
    - **Coherent Risk Measures**: CVaR, EVaR satisfy coherence axioms
      (monotonicity, sub-additivity, positive homogeneity, translation invariance)
    
    - **Convex Risk Measures**: Relax sub-additivity to convexity
    
    - **Spectral Risk Measures**: Weighted average of quantiles
    
    Mathematical Definitions:
    
    $$
    \\text{VaR}_\\alpha = \\inf\\{x : P(X \\leq x) \\geq \\alpha\\}
    $$
    
    $$
    \\text{CVaR}_\\alpha = \\mathbb{E}[X | X \\leq \\text{VaR}_\\alpha]
    $$
    
    $$
    \\text{MDD} = \\max_{t \\in [0,T]} \\left( \\max_{s \\in [0,t]} S_s - S_t \\right)
    $$
    
    Examples
    --------
    >>> risk_type = risk_measure_type.CONDITIONAL_VALUE_AT_RISK
    >>> print(risk_type.value)
    'CONDITIONAL_VALUE_AT_RISK'
    
    >>> # Check if risk measure is drawdown-based
    >>> risk_type = risk_measure_type.MAXIMUM_DRAWDOWN
    >>> is_drawdown = risk_type in risk_measure_type.get_drawdown_measures()
    >>> print(is_drawdown)
    True
    
    >>> # Get all coherent risk measures for optimization
    >>> coherent_measures = risk_measure_type.get_coherent_measures()
    >>> print([m.name for m in coherent_measures])
    ['CONDITIONAL_VALUE_AT_RISK', 'ENTROPIC_VALUE_AT_RISK', ...]
    
    References
    ----------
    - Rockafellar, R.T. & Uryasev, S. (2000). Optimization of Conditional 
      Value-at-Risk. Journal of Risk.
    - Ahmadi-Javid, A. (2012). Entropic Value-at-Risk: A New Coherent Risk 
      Measure. Journal of Optimization Theory and Applications.
    - Chekhlov, A., Uryasev, S., & Zabarankin, M. (2005). Drawdown Measure 
      in Portfolio Optimization. International Journal of Theoretical and 
      Applied Finance.
    """
    
    # Variance-Based Measures
    VARIANCE = "VARIANCE"
    SEMI_VARIANCE = "SEMI_VARIANCE"
    MEAN_ABSOLUTE_DEVIATION = "MEAN_ABSOLUTE_DEVIATION"
    FIRST_LOWER_PARTIAL_MOMENT = "FIRST_LOWER_PARTIAL_MOMENT"
    
    # Value-at-Risk Family
    VALUE_AT_RISK = "VALUE_AT_RISK"
    CONDITIONAL_VALUE_AT_RISK = "CONDITIONAL_VALUE_AT_RISK"
    ENTROPIC_VALUE_AT_RISK = "ENTROPIC_VALUE_AT_RISK"
    WORST_REALIZATION = "WORST_REALIZATION"
    
    # Drawdown Measures
    MAXIMUM_DRAWDOWN = "MAXIMUM_DRAWDOWN"
    AVERAGE_DRAWDOWN = "AVERAGE_DRAWDOWN"
    CONDITIONAL_DRAWDOWN_AT_RISK = "CONDITIONAL_DRAWDOWN_AT_RISK"
    DRAWDOWN_AT_RISK = "DRAWDOWN_AT_RISK"
    ENTROPIC_DRAWDOWN_AT_RISK = "ENTROPIC_DRAWDOWN_AT_RISK"
    ULCER_INDEX = "ULCER_INDEX"
    
    # Higher Moment Measures
    FOURTH_CENTRAL_MOMENT = "FOURTH_CENTRAL_MOMENT"
    FOURTH_LOWER_PARTIAL_MOMENT = "FOURTH_LOWER_PARTIAL_MOMENT"
    SKEW = "SKEW"
    KURTOSIS = "KURTOSIS"
    
    # Other Risk Measures
    GINI_MEAN_DIFFERENCE = "GINI_MEAN_DIFFERENCE"
    ENTROPIC_RISK_MEASURE = "ENTROPIC_RISK_MEASURE"
    
    @classmethod
    def get_variance_based_measures(cls) -> list["risk_measure_type"]:
        """Return all variance-based risk measures.
        
        Returns
        -------
        list[risk_measure_type]
            List of variance-based risk measure types including
            variance, semi-variance, MAD, and first LPM.
        """
        return [
            cls.VARIANCE,
            cls.SEMI_VARIANCE,
            cls.MEAN_ABSOLUTE_DEVIATION,
            cls.FIRST_LOWER_PARTIAL_MOMENT,
        ]
    
    @classmethod
    def get_var_family_measures(cls) -> list["risk_measure_type"]:
        """Return all Value-at-Risk family risk measures.
        
        Returns
        -------
        list[risk_measure_type]
            List of VaR family risk measure types including
            VaR, CVaR, EVaR, and worst realization.
        """
        return [
            cls.VALUE_AT_RISK,
            cls.CONDITIONAL_VALUE_AT_RISK,
            cls.ENTROPIC_VALUE_AT_RISK,
            cls.WORST_REALIZATION,
        ]
    
    @classmethod
    def get_drawdown_measures(cls) -> list["risk_measure_type"]:
        """Return all drawdown-based risk measures.
        
        Returns
        -------
        list[risk_measure_type]
            List of drawdown risk measure types including
            MDD, average drawdown, CDaR, DaR, EDaR, and Ulcer Index.
        """
        return [
            cls.MAXIMUM_DRAWDOWN,
            cls.AVERAGE_DRAWDOWN,
            cls.CONDITIONAL_DRAWDOWN_AT_RISK,
            cls.DRAWDOWN_AT_RISK,
            cls.ENTROPIC_DRAWDOWN_AT_RISK,
            cls.ULCER_INDEX,
        ]
    
    @classmethod
    def get_higher_moment_measures(cls) -> list["risk_measure_type"]:
        """Return all higher moment risk measures.
        
        Returns
        -------
        list[risk_measure_type]
            List of higher moment risk measure types including
            fourth central moment, fourth LPM, skew, and kurtosis.
        """
        return [
            cls.FOURTH_CENTRAL_MOMENT,
            cls.FOURTH_LOWER_PARTIAL_MOMENT,
            cls.SKEW,
            cls.KURTOSIS,
        ]
    
    @classmethod
    def get_coherent_measures(cls) -> list["risk_measure_type"]:
        """Return all coherent risk measures.
        
        Coherent risk measures satisfy the axioms of monotonicity,
        sub-additivity, positive homogeneity, and translation invariance.
        
        Returns
        -------
        list[risk_measure_type]
            List of coherent risk measure types.
        
        References
        ----------
        - Artzner, P., Delbaen, F., Eber, J.M., & Heath, D. (1999).
          Coherent Measures of Risk. Mathematical Finance.
        """
        return [
            cls.CONDITIONAL_VALUE_AT_RISK,
            cls.ENTROPIC_VALUE_AT_RISK,
            cls.WORST_REALIZATION,
            cls.CONDITIONAL_DRAWDOWN_AT_RISK,
            cls.ENTROPIC_DRAWDOWN_AT_RISK,
            cls.MEAN_ABSOLUTE_DEVIATION,
            cls.SEMI_VARIANCE,
            cls.FIRST_LOWER_PARTIAL_MOMENT,
            cls.GINI_MEAN_DIFFERENCE,
        ]
    
    @classmethod
    def get_convex_measures(cls) -> list["risk_measure_type"]:
        """Return all convex risk measures.
        
        Convex risk measures satisfy monotonicity, convexity,
        and translation invariance (relaxing sub-additivity).
        
        Returns
        -------
        list[risk_measure_type]
            List of convex risk measure types.
        """
        return [
            *cls.get_coherent_measures(),
            cls.VARIANCE,
            cls.ENTROPIC_RISK_MEASURE,
            cls.VALUE_AT_RISK,
        ]