"""
Annuity base class.

===============================

Abstract base class defining the interface and common functionality
for all annuity product types (SPIA, DIA, FIA, VA).

Author
------
QWIM Team

Version
-------
0.5.2 (2026-03-01)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import msgspec

from aenum import Enum


if TYPE_CHECKING:
    import polars as pl

from src.utils.custom_exceptions_errors_loggers.exception_custom import (
    Exception_Validation_Input,
)
from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


logger = get_logger(__name__)


# Defined in C for maximum speed
class Withdrawal_Rates(msgspec.Struct):
    """Withdrawal rates structure for annuity income calculations.

    This struct holds both nominal and real (inflation-adjusted) withdrawal rates
    as decimals. Defined in C for maximum speed.

    Attributes
    ----------
    nominal_WR : float
        Nominal withdrawal rate as a decimal (e.g., 0.05 for 5%).
    real_WR : float
        Real withdrawal rate adjusted for inflation as a decimal.
    """

    nominal_WR: float  # nominal withdrawal rate (as a decimal, e.g. 0.05 for 5 %)
    real_WR: float  # real withdrawal rate (adjusted for inflation, as a decimal)


class Annuity_Type(Enum):
    """Enumeration for different types of annuities.

    This enum represents the standard annuity types available in the system.

    Attributes
    ----------
    ANNUITY_SPIA : str
        Single Premium Immediate Annuity — immediate lifetime/period income.
    ANNUITY_DIA : str
        Deferred Income Annuity — income begins at a future specified age.
    ANNUITY_FIA : str
        Fixed Indexed Annuity — returns linked to a market index with floor/cap.
    ANNUITY_VA : str
        Variable Annuity — market growth potential with income guarantees.
    ANNUITY_RILA : str
        Registered Index-Linked Annuity — buffered index exposure with higher caps.
    """

    ANNUITY_SPIA = "Single Premium Immediate Annuity"
    ANNUITY_DIA = "Deferred Income Annuity"
    ANNUITY_FIA = "Fixed Indexed Annuity"
    ANNUITY_VA = "Variable Annuity"
    ANNUITY_RILA = "Registered Index-Linked Annuity"


class Annuity_Payout_Option(Enum):
    """Payout structure options for income annuities.

    Attributes
    ----------
    LIFE_ONLY : str
        Payments continue for the annuitant's lifetime only; no residual
        benefit at death.
    PERIOD_CERTAIN : str
        Payments are guaranteed for a fixed number of years regardless of
        survival.
    LIFE_WITH_PERIOD_CERTAIN : str
        Payments continue for the longer of the annuitant's lifetime or
        a guaranteed period.
    JOINT_LIFE : str
        Payments continue for the lifetimes of two annuitants, with a
        survivor benefit percentage applied after the first death.
    """

    LIFE_ONLY = "Life Only"
    PERIOD_CERTAIN = "Period Certain"
    LIFE_WITH_PERIOD_CERTAIN = "Life with Period Certain"
    JOINT_LIFE = "Joint Life"


class Annuity_Base(ABC):
    """Abstract base class for annuities.

    This class defines the interface and common functionality for all annuity types.
    Derived classes must implement the :meth:`calc_annuity_payout` method.

    Attributes
    ----------
    m_client_age : int
        The age of the client.
    m_annuity_payout_rate : float
        The annual payout rate of the annuity (as a decimal, e.g. 0.05 for 5 %).
    m_annuity_type : Annuity_Type
        The type of annuity.
    m_annuity_income_starting_age : int
        The age when the client will start receiving income from the annuity.
        Defaults to :attr:`m_client_age` when not explicitly provided.
    """

    def __init__(
        self,
        client_age: int,
        annuity_payout_rate: float,
        annuity_type: Annuity_Type,
        annuity_income_starting_age: int | None = None,
    ) -> None:
        """Initialize an annuity base object.

        Parameters
        ----------
        client_age : int
            The age of the client.
        annuity_payout_rate : float
            The annual payout rate of the annuity (as a decimal).
        annuity_type : Annuity_Type
            The type of annuity.
        annuity_income_starting_age : int | None, optional
            The age when the client will start receiving income from the
            annuity.  Must be ``>= client_age`` when provided.  Defaults
            to ``client_age`` when ``None``.

        Raises
        ------
        Exception_Validation_Input
            If ``client_age`` is not a positive integer.
            If ``annuity_payout_rate`` is not a positive number.
            If ``annuity_type`` is not a valid :class:`Annuity_Type`.
            If ``annuity_income_starting_age`` is provided but is not an
            integer ``>= client_age``.
        """
        # --- Input validation with early returns ---
        if not isinstance(client_age, int) or client_age <= 0:
            raise Exception_Validation_Input(
                "client_age must be a positive integer",
                field_name="client_age",
                expected_type=int,
                actual_value=client_age,
            )

        if not isinstance(annuity_payout_rate, (int, float)) or annuity_payout_rate <= 0:
            raise Exception_Validation_Input(
                "annuity_payout_rate must be a positive number",
                field_name="annuity_payout_rate",
                expected_type=float,
                actual_value=annuity_payout_rate,
            )

        if not isinstance(annuity_type, Annuity_Type):
            raise Exception_Validation_Input(
                "annuity_type must be a valid Annuity_Type enum",
                field_name="annuity_type",
                expected_type=Annuity_Type,
                actual_value=annuity_type,
            )

        if annuity_income_starting_age is not None and (
            not isinstance(annuity_income_starting_age, int)
            or annuity_income_starting_age < client_age
        ):
            raise Exception_Validation_Input(
                "annuity_income_starting_age must be an integer >= client_age",
                field_name="annuity_income_starting_age",
                expected_type=int,
                actual_value=annuity_income_starting_age,
            )

        # --- Set member variables ---
        self.m_client_age: int = client_age
        self.m_annuity_payout_rate: float = float(annuity_payout_rate)
        self.m_annuity_type: Annuity_Type = annuity_type
        self.m_annuity_income_starting_age: int = (
            annuity_income_starting_age if annuity_income_starting_age is not None else client_age
        )

        logger.info(
            f"Created {annuity_type.value} for client age {client_age} "
            f"with payout rate {annuity_payout_rate:.2%}",
        )

    # ------------------------------------------------------------------
    # Properties (replace former getter methods)
    # ------------------------------------------------------------------

    @property
    def client_age(self) -> int:
        """Return the client's age."""
        return self.m_client_age

    @property
    def annuity_payout_rate(self) -> float:
        """Return the annuity payout rate (as a decimal)."""
        return self.m_annuity_payout_rate

    @property
    def annuity_type(self) -> Annuity_Type:
        """Annuity_Type : The type of annuity."""
        return self.m_annuity_type

    @property
    def annuity_income_starting_age(self) -> int:
        """Return the age when the client will start receiving income from the annuity."""
        return self.m_annuity_income_starting_age

    # ------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------

    @abstractmethod
    def calc_annuity_payout(
        self,
        amount_principal: float,
        obj_scenarios: pl.DataFrame | None = None,
        obj_inflation: pl.DataFrame | None = None,
    ) -> float:
        """Calculate the annuity payout based on the principal amount.

        This is a pure virtual method that must be implemented by derived classes.

        Parameters
        ----------
        amount_principal : float
            The principal amount invested in the annuity.
        obj_scenarios : pl.DataFrame | None, optional
            Polars DataFrame containing scenario market data.  The first column
            must be ``Date`` (containing dates); the remaining columns each
            represent returns for a specific financial component, identified by
            column name (e.g. ``"S&P 500"``, ``"Equity"``, ``"AMZN"``,
            ``"Fixed Income"``).  Derived classes that use scenario data select
            the relevant column by their stored index/proxy name.
        obj_inflation : pl.DataFrame | None, optional
            Polars DataFrame with pre-calculated inflation factors.  Must
            contain exactly four columns:

            - ``Start Date`` : start of the inflation period.
            - ``End Date``   : end of the inflation period.
            - ``Inflation Factor`` : cumulative inflation already calculated
              for the interval ``[Start Date, End Date]``
            - ``Inverse Inflation Factor`` : inverse of cumulative inflation already calculated
              for the interval ``[Start Date, End Date]``.

            When multiple rows are present the individual factors are
            multiplied to produce a single cumulative adjustment.

        Returns
        -------
        float
            The calculated annuity payout.

        Raises
        ------
        NotImplementedError
            If the derived class does not implement this method.
        """

    @abstractmethod
    def calc_withdrawal_rates(
        self,
        amount_principal: float,
        desired_WR: float | None = None,
        obj_scenarios: pl.DataFrame | None = None,
        obj_inflation: pl.DataFrame | None = None,
    ) -> Withdrawal_Rates:
        """Calculate the withdrawal rates based on a desired withdrawal rate (usually the annuity payout rate).

        This is a pure virtual method that must be implemented by derived classes.

        Parameters
        ----------
        amount_principal : float
            The principal amount invested in the annuity.
        desired_WR : float | None, optional
            The desired withdrawal rate.  When ``None`` (the default),
            the concrete implementation uses ``self.m_annuity_payout_rate``.
        obj_scenarios : pl.DataFrame | None, optional
            Polars DataFrame containing scenario market data.  The first column
            must be ``Date`` (containing dates); the remaining columns each
            represent returns for a specific financial component, identified by
            column name (e.g. ``"S&P 500"``, ``"Equity"``, ``"AMZN"``,
            ``"Fixed Income"``).  Derived classes that use scenario data select
            the relevant column by their stored index/proxy name.
        obj_inflation : pl.DataFrame | None, optional
            Polars DataFrame with pre-calculated inflation factors.  Must
            contain exactly four columns:

            - ``Start Date`` : start of the inflation period.
            - ``End Date``   : end of the inflation period.
            - ``Inflation Factor`` : cumulative inflation already calculated
              for the interval ``[Start Date, End Date]``
            - ``Inverse Inflation Factor`` : inverse of cumulative inflation already calculated
              for the interval ``[Start Date, End Date]``.

            When multiple rows are present the individual factors are
            multiplied to produce a single cumulative adjustment.

        Returns
        -------
        Withdrawal_Rates
            The calculated withdrawal rates (nominal and real).

        Raises
        ------
        NotImplementedError
            If the derived class does not implement this method.
        """

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def get_annuity_as_string(self) -> str:
        """Return a human-readable string representation of the annuity.

        Returns
        -------
        str
            String representation of the annuity.
        """
        return (
            f"{self.m_annuity_type.value} "
            f"(Client Age: {self.m_client_age}, "
            f"Payout Rate: {self.m_annuity_payout_rate:.2%})"
        )
