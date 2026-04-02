"""
Annuity base class
===============================

Author
------
QWIM Team

Version
-------
0.5.1 (2026-03-01)
"""

import logging

from abc import ABC, abstractmethod
from enum import Enum


class Annuity_Type(Enum):
    """
    Enumeration for different types of annuities.

    This enum represents the standard annuity types available in the system.
    """

    ANNUITY_SPIA = "Single Premium Immediate Annuity"
    ANNUITY_DIA = "Deferred Income Annuity"
    ANNUITY_FIA = "Fixed Indexed Annuity"
    ANNUITY_VA = "Variable Annuity"


class annuity_base(ABC):
    """
    Abstract base class for annuities.

    This class defines the interface and common functionality for all annuity types.
    Derived classes must implement the calc_annuity_payout method.

    Attributes
    ----------
    m_client_age : int
        The age of the client
    m_annuity_payout_rate : float
        The payout rate of the annuity
    m_annuity_type : Annuity_Type
        The type of annuity
    """

    def __init__(self, client_age: int, annuity_payout_rate: float, annuity_type: Annuity_Type):
        """
        Initialize an annuity base object.

        Parameters
        ----------
        client_age : int
            The age of the client
        annuity_payout_rate : float
            The payout rate of the annuity
        annuity_type : Annuity_Type
            The type of annuity

        Raises
        ------
        ValueError
            If client_age is not a positive integer
            If annuity_payout_rate is not a positive float
            If annuity_type is not a valid Annuity_Type
        """
        # Validate inputs
        if not isinstance(client_age, int) or client_age <= 0:
            raise ValueError("client_age must be a positive integer")

        if not isinstance(annuity_payout_rate, (int, float)) or annuity_payout_rate <= 0:
            raise ValueError("annuity_payout_rate must be a positive number")

        if not isinstance(annuity_type, Annuity_Type):
            raise ValueError("annuity_type must be a valid Annuity_Type enum")

        # Set member variables
        self.m_client_age = client_age
        self.m_annuity_payout_rate = float(annuity_payout_rate)
        self.m_annuity_type = annuity_type

        # Initialize logger
        self.logger = logging.getLogger("QWIM.Annuity")
        self.logger.info(
            f"Created {annuity_type.value} for client age {client_age} with payout rate {annuity_payout_rate:.2%}",
        )

    def get_client_age(self) -> int:
        """
        Get the client's age.

        Returns
        -------
        int
            The client's age
        """
        return self.m_client_age

    def get_annuity_payout_rate(self) -> float:
        """
        Get the annuity payout rate.

        Returns
        -------
        float
            The annuity payout rate
        """
        return self.m_annuity_payout_rate

    def get_annuity_type(self) -> Annuity_Type:
        """
        Get the annuity type.

        Returns
        -------
        Annuity_Type
            The type of annuity
        """
        return self.m_annuity_type

    @abstractmethod
    def calc_annuity_payout(
        self,
        amount_principal: float,
        obj_scenarios=None,
        obj_inflation=None,
    ) -> float:
        """
        Calculate the annuity payout based on the principal amount, scenarios, and inflation.

        This is a pure virtual method that must be implemented by derived classes.

        Parameters
        ----------
        amount_principal : float
            The principal amount invested in the annuity
        obj_scenarios : object, optional
            Scenario data used for annuity payout calculations
        obj_inflation : object, optional
            Inflation data used for annuity payout calculations

        Returns
        -------
        float
            The calculated annuity payout

        Raises
        ------
        NotImplementedError
            If the derived class does not implement this method
        """

    def __str__(self) -> str:
        """
        Return a string representation of the annuity.

        Returns
        -------
        str
            String representation of the annuity
        """
        return f"{self.m_annuity_type.value} (Client Age: {self.m_client_age}, Payout Rate: {self.m_annuity_payout_rate:.2%})"
