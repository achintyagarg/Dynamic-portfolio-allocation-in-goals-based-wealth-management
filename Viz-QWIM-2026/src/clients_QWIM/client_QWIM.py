"""
Class of a client in QWIM (Quantitative Wealth and Investment Management)
===============================

Author
------
QWIM Team

Version
-------
0.5.1 (2026-03-01)
"""

import logging
import os
import sys
import typing
import platform
from datetime import datetime
from pathlib import Path
import asyncio
import traceback
from enum import Enum, auto

import numpy as np
import polars as pl


class Marital_Status(Enum):
    """Enumeration for marital status options."""
    SINGLE = "Single"
    MARRIED = "Married"
    DIVORCED = "Divorced"
    WIDOWED = "Widowed"
    DOMESTIC_PARTNERSHIP = "Domestic Partnership"


class Employment_Status(Enum):
    """Enumeration for employment status options."""
    EMPLOYED = "Employed"
    UNEMPLOYED = "Unemployed"
    SELF_EMPLOYED = "Self-employed"
    RETIRED = "Retired"
    STUDENT = "Student"


class Asset_Type(Enum):
    """Enumeration for asset types."""
    CASH = "Cash"
    STOCKS = "Stocks"
    BONDS = "Bonds"
    REAL_ESTATE = "Real Estate"
    RETIREMENT_ACCOUNT = "Retirement Account"
    OTHER = "Other"


class Goal_Type(Enum):
    """Enumeration for financial goal types."""
    RETIREMENT = "Retirement"
    EDUCATION = "Education"
    HOME_PURCHASE = "Home Purchase"
    TRAVEL = "Travel"
    EMERGENCY_FUND = "Emergency Fund"
    OTHER = "Other"


class Income_Type(Enum):
    """Enumeration for income source types."""
    SALARY = "Salary"
    BUSINESS = "Business Income"
    INVESTMENT = "Investment Income"
    PENSION = "Pension"
    SOCIAL_SECURITY = "Social Security"
    OTHER = "Other"


class Client_Type(Enum):
    """Enumeration for client types."""
    CLIENT_PRIMARY = "Client Primary"
    CLIENT_PARTNER = "Client Partner"


class Client_QWIM:
    """
    Represents a client in the QWIM system, storing personal information, assets,
    financial goals, and income details from the dashboard inputs.
    """
    
    def __init__(self, 
        client_ID: str, 
        first_name: str, 
        last_name: str,
        client_type: Client_Type):
        """
        Initialize a new QWIM client.
        
        Parameters
        ----------
        client_ID : str
            Unique identifier for the client
        first_name : str
            Client's first name
        last_name : str
            Client's last name
        client_type : ClientType
            Type of client (primary or partner)
        """
        self.m_client_ID = client_ID
        self.m_first_name = first_name
        self.m_last_name = last_name
        self.m_creation_date = datetime.now()
        self.m_last_updated = datetime.now()
        # Set client type based on the provided parameter
        self.m_client_type = client_type
        
        # Personal info schema with specific fields
        self.m_personal_info = pl.DataFrame(schema={
            "First Name": pl.Utf8,
            "Last Name": pl.Utf8,
            "Marital Status": pl.Utf8,
            "Gender": pl.Utf8,
            "Risk Tolerance": pl.Int64,
            "Current Age": pl.Int64,
            "Retirement Age": pl.Int64,
            "Income Start Age for Strategy Annuity": pl.Int64
        })
        
        # Updated assets schema with specific asset categories
        self.m_assets = pl.DataFrame(schema={
            "Investable Assets": pl.Float64,
            "Taxable Assets": pl.Float64,
            "Tax Deferred Assets": pl.Float64,
            "Tax Free Assets": pl.Float64,
            "Asset Name": pl.Utf8,  # Optional descriptive name
            "Asset Class": pl.Utf8   # Optional classification
        })
        
        # Updated goals schema with specific expense categories
        self.m_goals = pl.DataFrame(schema={
            "Essential Annual Expense": pl.Float64,
            "Important Annual Expense": pl.Float64,
            "Aspirational Annual Expense": pl.Float64,
            "Essential Annual Expense is Inflation Indexed": pl.Boolean,
            "Important Annual Expense is Inflation Indexed": pl.Boolean,
            "Aspirational Annual Expense is Inflation Indexed": pl.Boolean
        })
        
        # Updated income schema with specific income categories
        self.m_income = pl.DataFrame(schema={
            "Annual Social Security": pl.Float64,
            "Annual Income from Pension": pl.Float64,
            "Annual Income from Existing Annuity": pl.Float64,
            "Annual Income from Other Sources": pl.Float64,
            "Annual Income from Pension is Inflation Indexed": pl.Boolean,
            "Annual Income from Existing Annuity is Inflation Indexed": pl.Boolean,
            "Annual Income from Other Sources is Inflation Indexed": pl.Boolean,
            "Income Start Age for Pension": pl.Int64,
            "Income Start Age for Existing Annuity": pl.Int64,
            "Income Start Age for Other Sources": pl.Int64,
            "Income Duration for Pension": pl.Int64,
            "Income Duration for Existing Annuity": pl.Int64,
            "Income Duration for Other Sources": pl.Int64
        })
        
        self.logger = logging.getLogger("client.QWIM")
    
    def update_personal_info(self, data_personal_info: dict):
        """
        Update client personal information from dashboard inputs.
        
        Parameters
        ----------
        data_personal_info : dict
            Dictionary containing personal information fields and values
        """
        try:
            # Create a DataFrame with one row containing all the personal information
            # Filter out any keys that aren't in our schema
            valid_fields = ["First Name", "Last Name", "Marital Status", "Gender", 
                           "Risk Tolerance", "Current Age", "Retirement Age", "Income Start Age"]
            
            # Create a new dictionary with only valid fields
            filtered_data = {item_k: item_v for item_k, item_v in data_personal_info.items() if item_k in valid_fields}
            
            # Ensure numeric fields are properly typed
            numeric_fields = ["Risk Tolerance", "Current Age", "Retirement Age", "Income Start Age"]
            for item_field in numeric_fields:
                if item_field in filtered_data:
                    try:
                        filtered_data[item_field] = int(filtered_data[item_field])
                    except (ValueError, TypeError):
                        # If conversion fails, set to None
                        self.logger.warning(f"Could not convert {item_field} to integer: {filtered_data[item_field]}")
                        filtered_data[item_field] = None
            
            # Create new DataFrame with the single row of data - updated to use m_personal_info
            self.m_personal_info = pl.DataFrame([filtered_data])
            
            # Extract common fields if present
            if "First Name" in data_personal_info:
                self.m_first_name = data_personal_info["First Name"]
            if "Last Name" in data_personal_info:
                self.m_last_name = data_personal_info["Last Name"]
            
            self.m_last_updated = datetime.now()
            self.logger.info(f"Updated personal information for client {self.m_client_ID}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating personal information: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def update_assets(self, data_assets: list):
        """
        Update client assets from dashboard inputs.
        
        Parameters
        ----------
        data_assets : list
            List of dictionaries containing asset information
        """
        try:
            # Convert input list to DataFrame with the new schema
            if data_assets:
                # Create a clean DataFrame with the expected schema
                # For each asset record in data, ensure it matches our schema structure
                clean_data = []
                
                for item_asset in data_assets:
                    clean_asset = {
                        "Taxable Assets": 0.0,
                        "Tax Deferred Assets": 0.0,
                        "Tax Free Assets": 0.0,
                        "Asset Name": "",
                        "Asset Class": ""
                    }
                    
                    # Map data values to our schema
                    if "Taxable Assets" in item_asset:
                        try:
                            clean_asset["Taxable Assets"] = float(item_asset["Taxable Assets"])
                        except (ValueError, TypeError):
                            pass
                            
                    if "Tax Deferred Assets" in item_asset:
                        try:
                            clean_asset["Tax Deferred Assets"] = float(item_asset["Tax Deferred Assets"])
                        except (ValueError, TypeError):
                            pass
                            
                    if "Tax Free Assets" in item_asset:
                        try:
                            clean_asset["Tax Free Assets"] = float(item_asset["Tax Free Assets"])
                        except (ValueError, TypeError):
                            pass
                    
                    # Optional fields
                    if "Asset Name" in item_asset:
                        clean_asset["Asset Name"] = str(item_asset["Asset Name"])
                        
                    if "Asset Class" in item_asset:
                        clean_asset["Asset Class"] = str(item_asset["Asset Class"])
                    elif "Type" in asset:  # Backward compatibility
                        clean_asset["Asset Class"] = str(item_asset["Type"])
                    
                    clean_data.append(clean_asset)
                
                self.m_assets = pl.DataFrame(clean_data)
            else:
                # Reset to empty dataframe if no data
                self.m_assets = pl.DataFrame(schema={
                    "Taxable Assets": pl.Float64,
                    "Tax Deferred Assets": pl.Float64,
                    "Tax Free Assets": pl.Float64,
                    "Asset Name": pl.Utf8,
                    "Asset Class": pl.Utf8
                })
            
            self.m_last_updated = datetime.now()
            self.logger.info(f"Updated assets for client {self.m_client_ID}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating assets: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def update_goals(self, data_goals: list):
        """
        Update client financial goals from dashboard inputs.
        
        Parameters
        ----------
        data_goals : list or dict
            List of dictionaries or single dictionary containing goals information
        """
        try:
            # Handle both list and dictionary input formats
            if isinstance(data_goals, dict):
                data_goals = [data_goals]  # Convert single dict to list
                
            if data_goals:
                # Create a clean DataFrame with the expected schema
                clean_data = []
                
                for item_goal in data_goals:
                    clean_goal = {
                        "Essential Annual Expense": 0.0,
                        "Important Annual Expense": 0.0,
                        "Aspirational Annual Expense": 0.0,
                        "Essential Annual Expense is Inflation Indexed": False,
                        "Important Annual Expense is Inflation Indexed": False,
                        "Aspirational Annual Expense is Inflation Indexed": False
                    }
                    
                    # Process expense amounts
                    for expense_type in ["Essential Annual Expense", "Important Annual Expense", "Aspirational Annual Expense"]:
                        if expense_type in item_goal:
                            try:
                                clean_goal[expense_type] = float(item_goal[expense_type])
                            except (ValueError, TypeError):
                                pass
                    
                    # Process inflation indexing flags
                    for idx_field in [
                        "Essential Annual Expense is Inflation Indexed",
                        "Important Annual Expense is Inflation Indexed",
                        "Aspirational Annual Expense is Inflation Indexed"
                    ]:
                        if idx_field in item_goal:
                            # Convert various representations of boolean values
                            value = item_goal[idx_field]
                            if isinstance(value, bool):
                                clean_goal[idx_field] = value
                            elif isinstance(value, str):
                                clean_goal[idx_field] = value.lower() in ('true', 'yes', 'y', '1')
                            elif isinstance(value, (int, float)):
                                clean_goal[idx_field] = bool(value)
                    
                    clean_data.append(clean_goal)
                
                self.m_goals = pl.DataFrame(clean_data)
            else:
                # Reset to empty dataframe if no data
                self.m_goals = pl.DataFrame(schema={
                    "Essential Annual Expense": pl.Float64,
                    "Important Annual Expense": pl.Float64,
                    "Aspirational Annual Expense": pl.Float64,
                    "Essential Annual Expense is Inflation Indexed": pl.Boolean,
                    "Important Annual Expense is Inflation Indexed": pl.Boolean,
                    "Aspirational Annual Expense is Inflation Indexed": pl.Boolean
                })
            
            self.m_last_updated = datetime.now()
            self.logger.info(f"Updated goals for client {self.m_client_ID}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating goals: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def update_income(self, data_income: list):
        """
        Update client income sources from dashboard inputs.
        
        Parameters
        ----------
        data_income : list or dict
            List of dictionaries or single dictionary containing income information
        """
        try:
            # Handle both list and dictionary input formats
            if isinstance(data_income, dict):
                data_income = [data_income]  # Convert single dict to list
                
            if data_income:
                # Create a clean DataFrame with the expected schema
                clean_data = []
                
                for income_entry in data_income:
                    clean_income = {
                       "Annual Social Security": 0.0,
                        "Annual Income from Pension": 0.0,
                        "Annual Income from Existing Annuity": 0.0,
                        "Annual Income from Other Sources": 0.0,
                        "Annual Income from Pension is Inflation Indexed": False,
                        "Annual Income from Existing Annuity is Inflation Indexed": False,
                        "Annual Income from Other Sources is Inflation Indexed": False,
                        "Income Start Age for Pension": 65,
                        "Income Start Age for Existing Annuity": 65,
                        "Income Start Age for Other Sources": 65,
                        "Income Duration for Pension": 25,
                        "Income Duration for Existing Annuity": 25,
                        "Income Duration for Other Sources": 25
                    }
                    
                    # Process income amounts
                    for income_type in ["Annual Social Security", "Annual Pension in Retirement", 
                                       "Annual Annuity Income", "Annual Other Income"]:
                        if income_type in income_entry:
                            try:
                                clean_income[income_type] = float(income_entry[income_type])
                            except (ValueError, TypeError):
                                pass
                    
                    # Process inflation indexing flags
                    for idx_field in [
                        "Annual Pension in Retirement is Inflation Indexed",
                        "Annual Annuity Income is Inflation Indexed",
                        "Annual Other Income is Inflation Indexed"
                    ]:
                        if idx_field in income_entry:
                            # Convert various representations of boolean values
                            value = income_entry[idx_field]
                            if isinstance(value, bool):
                                clean_income[idx_field] = value
                            elif isinstance(value, str):
                                clean_income[idx_field] = value.lower() in ('true', 'yes', 'y', '1')
                            elif isinstance(value, (int, float)):
                                clean_income[idx_field] = bool(value)
                    
                    clean_data.append(clean_income)
                
                self.m_income = pl.DataFrame(clean_data)
            else:
                # Reset to empty dataframe if no data
                self.m_income = pl.DataFrame(schema={
                    "Annual Social Security": pl.Float64,
                    "Annual Pension in Retirement": pl.Float64,
                    "Annual Annuity Income": pl.Float64,
                    "Annual Other Income": pl.Float64,
                    "Annual Pension in Retirement is Inflation Indexed": pl.Boolean,
                    "Annual Annuity Income is Inflation Indexed": pl.Boolean,
                    "Annual Other Income is Inflation Indexed": pl.Boolean
                })
            
            self.m_last_updated = datetime.now()
            self.logger.info(f"Updated income for client {self.m_client_ID}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating income: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def get_personal_info(self) -> pl.DataFrame:
        """Return the client's personal information DataFrame."""
        return self.m_personal_info

    def get_marital_status(self) -> str:
        """Return the client's marital status."""
        if self.m_personal_info.is_empty():
            return None
        try:
            return self.m_personal_info.select("Marital Status").item()
        except:
            return None

    def get_risk_tolerance(self) -> int:
        """Return the client's risk tolerance."""
        if self.m_personal_info.is_empty():
            return None
        try:
            return self.m_personal_info.select("Risk Tolerance").item()
        except:
            return None

    def get_current_age(self) -> int:
        """Return the client's current age."""
        if self.m_personal_info.is_empty():
            return None
        try:
            return self.m_personal_info.select("Current Age").item()
        except:
            return None

    def get_retirement_age(self) -> int:
        """Return the client's retirement age."""
        if self.m_personal_info.is_empty():
            return None
        try:
            return self.m_personal_info.select("Retirement Age").item()
        except:
            return None

    def get_income_start_age(self) -> int:
        """Return the client's income start age."""
        if self.m_personal_info.is_empty():
            return None
        try:
            return self.m_personal_info.select("Income Start Age").item()
        except:
            return None

    def get_assets(self) -> pl.DataFrame:
        """Return the client's assets DataFrame."""
        return self.m_assets
    
    def get_goals(self) -> pl.DataFrame:
        """Return the client's financial goals DataFrame."""
        return self.m_goals
    
    def get_income(self) -> pl.DataFrame:
        """Return the client's income sources DataFrame."""
        return self.m_income
    
    def get_total_assets(self) -> float:
        """Calculate and return the total value of client's assets."""
        if self.m_assets.is_empty():
            return 0.0
        
        try:
            # Sum the three asset categories
            asset_columns = ["Taxable Assets", "Tax Deferred Assets", "Tax Free Assets"]
            asset_sums = self.m_assets.select([
                pl.sum(idx_col).alias(f"total_{idx_col.lower().replace(' ', '_')}")
                for idx_col in asset_columns
            ])
            
            # Calculate the grand total
            totals = [asset_sums[0, f"total_{idx_col.lower().replace(' ', '_')}"] for idx_col in asset_columns]
            total_assets = sum(idx_t for idx_t in totals if idx_t is not None)
            
            return float(total_assets)
                
        except Exception as e:
            self.logger.error(f"Error calculating total assets: {e}")
            return 0.0

    def get_taxable_assets(self) -> float:
        """Get total taxable assets."""
        if self.m_assets.is_empty():
            return 0.0
            
        try:
            return float(self.m_assets.select(pl.sum("Taxable Assets")).item() or 0.0)
        except Exception as e:
            self.logger.error(f"Error getting taxable assets: {e}")
            return 0.0
            
    def get_tax_deferred_assets(self) -> float:
        """Get total tax deferred assets."""
        if self.m_assets.is_empty():
            return 0.0
            
        try:
            return float(self.m_assets.select(pl.sum("Tax Deferred Assets")).item() or 0.0)
        except Exception as e:
            self.logger.error(f"Error getting tax deferred assets: {e}")
            return 0.0
            
    def get_tax_free_assets(self) -> float:
        """Get total tax free assets."""
        if self.m_assets.is_empty():
            return 0.0
            
        try:
            return float(self.m_assets.select(pl.sum("Tax Free Assets")).item() or 0.0)
        except Exception as e:
            self.logger.error(f"Error getting tax free assets: {e}")
            return 0.0
    
    def get_total_annual_income(self) -> float:
        """Calculate and return the total annual income for the client."""
        if self.m_income.is_empty():
            return 0.0
        
        try:
            income_columns = [
                "Annual Social Security",
                "Annual Pension in Retirement",
                "Annual Annuity Income",
                "Annual Other Income"
            ]
            
            # Sum all income columns
            income_sum = self.m_income.select([
                pl.sum(col).alias(f"total_{col.lower().replace(' ', '_')}")
                for idx_col in income_columns
            ])
            
            # Calculate the grand total
            totals = [income_sum[0, f"total_{idx_col.lower().replace(' ', '_')}"] for idx_col in income_columns]
            total_income = sum(t for idx_t in totals if idx_t is not None)
            
            return float(total_income)
                
        except Exception as e:
            self.logger.error(f"Error calculating total annual income: {e}")
            return 0.0

    def get_social_security_income(self) -> float:
        """Get total social security income."""
        if self.m_income.is_empty():
            return 0.0
        
        try:
            return float(self.m_income.select(pl.sum("Annual Social Security")).item() or 0.0)
        except Exception as e:
            self.logger.error(f"Error getting social security income: {e}")
            return 0.0

    def get_pension_income(self) -> float:
        """Get total pension income."""
        if self.m_income.is_empty():
            return 0.0
        
        try:
            return float(self.m_income.select(pl.sum("Annual Pension in Retirement")).item() or 0.0)
        except Exception as e:
            self.logger.error(f"Error getting pension income: {e}")
            return 0.0

    def get_annuity_income(self) -> float:
        """Get total annuity income."""
        if self.m_income.is_empty():
            return 0.0
        
        try:
            return float(self.m_income.select(pl.sum("Annual Annuity Income")).item() or 0.0)
        except Exception as e:
            self.logger.error(f"Error getting annuity income: {e}")
            return 0.0

    def get_other_income(self) -> float:
        """Get total other income."""
        if self.m_income.is_empty():
            return 0.0
        
        try:
            return float(self.m_income.select(pl.sum("Annual Other Income")).item() or 0.0)
        except Exception as e:
            self.logger.error(f"Error getting other income: {e}")
            return 0.0

    def is_pension_income_inflation_indexed(self) -> bool:
        """Check if pension income is inflation indexed."""
        if self.m_income.is_empty():
            return False
        
        try:
            return bool(self.m_income.select(pl.any("Annual Income from Pension is Inflation Indexed")).item())
        except Exception as e:
            self.logger.error(f"Error checking if pension is inflation indexed: {e}")
            return False

    def is_income_from_existing_annuity_inflation_indexed(self) -> bool:
        """Check if income from existing annuity is inflation indexed."""
        if self.m_income.is_empty():
            return False
        
        try:
            return bool(self.m_income.select(pl.any("Annual Income from Existing Annuity is Inflation Indexed")).item())
        except Exception as e:
            self.logger.error(f"Error checking if income from existing annuity is inflation indexed: {e}")
            return False

    def is_income_from_other_sources_inflation_indexed(self) -> bool:
        """Check if other income is inflation indexed."""
        if self.m_income.is_empty():
            return False
        
        try:
            return bool(self.m_income.select(pl.any("Annual Income from Other Sources is Inflation Indexed")).item())
        except Exception as e:
            self.logger.error(f"Error checking if income from other sources is inflation indexed: {e}")
            return False

    def get_total_annual_expenses(self) -> float:
        """Calculate and return total annual expenses across all categories."""
        if self.m_goals.is_empty():
            return 0.0
        
        try:
            expense_columns = [
                "Essential Annual Expense", 
                "Important Annual Expense", 
                "Aspirational Annual Expense"
            ]
            
            # Sum all expense columns
            expenses_sum = self.m_goals.select([
                pl.sum(idx_col).alias(f"total_{idx_col.lower().replace(' ', '_')}")
                for idx_col in expense_columns
            ])
            
            # Calculate the grand total
            totals = [expenses_sum[0, f"total_{idx_col.lower().replace(' ', '_')}"] for idx_col in expense_columns]
            total_expenses = sum(idx_t for idx_t in totals if idx_t is not None)
            
            return float(total_expenses)
        
        except Exception as e:
            self.logger.error(f"Error calculating total annual expenses: {e}")
            return 0.0

    def get_annual_essential_expenses(self) -> float:
        """Get total essential annual expenses."""
        if self.m_goals.is_empty():
            return 0.0
        
        try:
            return float(self.m_goals.select(pl.sum("Essential Annual Expense")).item() or 0.0)
        except Exception as e:
            self.logger.error(f"Error getting essential expenses: {e}")
            return 0.0

    def get_annual_important_expenses(self) -> float:
        """Get total important annual expenses."""
        if self.m_goals.is_empty():
            return 0.0
        
        try:
            return float(self.m_goals.select(pl.sum("Important Annual Expense")).item() or 0.0)
        except Exception as e:
            self.logger.error(f"Error getting important expenses: {e}")
            return 0.0

    def get_annual_aspirational_expenses(self) -> float:
        """Get total aspirational annual expenses."""
        if self.m_goals.is_empty():
            return 0.0
        
        try:
            return float(self.m_goals.select(pl.sum("Aspirational Annual Expense")).item() or 0.0)
        except Exception as e:
            self.logger.error(f"Error getting aspirational expenses: {e}")
            return 0.0

    def is_annual_essential_expense_inflation_indexed(self) -> bool:
        """Check if essential expenses are inflation indexed."""
        if self.m_goals.is_empty():
            return False
        
        try:
            # Return True if any row has this set to True
            return bool(self.m_goals.select(pl.any("Essential Annual Expense is Inflation Indexed")).item())
        except Exception as e:
            self.logger.error(f"Error checking if essential expenses are inflation indexed: {e}")
            return False

    def is_annual_important_expense_inflation_indexed(self) -> bool:
        """Check if important expenses are inflation indexed."""
        if self.m_goals.is_empty():
            return False
        
        try:
            return bool(self.m_goals.select(pl.any("Important Annual Expense is Inflation Indexed")).item())
        except Exception as e:
            self.logger.error(f"Error checking if important expenses are inflation indexed: {e}")
            return False

    def is_annual_aspirational_expense_inflation_indexed(self) -> bool:
        """Check if aspirational expenses are inflation indexed."""
        if self.m_goals.is_empty():
            return False
        
        try:
            return bool(self.m_goals.select(pl.any("Aspirational Annual Expense is Inflation Indexed")).item())
        except Exception as e:
            self.logger.error(f"Error checking if aspirational expenses are inflation indexed: {e}")
            return False
    
    def to_dict(self) -> dict:
        """
        Convert client data to a dictionary representation.
        
        Returns
        -------
        dict
            Dictionary containing all client data
        """
        return {
            "client_ID": self.m_client_ID,
            "first_name": self.m_first_name,
            "last_name": self.m_last_name,
            "creation_date": self.m_creation_date.isoformat(),
            "last_updated": self.m_last_updated.isoformat(),
            "personal_info": self.m_personal_info.to_dict(as_series=False),
            "assets": self.m_assets.to_dict(as_series=False),
            "goals": self.m_goals.to_dict(as_series=False),
            "income": self.m_income.to_dict(as_series=False)
        }
    
    @classmethod
    def from_dict(cls, input_data: dict) -> 'client_QWIM':
        """
        Create a client instance from a dictionary representation.
        
        Parameters
        ----------
        input_data : dict
            Dictionary containing client data
            
        Returns
        -------
        client_QWIM
            New client instance populated with the provided data
        """
        client = cls()
        client.m_client_ID = input_data.get("client_ID")
        client.m_first_name = input_data.get("first_name")
        client.m_last_name = input_data.get("last_name")
        
        # Parse dates - updated to use m_creation_date and m_last_updated
        if "creation_date" in input_data:
            client.m_creation_date = datetime.fromisoformat(input_data["creation_date"])
        if "last_updated" in input_data:
            client.m_last_updated = datetime.fromisoformat(input_data["last_updated"])
        
        # Create dataframes from dictionary data
        if "personal_info" in input_data:
            client.m_personal_info = pl.DataFrame(input_data["personal_info"])  # Updated
        
        if "assets" in input_data:
            client.m_assets = pl.DataFrame(input_data["assets"])
        
        if "goals" in input_data:
            client.m_goals = pl.DataFrame(input_data["goals"])
        
        if "income" in input_data:
            client.m_income = pl.DataFrame(input_data["income"])
        
        return client
    
    def __str__(self) -> str:
        """Return a string representation of the client."""
        # Updated to use m_first_name and m_last_name
        name = f"{self.m_first_name} {self.m_last_name}" if self.m_first_name and self.m_last_name else self.m_client_ID
        return f"QWIM Client: {name} (Total Assets: ${self.get_total_assets():,.2f})"
