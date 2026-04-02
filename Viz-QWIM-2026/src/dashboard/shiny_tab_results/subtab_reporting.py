"""PDF Report Generation Module for QWIM Dashboard.

Provides PDF report generation functionality using report_QWIM.py and the
Typst report_QWIM.typ template.  Generates comprehensive reports with
portfolio analysis and investor information.

Features
--------
* User-specified filename with secure file download
* Secure file path sanitization for cloud deployment
* Default save to outputs/reports folder with default filename pattern
* Professional PDF reports via Typst typesetting system
* Integration with reactives_shiny data structure
* Dynamic data integration from dashboard inputs
* Polars DataFrames support (Data_Personal_Info_DF, Data_Assets_DF,
  Data_Goals_DF, Data_Income_DF)

Security Features
-----------------
* Path sanitization to prevent directory traversal attacks
* Filename validation to prevent malicious file operations
* Secure temporary file handling for cloud environments
* Download-based file delivery for Posit Connect Cloud

Dependencies
-----------
* shiny: UI framework
* report_QWIM: Custom PDF generation module (Typst pipeline)
* typst: Typst compilation system
"""

from __future__ import annotations

import contextlib
import datetime
import os
import re
import shutil
import tempfile
import threading
import typing

from datetime import UTC
from pathlib import Path
from typing import Any

from shiny import module, reactive, render, ui

from src.utils.custom_exceptions_errors_loggers.logger_custom import get_logger


#: Module-level logger instance
_logger = get_logger(__name__)


# Import PDF report generation module
try:
    from src.dashboard.reporting.report_QWIM import generate_report_PDF  # noqa: F401

    REPORT_QWIM_AVAILABLE = True
    _logger.info("report_QWIM module imported successfully")
except ImportError as import_error:
    _logger.warning("report_QWIM module import failed: %s", import_error)
    REPORT_QWIM_AVAILABLE = False

# Import Typst compilation functionality
try:
    import typst

    TYPST_AVAILABLE = True
    _logger.info("typst module imported successfully")
except ImportError as import_error:
    _logger.warning("typst module import failed: %s", import_error)
    TYPST_AVAILABLE = False

# Import data access utilities
try:
    from src.dashboard.reporting.report_data_export import (
        export_all_report_data,
        validate_report_data_quality,
    )
    from src.dashboard.reporting.report_plot_export import (
        ensure_all_svg_files_exist,
        export_all_report_plots,
    )
    from src.dashboard.shiny_utils.reactives_shiny import get_value_from_reactives_shiny
    from src.dashboard.shiny_utils.utils_reporting import export_visual_objects_to_svg
    from src.dashboard.shiny_utils.utils_tab_clients import get_investor_data_from_dashboard

    DATA_UTILS_AVAILABLE = True
    _logger.info("reactives_shiny and utils_tab_clients utilities imported successfully")
except ImportError as import_error:
    _logger.warning("Data utilities import failed: %s", import_error)
    DATA_UTILS_AVAILABLE = False

    def export_all_report_data(
        reactives_shiny: typing.Any,
    ) -> dict:  # type: ignore[misc]
        """Fallback: report data export unavailable."""
        print("⚠️ Fallback: export_all_report_data not available")
        return {}

    def export_all_report_plots(
        reactives_shiny: typing.Any,
    ) -> dict:  # type: ignore[misc]
        """Fallback: report plot export unavailable."""
        print("⚠️ Fallback: export_all_report_plots not available")
        return {}

    def export_visual_objects_to_svg(
        reactives_shiny: typing.Any,
        output_dir: typing.Any = None,
    ) -> dict:  # type: ignore[misc]
        """Fallback: SVG export unavailable."""
        print("⚠️ Fallback: export_visual_objects_to_svg not available")
        return {}

    def validate_report_data_quality() -> dict:  # type: ignore[misc]
        """Fallback: data quality validation unavailable."""
        return {}

    def ensure_all_svg_files_exist() -> dict:  # type: ignore[misc]
        """Fallback: SVG existence check unavailable."""
        return {}

    # Fallback function for defensive programming
    def get_value_from_reactives_shiny(
        reactives_shiny: dict,
        key_name: str,
        key_category: str,
    ) -> typing.Any | None:
        """
        Fallback function when reactives_shiny utilities are not available.

        Args:
            reactives_shiny (dict): Reactive values structure
            key_name (str): Key name to retrieve
            key_category (str): Key category

        Returns
        -------
            Optional[typing.Any]: None as fallback
        """
        print(f"⚠️ Fallback: Unable to retrieve {key_name} from {key_category}")
        return None

    def get_investor_data_from_dashboard(
        reactives_shiny: dict,
    ) -> tuple[dict, dict, dict, dict]:
        """
        Fallback function when utils_tab_investors is not available.

        Args:
            reactives_shiny (dict): Reactive values structure

        Returns
        -------
            Tuple[dict, dict, dict, dict]: Fallback data structures with realistic values
        """
        print("⚠️ Fallback: utils_tab_investors not available, generating fallback investor data")

        # Generate realistic fallback data for demonstration purposes
        fallback_personal_info = {
            "primary": {
                "name": "Primary Investor",
                "age_current": 35,
                "age_retirement": 65,
                "age_income_starting": 67,
                "status_marital": "Married",
                "gender": "Not Specified",
                "tolerance_risk": "Moderate",
                "state": "Not Specified",
                "code_zip": "00000",
            },
            "partner": {
                "name": "Partner Investor",
                "age_current": 33,
                "age_retirement": 65,
                "age_income_starting": 67,
                "status_marital": "Married",
                "gender": "Not Specified",
                "tolerance_risk": "Moderate",
                "state": "Not Specified",
                "code_zip": "00000",
            },
        }

        fallback_assets_data = {
            "primary": {
                "taxable": 100000.0,
                "tax_deferred": 200000.0,
                "tax_free": 50000.0,
                "total": 350000.0,
            },
            "partner": {
                "taxable": 75000.0,
                "tax_deferred": 150000.0,
                "tax_free": 25000.0,
                "total": 250000.0,
            },
            "combined": {
                "taxable": 175000.0,
                "tax_deferred": 350000.0,
                "tax_free": 75000.0,
                "total": 600000.0,
            },
        }

        fallback_goals_data = {
            "primary": {
                "essential": 50000.0,
                "important": 25000.0,
                "aspirational": 15000.0,
                "total": 90000.0,
            },
            "partner": {
                "essential": 45000.0,
                "important": 20000.0,
                "aspirational": 10000.0,
                "total": 75000.0,
            },
            "combined": {
                "essential": 95000.0,
                "important": 45000.0,
                "aspirational": 25000.0,
                "total": 165000.0,
            },
        }

        fallback_income_data = {
            "primary": {
                "social_security": 30000.0,
                "pension": 25000.0,
                "annuity_existing": 15000.0,
                "other": 10000.0,
                "total": 80000.0,
            },
            "partner": {
                "social_security": 25000.0,
                "pension": 20000.0,
                "annuity_existing": 10000.0,
                "other": 5000.0,
                "total": 60000.0,
            },
            "combined": {
                "social_security": 55000.0,
                "pension": 45000.0,
                "annuity_existing": 25000.0,
                "other": 15000.0,
                "total": 140000.0,
            },
        }

        print("✅ Generated comprehensive fallback investor data successfully")
        return (
            fallback_personal_info,
            fallback_assets_data,
            fallback_goals_data,
            fallback_income_data,
        )


# Security constants for file path validation
ALLOWED_FILENAME_PATTERN = re.compile(r"^[a-zA-Z0-9._\-\s()]+\.pdf$")
MAX_FILENAME_LENGTH = 200
FORBIDDEN_FILENAME_PARTS = [
    "..",
    "./",
    "\\",
    "//",
    "~",
    "$",
    "%",
    "&",
    "*",
    "?",
    "<",
    ">",
    "|",
    '"',
    "'",
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "COM1",
    "COM2",
    "COM3",
    "COM4",
    "COM5",
    "COM6",
    "COM7",
    "COM8",
    "COM9",
    "LPT1",
    "LPT2",
    "LPT3",
    "LPT4",
    "LPT5",
    "LPT6",
    "LPT7",
    "LPT8",
    "LPT9",
]


# ---------------------------------------------------------------------------
# Module-level utility functions (pure helpers, no Shiny context needed)
# ---------------------------------------------------------------------------


def sanitize_filename_for_security(
    filename: str,
) -> tuple[bool, str, str]:
    """Sanitize filename for security following QWIM standards.

    Parameters
    ----------
    filename : str
        User-provided filename to sanitize.

    Returns
    -------
    tuple[bool, str, str]
        ``(is_valid, sanitized_filename, error_message)``
    """
    # Input validation with early returns
    if not filename or len(filename.strip()) == 0:
        return False, "", "Filename cannot be empty"

    Original_Filename = filename.strip()

    # Configuration validation - check length
    if len(Original_Filename) > MAX_FILENAME_LENGTH:
        return False, "", f"Filename too long (maximum {MAX_FILENAME_LENGTH} characters)"

    if len(Original_Filename) < 5:  # Minimum: "a.pdf"
        return False, "", "Filename too short (minimum 5 characters including .pdf)"

    # Business logic validation - check extension
    if not Original_Filename.lower().endswith(".pdf"):
        return False, "", "Filename must end with .pdf extension"

    # Defensive programming - check for forbidden patterns
    Filename_Upper = Original_Filename.upper()
    for Forbidden_Part in FORBIDDEN_FILENAME_PARTS:
        if Forbidden_Part in Filename_Upper:
            return False, "", f"Filename contains forbidden pattern: {Forbidden_Part}"

    # Security validation - check against allowed pattern
    if not ALLOWED_FILENAME_PATTERN.match(Original_Filename):
        return (
            False,
            "",
            "Filename contains invalid characters. Only letters, numbers, spaces, dots, hyphens, underscores and parentheses allowed",
        )

    # Additional security checks
    if Original_Filename.startswith((".", "-")):
        return False, "", "Filename cannot start with dot or hyphen"

    if ".." in Original_Filename:
        return False, "", "Filename cannot contain double dots"

    # Create sanitized version
    Sanitized_Filename = re.sub(r"[^\w\s.\-()]+", "", Original_Filename)
    Sanitized_Filename = re.sub(r"\s+", " ", Sanitized_Filename).strip()

    # Ensure it still ends with .pdf after sanitization
    if not Sanitized_Filename.lower().endswith(".pdf"):
        Sanitized_Filename += ".pdf"

    return True, Sanitized_Filename, "Filename is valid and secure"


def create_secure_temp_directory() -> str:
    """Create secure temporary directory for PDF generation.

    Returns
    -------
    str
        Path to secure temporary directory.
    """
    try:
        Temp_Dir = tempfile.mkdtemp(prefix="qwim_reports_", suffix="_secure")

        if not os.path.exists(Temp_Dir):
            raise Exception("Failed to create temporary directory")

        if not os.access(Temp_Dir, os.W_OK):
            raise Exception("Temporary directory is not writable")

        return Temp_Dir

    except Exception:
        return tempfile.gettempdir()


def cleanup_temp_files(
    temp_directory: str,
) -> None:
    """Clean up temporary files securely.

    Parameters
    ----------
    temp_directory : str
        Temporary directory to clean up.
    """
    try:
        if not temp_directory or not os.path.exists(temp_directory):
            return

        if (
            not temp_directory.startswith(tempfile.gettempdir())
            and "qwim_reports" not in temp_directory
        ):
            return

        shutil.rmtree(temp_directory, ignore_errors=True)

    except Exception:
        pass


@module.ui
def subtab_reporting_ui(
    data_utils: dict,
    data_inputs: dict,
) -> Any:
    """Create streamlined UI for PDF report generation using Typst report_QWIM.typ with secure file download."""
    # Generate default filename with current date
    Current_Date_String = datetime.datetime.now(UTC).strftime("%Y%m%d")
    Default_Filename = f"Report-QWIM-{Current_Date_String}.pdf"

    return ui.div(
        # Header section
        ui.div(
            ui.h3("📊 QWIM PDF Report Generation", class_="text-primary mb-4"),
            ui.p(
                "Generate comprehensive PDF reports with portfolio analysis and investor information using the Typst typesetting system.",
                class_="text-muted mb-4",
            ),
            ui.p(
                "📥 Reports will be generated and made available for download to your workstation.",
                class_="text-info mb-4 small",
            ),
            class_="text-center",
        ),
        # Report configuration section
        ui.div(
            ui.div(
                # Report title input
                ui.input_text(
                    "input_ID_tab_portfolios_subtab_reporting_text_report_title",
                    "Report Title:",
                    value="QWIM Portfolio and Investor Analysis Report",
                    placeholder="Enter report title...",
                    width="100%",
                ),
                class_="mb-3",
            ),
            ui.div(
                ui.row(
                    ui.column(
                        6,
                        # Include charts checkbox
                        ui.input_checkbox(
                            "input_ID_tab_portfolios_subtab_reporting_checkbox_include_charts",
                            "Include Charts and Visualizations",
                            value=True,
                        ),
                    ),
                    ui.column(
                        6,
                        # Chart resolution selection
                        ui.input_select(
                            "input_ID_tab_portfolios_subtab_reporting_select_chart_resolution",
                            "Chart Resolution:",
                            choices={
                                "medium": "Medium (Fast)",
                                "high": "High (Recommended)",
                                "ultra": "Ultra (Slow)",
                            },
                            selected="high",
                        ),
                    ),
                ),
                class_="mb-4",
            ),
            class_="card card-body bg-light mb-4",
        ),
        # PDF file save specification section
        ui.div(
            ui.div(
                ui.h5("� PDF Download Filename", class_="text-primary mb-3"),
                # Filename specification with security validation
                ui.div(
                    ui.input_text(
                        "input_ID_tab_portfolios_subtab_reporting_text_download_filename",
                        "PDF Download Filename:",
                        value=Default_Filename,
                        placeholder="Enter secure PDF filename with .pdf extension...",
                        width="100%",
                    ),
                    class_="mb-3",
                ),
                # Filename validation status display
                ui.div(
                    ui.output_text_verbatim(
                        "output_ID_tab_portfolios_subtab_reporting_text_filename_validation",
                    ),
                    class_="small text-muted border rounded p-2 bg-light",
                ),
                class_="card card-body bg-light",
            ),
            class_="mb-4",
        ),
        # Security information panel
        ui.div(
            ui.div(
                ui.h6("🔒 Security Information", class_="text-warning mb-2"),
                ui.tags.ul(
                    ui.tags.li("Filename will be sanitized for security"),
                    ui.tags.li(
                        "Only letters, numbers, spaces, dots, hyphens, underscores and parentheses allowed",
                    ),
                    ui.tags.li("Maximum filename length: 200 characters"),
                    ui.tags.li("Reports will be generated on cloud and available for download"),
                    ui.tags.li("No directory paths - filename only"),
                    class_="small text-muted",
                ),
                class_="alert alert-warning small",
            ),
            class_="mb-3",
        ),
        # PDF generation buttons
        ui.div(
            ui.h5("� PDF Generation", class_="text-primary mb-3"),
            ui.div(
                ui.input_action_button(
                    "input_ID_tab_portfolios_subtab_reporting_btn_generate_pdf",
                    "📄 Generate PDF Report for Download",
                    class_="btn-primary btn-lg mb-3",
                    style="min-width: 350px; padding: 20px 40px; font-size: 18px;",
                ),
                class_="text-center mb-4",
            ),
            ui.div(
                ui.output_ui("output_ID_tab_portfolios_subtab_reporting_ui_pdf_progress"),
                class_="mb-3",
            ),
            # Download section
            ui.div(
                ui.output_ui("output_ID_tab_portfolios_subtab_reporting_ui_download_section"),
                class_="text-center mb-4",
            ),
            class_="card card-body bg-light mb-4",
        ),
        # Status display
        ui.div(
            ui.output_text_verbatim("output_ID_tab_portfolios_subtab_reporting_text_pdf_status"),
            class_="text-center mb-4",
        ),
        # Information panel
        ui.div(
            ui.div(
                ui.h5("📋 Report Information"),
                ui.tags.ul(
                    ui.tags.li("📁 Reports generated on Posit Connect Cloud"),
                    ui.tags.li("📄 Default filename pattern: Report-QWIM-YYYYMMDD.pdf"),
                    ui.tags.li("📊 Includes data from Portfolios and Investors tabs"),
                    ui.tags.li(
                        "📄 Typst: Professional typography with mathematical notation support",
                    ),
                    ui.tags.li("📈 Polars DataFrames integration for enhanced performance"),
                    ui.tags.li("📥 Secure download delivery to your workstation"),
                    ui.tags.li(
                        "🎯 Data sources: Data_Personal_Info_DF, Data_Assets_DF, Data_Goals_DF, Data_Income_DF",
                    ),
                    ui.tags.li("🔒 Filename sanitization for security protection"),
                ),
                class_="card card-body bg-light",
            ),
            class_="mt-4",
        ),
        # Technical status panel
        ui.div(
            ui.div(
                ui.h6("🔧 System Status"),
                ui.div(
                    f"📊 report_QWIM module: {'✅ Available' if REPORT_QWIM_AVAILABLE else '❌ Not Available'}",
                    ui.br(),
                    f"📝 typst module: {'✅ Available' if TYPST_AVAILABLE else '❌ Not Available'}",
                    ui.br(),
                    f"🔄 reactives_shiny utils: {'✅ Available' if DATA_UTILS_AVAILABLE else '❌ Not Available'}",
                    ui.br(),
                    "☁️ Deployment: Posit Connect Cloud",
                    ui.br(),
                    "🔒 Security: Filename sanitization enabled",
                    ui.br(),
                    f"🕒 Status updated: {datetime.datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}",
                    class_="small text-muted",
                ),
                class_="card card-body border-info",
            ),
            class_="mt-3",
        ),
    )


@module.server
def subtab_reporting_server(
    input: typing.Any,
    output: typing.Any,
    session: typing.Any,
    data_utils: dict,
    data_inputs: dict,
    reactives_shiny: dict,
) -> None:
    """Server logic for PDF report generation using Typst report_QWIM.typ with secure download delivery."""
    PDF_GENERATION_TOTAL_STEPS = 12

    # Reactive value to store the generated PDF file path
    generated_pdf_path: reactive.Value[str | None] = reactive.Value(None)
    pdf_status_message: reactive.Value[str] = reactive.Value(
        "Ready to generate PDF reports for download",
    )
    initial_pdf_generation_in_progress = False
    pdf_generation_in_progress: reactive.Value[bool] = reactive.Value(
        initial_pdf_generation_in_progress,
    )
    pdf_generation_progress_pct: reactive.Value[int] = reactive.Value(0)
    pdf_generation_progress_label: reactive.Value[str] = reactive.Value("Waiting to start")
    pdf_generation_step_current: reactive.Value[int] = reactive.Value(0)
    pdf_generation_step_total: reactive.Value[int] = reactive.Value(PDF_GENERATION_TOTAL_STEPS)

    def set_pdf_status(message: str) -> None:
        """Set PDF status message shown in reporting UI."""
        pdf_status_message.set(message)

    def set_pdf_progress(
        progress_pct: int,
        progress_label: str,
        step_current: int | None = None,
        step_total: int | None = None,
    ) -> None:
        """Update PDF generation progress values for dashboard progress bar."""
        Safe_Progress = max(0, min(100, int(progress_pct)))
        pdf_generation_progress_pct.set(Safe_Progress)
        pdf_generation_progress_label.set(progress_label)
        if step_total is not None:
            Safe_Total = max(1, int(step_total))
            pdf_generation_step_total.set(Safe_Total)
        if step_current is not None:
            Safe_Current = max(0, min(int(step_current), pdf_generation_step_total.get()))
            pdf_generation_step_current.set(Safe_Current)

    def get_user_id_from_session() -> str:
        """
        Get user ID from Shiny session or generate default following QWIM standards.

        Returns
        -------
            str: User identifier for PDF generation
        """
        try:
            # Configuration validation - try to get user ID from session
            if hasattr(session, "user_id") and session.user_id:
                User_ID = str(session.user_id)
                print(f"📋 User ID from session.user_id: {User_ID}")
                return User_ID
            if hasattr(session, "get") and callable(session.get):
                Session_User_ID = session.get("user_id")
                if Session_User_ID:
                    User_ID = str(Session_User_ID)
                    print(f"📋 User ID from session.get(): {User_ID}")
                    return User_ID
            elif hasattr(session, "user") and session.user:
                User_ID = str(session.user)
                print(f"📋 User ID from session.user: {User_ID}")
                return User_ID

            # Business logic validation - generate default user ID with timestamp
            Timestamp_String = datetime.datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            Default_User_ID = f"qwim_user_{Timestamp_String}"
            print(f"📋 Generated default User ID: {Default_User_ID}")
            return Default_User_ID

        except Exception as session_error:
            print(f"⚠️ Session user ID error: {session_error}")
            # Defensive programming - fallback to timestamp-based ID
            Timestamp_String = datetime.datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            Fallback_User_ID = f"fallback_user_{Timestamp_String}"
            print(f"📋 Fallback User ID: {Fallback_User_ID}")
            return Fallback_User_ID

    def create_typst_report_with_data(
        typst_template_path: str,
        output_path: str,
        personal_info_data: dict,
        assets_data: dict,
        goals_data: dict,
        income_data: dict,
    ) -> tuple[bool, str]:
        """
        Create Typst report by substituting data into template placeholders.

        Args:
            typst_template_path (str): Path to Typst template file
            output_path (str): Path for generated Typst file with data
            personal_info_data (dict): Personal information data structure
            assets_data (dict): Assets data structure
            goals_data (dict): Goals data structure
            income_data (dict): Income data structure

        Returns
        -------
            Tuple[bool, str]: (success_flag, status_message)
        """
        try:
            # Input validation with early returns
            if not os.path.exists(typst_template_path):
                return False, f"Template file not found: {typst_template_path}"

            # Configuration validation - get template directory and output directory
            Template_Dir = os.path.dirname(typst_template_path)
            Output_Dir = os.path.dirname(output_path)

            # Copy bibliography file if it exists
            Bibliography_Source_Path = os.path.join(Template_Dir, "sample_refs.bib")
            Bibliography_Target_Path = os.path.join(Output_Dir, "sample_refs.bib")

            if os.path.exists(Bibliography_Source_Path):
                try:
                    shutil.copy2(Bibliography_Source_Path, Bibliography_Target_Path)
                    print(f"✅ Copied bibliography file to: {Bibliography_Target_Path}")
                except Exception as copy_error:
                    print(f"⚠️ Warning: Could not copy bibliography file: {copy_error}")
            else:
                print(f"⚠️ Warning: Bibliography file not found at: {Bibliography_Source_Path}")
                # Create empty bibliography file to prevent compilation errors
                try:
                    with open(Bibliography_Target_Path, "w", encoding="utf-8") as empty_bib:
                        empty_bib.write("% Empty bibliography file created automatically\n")
                    print(f"✅ Created empty bibliography file: {Bibliography_Target_Path}")
                except Exception as create_error:
                    print(f"⚠️ Warning: Could not create empty bibliography file: {create_error}")

            if not personal_info_data or not assets_data:
                print("⚠️ Missing personal info or assets data, using fallback template")
                # Use original template without substitutions for defensive programming
                with open(typst_template_path, encoding="utf-8") as template_file:
                    template_content = template_file.read()
                with open(output_path, "w", encoding="utf-8") as output_file:
                    output_file.write(template_content)
                return True, "Template copied without data substitutions"

            # Read the template file
            with open(typst_template_path, encoding="utf-8") as template_file:
                template_content = template_file.read()

            # Define data substitutions with defensive programming
            primary_info = personal_info_data.get("primary", {})
            partner_info = personal_info_data.get("partner", {})
            primary_assets = assets_data.get("primary", {})
            partner_assets = assets_data.get("partner", {})
            combined_assets = assets_data.get("combined", {})
            primary_goals = goals_data.get("primary", {})
            partner_goals = goals_data.get("partner", {})
            combined_goals = goals_data.get("combined", {})
            primary_income = income_data.get("primary", {})
            partner_income = income_data.get("partner", {})
            combined_income = income_data.get("combined", {})

            # Format currency values with defensive programming
            def format_currency(amount: Any) -> str:
                try:
                    return f"\\${float(amount):,.0f}"
                except (ValueError, TypeError):
                    return "\\$0"

            # Create substitution mapping
            substitutions = {
                # Primary Investor Personal Info
                "{{PRIMARY_INVESTOR_NAME}}": str(primary_info.get("name", "Primary Investor")),
                "{{PRIMARY_INVESTOR_AGE_CURRENT}}": str(
                    primary_info.get("age_current", "Not Specified"),
                ),
                "{{PRIMARY_INVESTOR_AGE_RETIREMENT}}": str(
                    primary_info.get("age_retirement", "Not Specified"),
                ),
                "{{PRIMARY_INVESTOR_AGE_INCOME_STARTING}}": str(
                    primary_info.get("age_income_starting", "Not Specified"),
                ),
                "{{PRIMARY_INVESTOR_STATUS_MARITAL}}": str(
                    primary_info.get("status_marital", "Not Specified"),
                ),
                "{{PRIMARY_INVESTOR_GENDER}}": str(primary_info.get("gender", "Not Specified")),
                "{{PRIMARY_INVESTOR_TOLERANCE_RISK}}": str(
                    primary_info.get("tolerance_risk", "Not Specified"),
                ),
                "{{PRIMARY_INVESTOR_STATE}}": str(primary_info.get("state", "Not Specified")),
                "{{PRIMARY_INVESTOR_CODE_ZIP}}": str(primary_info.get("code_zip", "Not Provided")),
                # Partner Investor Personal Info
                "{{PARTNER_INVESTOR_NAME}}": str(partner_info.get("name", "Partner Investor")),
                "{{PARTNER_INVESTOR_AGE_CURRENT}}": str(
                    partner_info.get("age_current", "Not Specified"),
                ),
                "{{PARTNER_INVESTOR_AGE_RETIREMENT}}": str(
                    partner_info.get("age_retirement", "Not Specified"),
                ),
                "{{PARTNER_INVESTOR_AGE_INCOME_STARTING}}": str(
                    partner_info.get("age_income_starting", "Not Specified"),
                ),
                "{{PARTNER_INVESTOR_STATUS_MARITAL}}": str(
                    partner_info.get("status_marital", "Not Specified"),
                ),
                "{{PARTNER_INVESTOR_GENDER}}": str(partner_info.get("gender", "Not Specified")),
                "{{PARTNER_INVESTOR_TOLERANCE_RISK}}": str(
                    partner_info.get("tolerance_risk", "Not Specified"),
                ),
                "{{PARTNER_INVESTOR_STATE}}": str(partner_info.get("state", "Not Specified")),
                "{{PARTNER_INVESTOR_CODE_ZIP}}": str(partner_info.get("code_zip", "Not Provided")),
                # Primary Investor Assets
                "{{PRIMARY_TAXABLE_ASSETS}}": format_currency(primary_assets.get("taxable", 0)),
                "{{PRIMARY_TAX_DEFERRED_ASSETS}}": format_currency(
                    primary_assets.get("tax_deferred", 0),
                ),
                "{{PRIMARY_TAX_FREE_ASSETS}}": format_currency(primary_assets.get("tax_free", 0)),
                "{{PRIMARY_TOTAL_ASSETS}}": format_currency(primary_assets.get("total", 0)),
                # Partner Investor Assets
                "{{PARTNER_TAXABLE_ASSETS}}": format_currency(partner_assets.get("taxable", 0)),
                "{{PARTNER_TAX_DEFERRED_ASSETS}}": format_currency(
                    partner_assets.get("tax_deferred", 0),
                ),
                "{{PARTNER_TAX_FREE_ASSETS}}": format_currency(partner_assets.get("tax_free", 0)),
                "{{PARTNER_TOTAL_ASSETS}}": format_currency(partner_assets.get("total", 0)),
                # Combined Assets
                "{{COMBINED_TAXABLE_ASSETS}}": format_currency(combined_assets.get("taxable", 0)),
                "{{COMBINED_TAX_DEFERRED_ASSETS}}": format_currency(
                    combined_assets.get("tax_deferred", 0),
                ),
                "{{COMBINED_TAX_FREE_ASSETS}}": format_currency(combined_assets.get("tax_free", 0)),
                "{{COMBINED_TOTAL_ASSETS}}": format_currency(combined_assets.get("total", 0)),
                # Primary Investor Goals
                "{{PRIMARY_ESSENTIAL_GOALS}}": format_currency(primary_goals.get("essential", 0)),
                "{{PRIMARY_IMPORTANT_GOALS}}": format_currency(primary_goals.get("important", 0)),
                "{{PRIMARY_ASPIRATIONAL_GOALS}}": format_currency(
                    primary_goals.get("aspirational", 0),
                ),
                "{{PRIMARY_TOTAL_GOALS}}": format_currency(primary_goals.get("total", 0)),
                # Partner Investor Goals
                "{{PARTNER_ESSENTIAL_GOALS}}": format_currency(partner_goals.get("essential", 0)),
                "{{PARTNER_IMPORTANT_GOALS}}": format_currency(partner_goals.get("important", 0)),
                "{{PARTNER_ASPIRATIONAL_GOALS}}": format_currency(
                    partner_goals.get("aspirational", 0),
                ),
                "{{PARTNER_TOTAL_GOALS}}": format_currency(partner_goals.get("total", 0)),
                # Combined Goals
                "{{COMBINED_ESSENTIAL_GOALS}}": format_currency(combined_goals.get("essential", 0)),
                "{{COMBINED_IMPORTANT_GOALS}}": format_currency(combined_goals.get("important", 0)),
                "{{COMBINED_ASPIRATIONAL_GOALS}}": format_currency(
                    combined_goals.get("aspirational", 0),
                ),
                "{{COMBINED_TOTAL_GOALS}}": format_currency(combined_goals.get("total", 0)),
                # Primary Investor Income
                "{{PRIMARY_SOCIAL_SECURITY_INCOME}}": format_currency(
                    primary_income.get("social_security", 0),
                ),
                "{{PRIMARY_PENSION_INCOME}}": format_currency(primary_income.get("pension", 0)),
                "{{PRIMARY_ANNUITY_EXISTING_INCOME}}": format_currency(
                    primary_income.get("annuity_existing", 0),
                ),
                "{{PRIMARY_OTHER_INCOME}}": format_currency(primary_income.get("other", 0)),
                "{{PRIMARY_TOTAL_INCOME}}": format_currency(primary_income.get("total", 0)),
                # Partner Investor Income
                "{{PARTNER_SOCIAL_SECURITY_INCOME}}": format_currency(
                    partner_income.get("social_security", 0),
                ),
                "{{PARTNER_PENSION_INCOME}}": format_currency(partner_income.get("pension", 0)),
                "{{PARTNER_ANNUITY_EXISTING_INCOME}}": format_currency(
                    partner_income.get("annuity_existing", 0),
                ),
                "{{PARTNER_OTHER_INCOME}}": format_currency(partner_income.get("other", 0)),
                "{{PARTNER_TOTAL_INCOME}}": format_currency(partner_income.get("total", 0)),
                # Combined Income
                "{{COMBINED_SOCIAL_SECURITY_INCOME}}": format_currency(
                    combined_income.get("social_security", 0),
                ),
                "{{COMBINED_PENSION_INCOME}}": format_currency(combined_income.get("pension", 0)),
                "{{COMBINED_ANNUITY_EXISTING_INCOME}}": format_currency(
                    combined_income.get("annuity_existing", 0),
                ),
                "{{COMBINED_OTHER_INCOME}}": format_currency(combined_income.get("other", 0)),
                "{{COMBINED_TOTAL_INCOME}}": format_currency(combined_income.get("total", 0)),
            }

            # Perform substitutions with defensive programming
            processed_content = template_content
            for placeholder, value in substitutions.items():
                try:
                    processed_content = processed_content.replace(placeholder, value)
                except Exception as substitution_error:
                    print(f"⚠️ Error substituting {placeholder}: {substitution_error}")
                    continue

            # Write the processed content to output file
            with open(output_path, "w", encoding="utf-8") as output_file:
                output_file.write(processed_content)

            print(f"✅ Successfully created Typst report with data: {output_path}")
            return True, "Typst report with data created successfully"

        except Exception as processing_error:
            return False, f"Error processing Typst template: {processing_error}"

    def copy_typst_dependencies_to_temp_directory(
        template_directory: str,
        temp_directory: str,
    ) -> None:
        """
        Copy Typst template dependencies to temporary directory following QWIM standards.

        Args:
            template_directory (str): Source template directory
            temp_directory (str): Target temporary directory
        """
        try:
            # Input validation with early returns
            if not os.path.exists(template_directory):
                print(f"⚠️ Template directory not found: {template_directory}")
                return

            if not os.path.exists(temp_directory):
                print(f"⚠️ Temporary directory not found: {temp_directory}")
                return

            Template_Dir_Path = Path(template_directory)
            Temp_Dir_Path = Path(temp_directory)

            # Copy root-level JSON data files required by report_QWIM.typ
            Root_JSON_Files = ["client_info.json", "report_metadata.json"]
            for Json_Name in Root_JSON_Files:
                Source_Json = Template_Dir_Path / Json_Name
                if Source_Json.exists():
                    shutil.copy2(Source_Json, Temp_Dir_Path / Json_Name)
                    print(f"✅ Copied JSON data file: {Json_Name}")
                else:
                    print(f"ℹ️ Optional JSON file not found: {Json_Name}")

            # Copy JSON subdirectories required by report_QWIM.typ
            Json_Subdirs = ["inputs_json", "outputs_json"]
            for Subdir_Name in Json_Subdirs:
                Source_Subdir = Template_Dir_Path / Subdir_Name
                if Source_Subdir.is_dir():
                    Dest_Subdir = Temp_Dir_Path / Subdir_Name
                    Dest_Subdir.mkdir(exist_ok=True)
                    for Json_File in Source_Subdir.iterdir():
                        if Json_File.is_file():
                            shutil.copy2(Json_File, Dest_Subdir / Json_File.name)
                    print(f"✅ Copied JSON subdirectory: {Subdir_Name}/")
                else:
                    print(f"ℹ️ JSON subdirectory not found: {Subdir_Name}/")

            # Copy outputs_images subdirectory (SVGs used by report_QWIM.typ)
            Source_Images_Dir = Template_Dir_Path / "outputs_images"
            if Source_Images_Dir.is_dir():
                Dest_Images_Dir = Temp_Dir_Path / "outputs_images"
                Dest_Images_Dir.mkdir(exist_ok=True)
                for Img_File in Source_Images_Dir.iterdir():
                    if Img_File.is_file():
                        shutil.copy2(Img_File, Dest_Images_Dir / Img_File.name)
                print("✅ Copied outputs_images/ directory")
            else:
                print("ℹ️ outputs_images/ directory not found")

            # Configuration validation - list of root-level file dependencies to copy
            Dependencies_To_Copy = [
                "sample_refs.bib",
                "*.png",
                "*.jpg",
                "*.jpeg",
            ]

            # Business logic validation - copy each dependency type
            for Dependency_Pattern in Dependencies_To_Copy:
                try:
                    if "*" in Dependency_Pattern:
                        # Handle wildcard patterns
                        import glob

                        Pattern_Path = os.path.join(template_directory, Dependency_Pattern)
                        Matching_Files = glob.glob(Pattern_Path)

                        for Source_File in Matching_Files:
                            if os.path.isfile(Source_File):
                                Target_File = os.path.join(
                                    temp_directory,
                                    os.path.basename(Source_File),
                                )
                                shutil.copy2(Source_File, Target_File)
                                print(f"✅ Copied dependency: {os.path.basename(Source_File)}")
                    else:
                        # Handle specific files
                        Source_File = os.path.join(template_directory, Dependency_Pattern)
                        if os.path.exists(Source_File):
                            Target_File = os.path.join(temp_directory, Dependency_Pattern)
                            shutil.copy2(Source_File, Target_File)
                            print(f"✅ Copied dependency: {Dependency_Pattern}")
                        else:
                            print(f"ℹ️ Optional dependency not found: {Dependency_Pattern}")

                            # Create empty .bib file if it's the bibliography
                            if Dependency_Pattern == "sample_refs.bib":
                                Empty_Bib_Path = os.path.join(temp_directory, Dependency_Pattern)
                                with open(Empty_Bib_Path, "w", encoding="utf-8") as empty_bib:
                                    empty_bib.write(
                                        "% Empty bibliography file created automatically\n",
                                    )
                                print(f"✅ Created empty bibliography: {Dependency_Pattern}")

                except Exception as copy_error:
                    print(f"⚠️ Error copying dependency {Dependency_Pattern}: {copy_error}")
                    continue

        except Exception as dependencies_error:
            print(f"⚠️ Error copying Typst dependencies: {dependencies_error}")

    def show_info_notification_internal(
        message: str,
    ) -> None:
        """
        Show info notification following QWIM standards.

        Args:
            message (str): Information message to display
        """
        try:
            print(f"ℹ️ INFO: {message}")
            set_pdf_status(f"ℹ️ {message}")
            # Show Shiny notification popup
            with contextlib.suppress(Exception):
                ui.notification_show(
                    message,
                    type="message",
                    duration=8,
                )
        except Exception as notification_error:
            print(f"⚠️ Notification error: {notification_error}")

    def show_success_notification_internal(
        message: str,
    ) -> None:
        """
        Show success notification following QWIM standards.

        Args:
            message (str): Success message to display
        """
        try:
            print(f"✅ SUCCESS: {message}")
            set_pdf_status(f"✅ {message}")
            # Show Shiny notification popup
            with contextlib.suppress(Exception):
                ui.notification_show(
                    message,
                    type="message",
                    duration=8,
                )
        except Exception as notification_error:
            print(f"⚠️ Notification error: {notification_error}")

    def show_error_notification_internal(
        message: str,
    ) -> None:
        """
        Show error notification as a popup modal window following QWIM standards.

        Uses ``ui.modal_show`` to display a prominent modal dialog so the user
        clearly sees which chart or table failed and can take corrective action.

        Args:
            message (str): Error message to display
        """
        try:
            print(f"❌ ERROR: {message}")
            set_pdf_status(f"❌ {message}")
            # Show a modal popup so the user sees the error prominently
            try:
                # Split message into lines for readable HTML rendering
                lines = message.split("\n")
                modal_body_parts: list[Any] = [
                    ui.p(
                        ui.strong("PDF Report Generation Error"),
                        style="color: #b91c1c; font-size: 1.1em;",
                    ),
                ]
                for line in lines:
                    stripped = line.strip()
                    if stripped:
                        modal_body_parts.append(ui.p(stripped))
                modal_body_parts.append(
                    ui.p(
                        ui.em(
                            "Go back to the relevant dashboard tabs and ensure "
                            "all charts and tables display valid data, then "
                            "regenerate the report.",
                        ),
                        style="margin-top: 1em; color: #374151;",
                    ),
                )
                error_modal = ui.modal(
                    *modal_body_parts,
                    title="❌ Report Generation Failed",
                    easy_close=True,
                    footer=ui.modal_button("Close", class_="btn-danger"),
                )
                ui.modal_show(error_modal)
            except Exception:
                # Fallback to notification if modal fails
                with contextlib.suppress(Exception):
                    ui.notification_show(
                        message,
                        type="error",
                        duration=15,
                    )
        except Exception as notification_error:
            print(f"⚠️ Notification error: {notification_error}")

    def compile_typst_report_to_pdf(
        typst_file_path: str,
        output_pdf_path: str,
    ) -> tuple[bool, str]:
        """
        Compile Typst report to PDF following QWIM standards.

        Args:
            typst_file_path (str): Path to Typst source file
            output_pdf_path (str): Path for output PDF file

        Returns
        -------
            Tuple[bool, str]: (success_flag, status_message)
        """
        try:
            # Input validation with early returns
            if not typst_file_path or not os.path.exists(typst_file_path):
                return False, f"Typst source file not found: {typst_file_path}"

            # Configuration validation - check Typst availability
            if not TYPST_AVAILABLE:
                return False, "Typst module not available for compilation"

            # Business logic validation - compile Typst to PDF
            typst.compile(str(typst_file_path), output=str(output_pdf_path))  # pyright: ignore[reportPossiblyUnboundVariable]

            # Defensive programming - verify output file was created
            if not os.path.exists(output_pdf_path):
                return False, "PDF compilation completed but output file not found"

            # Check file size to ensure it's not empty
            File_Size = os.path.getsize(output_pdf_path)
            if File_Size == 0:
                return False, "PDF compilation produced empty file"

            return True, f"Typst PDF compilation successful ({File_Size} bytes)"

        except Exception as compilation_error:
            return False, f"Typst compilation error: {compilation_error}"

    @output
    @render.text
    def output_ID_tab_portfolios_subtab_reporting_text_pdf_status() -> str:
        """Render PDF generation status following QWIM standards."""
        return pdf_status_message.get()

    @output
    @render.ui
    def output_ID_tab_portfolios_subtab_reporting_ui_pdf_progress():
        """Render PDF generation progress indicator in dashboard."""
        if not pdf_generation_in_progress.get():
            return ui.div()

        Progress_Percent = pdf_generation_progress_pct.get()
        Progress_Label = pdf_generation_progress_label.get()
        Current_Step = pdf_generation_step_current.get()
        Total_Steps = pdf_generation_step_total.get()

        return ui.div(
            ui.div(
                ui.tags.div(
                    ui.tags.span(class_="spinner-border spinner-border-sm me-2", role="status"),
                    ui.tags.strong(
                        f"Generating PDF... Step {Current_Step} of {Total_Steps} ({Progress_Percent}%)",
                    ),
                    class_="mb-2",
                ),
                ui.tags.div(
                    ui.tags.div(
                        style=f"width: {Progress_Percent}%;",
                        class_="progress-bar progress-bar-striped progress-bar-animated",
                        role="progressbar",
                        **{
                            "aria-valuenow": str(Progress_Percent),
                            "aria-valuemin": "0",
                            "aria-valuemax": "100",
                        },
                    ),
                    class_="progress mb-2",
                    style="height: 18px;",
                ),
                ui.tags.small(Progress_Label, class_="text-muted"),
            ),
            class_="alert alert-info",
        )

    @output
    @render.text
    def output_ID_tab_portfolios_subtab_reporting_text_filename_validation() -> str | None:
        """Render filename validation status following QWIM standards."""
        try:
            # Get current filename input
            Current_Filename = (
                input.input_ID_tab_portfolios_subtab_reporting_text_download_filename()
            )

            if not Current_Filename:
                return "📄 Enter a filename to validate"

            # Validate filename
            Is_Valid, Sanitized_Name, Validation_Message = sanitize_filename_for_security(
                Current_Filename,
            )

            if Is_Valid:
                if Sanitized_Name == Current_Filename:
                    return f"✅ Filename is valid and secure: {Sanitized_Name}"
                return f"⚠️ Filename will be sanitized to: {Sanitized_Name}"
            return f"❌ Invalid filename: {Validation_Message}"

        except Exception as validation_error:
            print(f"⚠️ Error validating filename: {validation_error}")
            return "❌ Error validating filename"

    @output
    @render.ui
    def output_ID_tab_portfolios_subtab_reporting_ui_download_section():
        """Render download section UI following QWIM standards."""
        try:
            Current_PDF_Path = generated_pdf_path.get()

            if Current_PDF_Path and os.path.exists(Current_PDF_Path):
                # Extract filename for display
                PDF_Filename = os.path.basename(Current_PDF_Path)

                return ui.div(
                    ui.h5("📥 PDF Ready", class_="text-success mb-3"),
                    ui.p(f"Report generated successfully: {PDF_Filename}", class_="mb-3"),
                    ui.download_button(
                        "btn_download_pdf_report",
                        "📄 Download PDF Report",
                        class_="btn-success btn-lg",
                        style="min-width: 300px; padding: 15px 30px; font-size: 16px;",
                    ),
                    class_="alert alert-success",
                )
            return ui.div()

        except Exception as download_ui_error:
            print(f"⚠️ Error rendering download UI: {download_ui_error}")
            return ui.div()

    @output
    @render.download(filename=lambda: os.path.basename(generated_pdf_path.get() or "report.pdf"))
    def btn_download_pdf_report():
        """Handle PDF report download following QWIM standards."""
        try:
            Current_PDF_Path = generated_pdf_path.get()

            # Input validation with early returns
            if not Current_PDF_Path or not os.path.exists(Current_PDF_Path):
                print("❌ No PDF file available for download")
                return

            # Configuration validation - ensure file is readable
            if not os.access(Current_PDF_Path, os.R_OK):
                print("❌ PDF file is not readable")
                return

            # Business logic validation - yield file bytes for download
            print(f"📥 Initiating PDF download for: {Current_PDF_Path}")
            with open(Current_PDF_Path, "rb") as _pdf_fh:
                yield _pdf_fh.read()

        except Exception as download_error:
            print(f"❌ Error handling PDF download: {download_error}")

    @reactive.effect
    @reactive.event(input.input_ID_tab_portfolios_subtab_reporting_btn_generate_pdf)
    def observer_generate_pdf_report() -> None:
        """Handle PDF report generation using Typst report_QWIM.typ following QWIM standards."""
        # Initialize before try so it is always bound in the except handler
        Temp_Directory: str = ""
        try:
            generated_pdf_path.set(None)
            pdf_generation_in_progress.set(True)
            pdf_generation_step_total.set(PDF_GENERATION_TOTAL_STEPS)
            set_pdf_progress(
                5,
                "Preparing generation workflow",
                step_current=1,
                step_total=PDF_GENERATION_TOTAL_STEPS,
            )
            set_pdf_status("⏳ Generating PDF report...")

            print("=" * 80)
            print("� QWIM TYPST PDF REPORT GENERATION (CLOUD)")
            print("=" * 80)
            set_pdf_progress(8, "Checking Typst runtime availability", step_current=2)

            # Configuration validation - check Typst availability
            if not TYPST_AVAILABLE:
                Error_Message = "Typst module not available. Please check installation."
                print(f"❌ {Error_Message}")
                show_error_notification_internal(Error_Message)
                return

            # Input validation with early returns - get report title
            Report_Title = input.input_ID_tab_portfolios_subtab_reporting_text_report_title()
            if not Report_Title or len(Report_Title.strip()) < 3:
                Report_Title = "QWIM Portfolio and Investor Analysis Report"
                print(f"📋 Using default report title: {Report_Title}")
            else:
                print(f"📋 User-specified report title: {Report_Title}")
            set_pdf_progress(14, "Validating report title and output filename", step_current=3)

            # Business logic validation - get and validate filename
            User_Filename = input.input_ID_tab_portfolios_subtab_reporting_text_download_filename()

            # Configuration validation - sanitize and validate filename
            if not User_Filename or len(User_Filename.strip()) == 0:
                Current_Date_String = datetime.datetime.now(UTC).strftime("%Y%m%d")
                User_Filename = f"Report-QWIM-{Current_Date_String}.pdf"
                print(f"📋 Using default filename: {User_Filename}")

            # Security validation - sanitize filename
            Is_Valid_Filename, Sanitized_Filename, Validation_Message = (
                sanitize_filename_for_security(User_Filename)
            )
            if not Is_Valid_Filename:
                Error_Message = f"Invalid filename: {Validation_Message}"
                print(f"❌ {Error_Message}")
                show_error_notification_internal(Error_Message)
                return

            print(f"📋 Sanitized filename: {Sanitized_Filename}")
            set_pdf_progress(18, "Validating inputs and preparing report settings", step_current=3)

            # Business logic validation - get user configuration
            User_ID = get_user_id_from_session()

            # Get chart settings from UI inputs
            Include_Charts = (
                input.input_ID_tab_portfolios_subtab_reporting_checkbox_include_charts()
            )
            Chart_Resolution = (
                input.input_ID_tab_portfolios_subtab_reporting_select_chart_resolution()
            )

            # Define Typst source file path
            Typst_Source_Path = (
                Path(__file__).resolve().parent.parent / "reporting" / "report_QWIM.typ"
            )

            # Input validation - check if Typst source file exists
            if not Typst_Source_Path.exists():
                Error_Message = f"Typst source file not found: {Typst_Source_Path}"
                print(f"❌ {Error_Message}")
                show_error_notification_internal(Error_Message)
                return

            # Create secure temporary directory for cloud environment
            Temp_Directory = create_secure_temp_directory()
            set_pdf_progress(24, "Created secure temporary workspace", step_current=4)

            # Create a temporary Typst file path (simple copy, not substitution)
            Temp_Typst_Path = Path(Temp_Directory) / "report_QWIM.typ"
            Output_PDF_Path = Path(Temp_Directory) / Sanitized_Filename

            print("📋 Typst report configuration:")
            print(f"   Title: {Report_Title}")
            print(f"   User ID: {User_ID}")
            print(f"   Original filename: {User_Filename}")
            print(f"   Sanitized filename: {Sanitized_Filename}")
            print(f"   Typst source: {Typst_Source_Path}")
            print(f"   Temporary directory: {Temp_Directory}")
            print(f"   Temp Typst file: {Temp_Typst_Path}")
            print(f"   Output PDF path: {Output_PDF_Path}")
            print(f"   Include charts: {Include_Charts}")
            print(f"   Chart resolution: {Chart_Resolution}")

            # Show progress notification
            show_info_notification_internal(
                "Exporting report data and charts from dashboard... Please wait.",
            )
            set_pdf_progress(30, "Exporting report JSON data", step_current=5)

            # --- Step 1: Export JSON data files to the reporting directory ---
            # This generates report_metadata.json, client_info.json, inputs_json/*,
            # outputs_json/* from the live dashboard state so Typst can read them.
            try:
                export_all_report_data(reactives_shiny)
                print("✅ Exported all report JSON data files")
            except Exception as data_export_error:
                _logger.warning(
                    "JSON data export failed (continuing with any existing files): %s",
                    data_export_error,
                )
                print(f"⚠️ JSON data export warning: {data_export_error}")

            set_pdf_progress(40, "JSON export completed", step_current=5)
            set_pdf_progress(46, "Exporting chart files", step_current=6)

            # --- Step 2: Export SVG charts ---
            # First generate static plotnine fallback charts, then overlay with
            # live Plotly figures from Visual_Objects_Shiny (if available).
            if Include_Charts:
                try:
                    export_all_report_plots(reactives_shiny)
                    print("✅ Exported plotnine fallback SVG charts")
                except Exception as plot_export_error:
                    _logger.warning(
                        "Plotnine chart export failed (continuing): %s",
                        plot_export_error,
                    )
                    print(f"⚠️ Plotnine chart export warning: {plot_export_error}")
            else:
                print("ℹ️ Chart export disabled by user input")

            # Export live Plotly figures from Visual_Objects_Shiny to SVG
            # (overwrites plotnine fallbacks with live charts when available)
            set_pdf_progress(54, "Exporting live dashboard visuals", step_current=7)
            _logger.info("Exporting live chart SVGs for report pipeline")
            export_visual_objects_to_svg(reactives_shiny)

            # --- Step 2b: Guarantee all SVG files exist (create placeholders) ---
            set_pdf_progress(62, "Verifying report image assets", step_current=8)
            svg_status = ensure_all_svg_files_exist()
            placeholder_charts = [
                key.replace("_", " ").title()
                for key, status in svg_status.items()
                if status == "placeholder"
            ]

            # --- Step 2c: Validate data quality ---
            data_issues = validate_report_data_quality()

            set_pdf_progress(70, "Running report data quality checks", step_current=9)

            # --- Step 2d: Show pre-flight warnings in dashboard ---
            warning_lines: list[str] = []
            if placeholder_charts:
                warning_lines.append(
                    "Charts unavailable (placeholder used): " + ", ".join(placeholder_charts),
                )
            warning_lines.extend(
                f"{section_name}: {issue_text}"
                for section_name, section_issues in data_issues.items()
                for issue_text in section_issues
            )

            if warning_lines:
                combined_warning = (
                    "⚠️ Report data quality warnings:\n• "
                    + "\n• ".join(warning_lines)
                    + "\n\nThe PDF will be generated with placeholder content for "
                    "missing items. To remove placeholders, go back to the "
                    "relevant dashboard tabs and ensure charts/tables display "
                    "valid data, then regenerate the report."
                )
                print(combined_warning)
                # Show a warning modal so the user sees exactly which items are missing
                try:
                    warning_body: list[Any] = [
                        ui.p(
                            ui.strong("Some report sections have missing or incomplete data:"),
                            style="color: #b45309; font-size: 1.05em;",
                        ),
                    ]
                    warning_body.extend(
                        ui.p(f"• {wl}", style="margin: 0.2em 0;") for wl in warning_lines
                    )
                    warning_body.append(
                        ui.p(
                            ui.em(
                                "The PDF will still be generated with placeholder content. "
                                "To remove placeholders, go back to the relevant dashboard "
                                "tabs and ensure charts/tables display valid data, then "
                                "regenerate the report.",
                            ),
                            style="margin-top: 1em; color: #374151;",
                        ),
                    )
                    warn_modal = ui.modal(
                        *warning_body,
                        title="⚠️ Report Data Quality Warnings",
                        easy_close=True,
                        footer=ui.modal_button("Continue Anyway", class_="btn-warning"),
                    )
                    ui.modal_show(warn_modal)
                except Exception:
                    show_info_notification_internal(combined_warning)
            set_pdf_progress(74, "Data quality pre-flight completed", step_current=9)

            # --- Step 3: Copy all dependencies to temp directory ---
            # Now that JSON + SVG files exist in the reporting directory, copy them.
            Template_Directory = str(Typst_Source_Path.parent)
            set_pdf_progress(80, "Copying Typst dependencies", step_current=10)
            copy_typst_dependencies_to_temp_directory(Template_Directory, Temp_Directory)

            # Copy the Typst template itself to the temp directory
            shutil.copy2(str(Typst_Source_Path), str(Temp_Typst_Path))
            print(f"✅ Copied Typst template to: {Temp_Typst_Path}")
            set_pdf_progress(86, "Preparing Typst compilation", step_current=10)

            # Show progress notification for Typst compilation
            show_info_notification_internal("Compiling Typst report to PDF... Please wait.")
            set_pdf_progress(92, "Compiling PDF with Typst", step_current=11)

            # Compile Typst report to PDF
            try:
                Compilation_Success, Compilation_Message = compile_typst_report_to_pdf(
                    typst_file_path=str(Temp_Typst_Path),
                    output_pdf_path=str(Output_PDF_Path),
                )
            except Exception as compilation_error:
                Error_Message = f"Error during Typst compilation: {compilation_error}"
                print(f"❌ {Error_Message}")
                show_error_notification_internal(Error_Message)
                cleanup_temp_files(Temp_Directory)
                return

            # Business logic validation - handle compilation result
            if Compilation_Success and Output_PDF_Path.exists():
                # Get file size for display
                File_Size = Output_PDF_Path.stat().st_size
                Success_Message_Display = f"Typst PDF report generated successfully! Ready for download: {Sanitized_Filename}"
                print(f"✅ {Success_Message_Display}")
                print(f"📁 File path: {Output_PDF_Path}")
                print(f"📄 Status: {Compilation_Message}")
                print(f"📊 File size: {File_Size:,} bytes")

                # Store the PDF path for download
                generated_pdf_path.set(str(Output_PDF_Path))
                set_pdf_progress(100, "PDF generation completed successfully", step_current=12)
                set_pdf_status(f"✅ Report generated successfully: {Sanitized_Filename}")

                show_success_notification_internal("Typst PDF report ready for download!")

                # Schedule cleanup after a delay (allow time for download)
                def delayed_cleanup() -> None:
                    import time

                    time.sleep(300)  # Wait 5 minutes before cleanup
                    cleanup_temp_files(Temp_Directory)

                cleanup_thread = threading.Thread(target=delayed_cleanup, daemon=True)
                cleanup_thread.start()

            else:
                # Build a helpful error message that names the specific issue
                Hint_Lines: list[str] = [f"Typst PDF compilation failed: {Compilation_Message}"]
                if placeholder_charts:
                    Hint_Lines.append(
                        "Charts with placeholder SVGs: " + ", ".join(placeholder_charts),
                    )
                if data_issues:
                    Hint_Lines.extend(
                        f"  {sect}: {m}" for sect, msgs in data_issues.items() for m in msgs
                    )
                Hint_Lines.append(
                    "Go back to the relevant dashboard tabs and ensure all charts "
                    "and tables display valid data, then regenerate the report.",
                )
                Error_Message = "\n".join(Hint_Lines)
                print(f"❌ {Error_Message}")
                set_pdf_progress(100, "PDF generation failed", step_current=12)
                show_error_notification_internal(Error_Message)
                cleanup_temp_files(Temp_Directory)

        except Exception as exc_error:
            Error_Message = f"PDF report generation observer error: {exc_error}"
            print(f"❌ {Error_Message}")
            import traceback

            traceback.print_exc()
            set_pdf_progress(100, "PDF generation failed", step_current=12)
            show_error_notification_internal(Error_Message)
            # Clean up on any error
            with contextlib.suppress(BaseException):
                cleanup_temp_files(Temp_Directory)
        finally:
            pdf_generation_in_progress.set(False)
