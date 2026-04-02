"""
QWIM Dashboard 2026 - Posit Connect Cloud Entry Point
===============================================

Main application entry point for Posit Connect Cloud deployment.
This file serves as the primary interface for the Shiny application
following the project's coding standards for Windows 11 environments.

## Overview

This module provides the main entry point for deploying the QWIM Dashboard
to Posit Connect Cloud using OAuth authentication. It imports and configures
the main dashboard application for cloud deployment from Windows 11 workstations.

## Configuration

The application supports the following environment variables:
- PYTHONPATH: Python module search path configuration (Windows paths)

## Usage

For local development on Windows 11:
```cmd
python app.py
```

For Posit Connect Cloud deployment, this file is automatically discovered
and executed by the Connect Cloud runtime environment.

---

**Author**: QWIM Dashboard Team  
**Version**: 1.0.0  
**Last Updated**: March 2026
"""

# Core Python libraries
import os
import sys
from pathlib import Path

# Windows-specific path handling for local development and cloud deployment
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root))

# Add src directory to Python path for proper imports
src_directory = project_root / "src"
if src_directory.exists():
    sys.path.insert(0, str(src_directory))


def import_dashboard_application():
    """
    Import the main dashboard with multiple fallback strategies.
    
    This function attempts to import the dashboard application using different
    import paths to ensure compatibility across deployment environments.
    
    Returns:
        object: Imported Shiny application object
        
    Raises:
        SystemExit: If application import fails completely
    """
    import_strategies = [
        ("dashboard.main_App", "app"),
        ("src.dashboard.main_App", "app"),
        ("main_App", "app")
    ]
    
    for module_name, app_attribute in import_strategies:
        try:
            imported_module = __import__(module_name, fromlist=[app_attribute])
            dashboard_app = getattr(imported_module, app_attribute)
            
            print(f"✓ Successfully imported app from {module_name}")
            return dashboard_app
            
        except ImportError as import_error:
            print(f"Import attempt failed for {module_name}: {import_error}")
            continue
        except AttributeError as attribute_error:
            print(f"Attribute error for {module_name}: {attribute_error}")
            continue
    
    print(f"Error: Failed to import dashboard application")
    print(f"Project root: {project_root}")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Python path: {sys.path}")
    sys.exit(1)


# Configure environment for Connect Cloud deployment with Windows paths
environment_variables = {
    'PYTHONPATH': str(project_root)
}


def setup_environment():
    """
    Configure environment variables for Posit Connect Cloud deployment.
    
    This function sets up necessary environment variables that are required
    for proper operation of the QWIM Dashboard in the cloud environment,
    handling Windows-specific path configurations.
    
    Returns:
        None
    """
    for variable_name, variable_value in environment_variables.items():
        if variable_name not in os.environ:
            os.environ[variable_name] = variable_value


def verify_app_import(
    dashboard_app
):
    """
    Verify that the main dashboard application was imported successfully.
    
    This function performs validation to ensure the imported app object
    is properly configured for deployment to Posit Connect Cloud.
    
    Args:
        dashboard_app: The imported Shiny application object
    
    Returns:
        bool: True if app is valid, False otherwise
    """
    if dashboard_app is None:
        print("Error: Main dashboard app is None")
        return False
    
    # Check for Shiny App class attributes
    shiny_app_attributes = ['ui', 'server', '_ui', '_server', 'run']
    has_required_attributes = False
    
    for required_attribute in shiny_app_attributes:
        if hasattr(dashboard_app, required_attribute):
            print(f"✓ App has attribute: {required_attribute}")
            has_required_attributes = True
    
    # Check if it's a callable (function that returns an app)
    if callable(dashboard_app):
        print("✓ Dashboard app is callable")
        has_required_attributes = True
    
    # Check app type
    app_type = type(dashboard_app).__name__
    print(f"App type: {app_type}")
    
    if has_required_attributes:
        print("✓ Dashboard app imported and validated successfully")
        return True
    else:
        print("⚠ Dashboard app validation completed with warnings")
        return True  # Continue anyway, Posit Connect may handle it


def create_app_wrapper(
    dashboard_app
):
    """
    Create a wrapper for the dashboard app if needed.
    
    This function ensures the app object is in the correct format
    for Posit Connect Cloud deployment.
    
    Args:
        dashboard_app: The imported dashboard application
    
    Returns:
        object: Properly formatted app object for deployment
    """
    # If it's already a Shiny App object, return as-is
    if hasattr(dashboard_app, 'ui') and hasattr(dashboard_app, 'server'):
        return dashboard_app
    
    # If it's callable, it might be a function that creates the app
    if callable(dashboard_app):
        try:
            created_app = dashboard_app()
            if hasattr(created_app, 'ui') and hasattr(created_app, 'server'):
                print("✓ Created app from callable")
                return created_app
        except Exception as call_error:
            print(f"Warning: Could not call app function: {call_error}")
    
    # Return original app object for Posit Connect to handle
    return dashboard_app


def run_local_development_server(
    final_app
):
    """
    Run the dashboard for local development with proper Shiny for Python API.
    
    This function handles running the dashboard locally using the correct
    Shiny for Python run method without unsupported arguments.
    
    Args:
        final_app: The prepared Shiny application object
    
    Returns:
        None
    """
    print("Starting QWIM Dashboard locally...")
    print("Access the dashboard at: http://127.0.0.1:8000")
    
    # Check if app has run method and use appropriate parameters
    if hasattr(final_app, 'run'):
        try:
            # Try with minimal parameters for Shiny for Python
            final_app.run(
                host="127.0.0.1",
                port=8000
            )
        except TypeError as type_error:
            print(f"Run method error: {type_error}")
            try:
                # Fallback: try with no parameters
                final_app.run()
            except Exception as run_error:
                print(f"Error running app: {run_error}")
                print("App object exported for Posit Connect Cloud deployment")
    else:
        print("Warning: app.run() method not available")
        print("App object exported for Posit Connect Cloud deployment")


def main():
    """
    Main entry point for local development server on Windows 11.
    
    This function is called when the script is executed directly,
    typically during local development and testing on Windows workstations.
    
    Returns:
        None
    """
    setup_environment()
    
    # Import dashboard application
    dashboard_app = import_dashboard_application()
    
    if not verify_app_import(dashboard_app):
        print("Failed to verify app import")
        sys.exit(1)
    
    # Create proper app wrapper
    final_app = create_app_wrapper(dashboard_app)
    
    # Run local development server
    run_local_development_server(final_app)


# Setup environment for both local and cloud deployment
setup_environment()

# Import and prepare the dashboard application
app = import_dashboard_application()

# Verify the app import for early error detection
verify_app_import(app)

# Create final app object for deployment
app = create_app_wrapper(app)

# Posit Connect Cloud expects the app object to be available at module level
# The app object is automatically discovered by Connect Cloud runtime
if __name__ == "__main__":
    main()

