"""Enhanced UI Components Module.

This module provides comprehensive enhanced UI components for the QWIM financial
dashboard system. It includes professional styling, validation, and interactive
components that maintain consistent design patterns across the application.

The module includes:
- Enhanced input components with validation
- Professional card sections and layouts
- Summary display components
- Interactive form elements
- Consistent styling and theming
- Accessibility features

Key Features:
    - Professional Bootstrap-based styling
    - Comprehensive input validation
    - Consistent component variants and themes
    - Responsive design patterns
    - Enhanced user experience elements
    - Integration with Shiny framework

Dependencies:
    - shiny: Web application framework for Python reactive programming
    - typing: Type hints and annotations for enhanced code clarity
    - enum: Enumeration support for component variants

Architecture:
    The module implements a component-based architecture with reusable UI
    elements that follow consistent design patterns and validation standards.
    All components support theming, validation, and accessibility features.

Author:
    QWIM Development Team

Version:
    0.5.1

Last Modified:
    2026-03-01

"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from shiny import ui


if TYPE_CHECKING:
    from htmltools import Tag


class ComponentVariant(Enum):
    """Enumeration of component variants for consistent styling."""

    PRIMARY = "primary"
    SECONDARY = "secondary"
    SUCCESS = "success"
    DANGER = "danger"
    WARNING = "warning"
    INFO = "info"
    LIGHT = "light"
    DARK = "dark"


def validate_component_identifier(
    component_id: str,
    component_type: str = "component",
) -> str:
    """Validate and sanitize component identifier following project standards.

    Args:
        component_id: The component identifier to validate
        component_type: Type of component for error messages

    Returns
    -------
        str: Validated and sanitized component identifier

    Raises
    ------
        ValueError: If component_id is invalid or empty
    """
    # Input validation with early returns
    if not isinstance(component_id, str):
        raise ValueError(f"{component_type} ID must be a string, got {type(component_id).__name__}")

    if not component_id or len(component_id.strip()) == 0:
        raise ValueError(f"{component_type} ID cannot be empty")

    # Configuration validation - sanitize identifier
    sanitized_id = component_id.strip()

    # Business logic validation - check basic format
    if len(sanitized_id) < 3:
        raise ValueError(f"{component_type} ID must be at least 3 characters long")

    return sanitized_id


def validate_component_label(
    label_text: str,
    component_type: str = "component",
    required: bool = True,
) -> str:
    """Validate component label text following project standards.

    Args:
        label_text: The label text to validate
        component_type: Type of component for error messages
        required: Whether the label is required

    Returns
    -------
        str: Validated label text

    Raises
    ------
        ValueError: If label_text is invalid when required
    """
    # Input validation with early returns
    if not isinstance(label_text, str):
        if required:
            raise ValueError(
                f"{component_type} label must be a string, got {type(label_text).__name__}",
            )
        return ""

    # Configuration validation
    if required and len(label_text.strip()) == 0:
        raise ValueError(f"{component_type} label cannot be empty when required")

    return label_text.strip()


def create_enhanced_numeric_input(
    input_ID: str,
    label_text: str,
    min_value: float = 0,
    max_value: float = 1000000,
    step_size: float = 1,
    default_value: float = 0,
    currency_format: bool = False,
    prefix_symbol: str = "",
    suffix_symbol: str = "",
    help_text: str = "",
    tooltip_text: str = "",
    required_field: bool = False,
    disabled_state: bool = False,
    validation_pattern: str = "",
    placeholder_text: str = "",
    input_width: str = "100%",
) -> Tag:
    """Create enhanced numeric input component with professional styling and validation.

    Args:
        input_ID: Unique identifier for the input component
        label_text: Display label for the input field
        min_value: Minimum allowed value for input validation
        max_value: Maximum allowed value for input validation
        step_size: Step increment for numeric input controls
        default_value: Default value when component initializes
        currency_format: Enable currency formatting display
        prefix_symbol: Symbol to display before input value
        suffix_symbol: Symbol to display after input value
        help_text: Explanatory text displayed below input
        tooltip_text: Tooltip text for additional context
        required_field: Whether input is required for form submission
        disabled_state: Whether input should be disabled
        validation_pattern: Regex pattern for input validation
        placeholder_text: Placeholder text when input is empty
        input_width: CSS width specification for input field

    Returns
    -------
        ui.div: Complete enhanced numeric input component

    Raises
    ------
        ValueError: If input parameters are invalid
    """
    # Input validation with early returns
    validated_id = validate_component_identifier(input_ID, "Input")
    validated_label = validate_component_label(label_text, "Input", required=True)

    # Configuration validation
    if not isinstance(min_value, (int, float)) or not isinstance(max_value, (int, float)):
        raise ValueError("min_value and max_value must be numeric")

    if min_value >= max_value:
        raise ValueError("min_value must be less than max_value")

    if not isinstance(step_size, (int, float)) or step_size <= 0:
        raise ValueError("step_size must be a positive number")

    if not isinstance(default_value, (int, float)):
        raise ValueError("default_value must be numeric")

    # Business logic validation
    if default_value < min_value or default_value > max_value:
        default_value = max(min_value, min(default_value, max_value))

    # Create currency-formatted input with integer-only display
    if currency_format or prefix_symbol:
        # Format the default value as whole dollars (no cents)
        formatted_default = f"${round(default_value):,}"

        # Create the display text input that shows formatted currency
        input_element = ui.input_text(
            id=validated_id,
            label=None,
            value=formatted_default,
            placeholder="$0",
        )

        # Add custom CSS for consistent font styling and proper alignment
        css_style = ui.tags.style(f"""
        #{validated_id} {{
            font-family: inherit;
            font-size: inherit;
            font-weight: inherit;
            text-align: left;
            padding-left: 12px;
            border: 1px solid #ced4da;
            border-radius: 0.375rem;
        }}
        #{validated_id}:focus {{
            border-color: #86b7fe;
            outline: 0;
            box-shadow: 0 0 0 0.25rem rgba(13, 110, 253, 0.25);
        }}
        """)

        # Combine input with styling
        formatted_input = ui.div(
            css_style,
            input_element,
        )

    else:
        formatted_input = ui.input_numeric(
            id=validated_id,
            label=None,
            value=default_value,
            min=min_value,
            max=max_value,
            step=step_size,
        )

    # Create horizontal layout with label and input on same row
    main_content = ui.row(
        ui.column(4, ui.tags.label(validated_label, class_="form-label pt-2")),
        ui.column(8, formatted_input),
    )

    # Build complete component
    component_elements = [main_content]

    # Add help text if provided
    if help_text:
        component_elements.append(
            ui.div(
                ui.tags.small(str(help_text), class_="form-text text-muted"),
                class_="mt-1",
            ),
        )

    return ui.div(
        *component_elements,
        class_="form-group mb-3",
        title=str(tooltip_text) if tooltip_text else None,
        **{
            "data-currency-format": str(currency_format).lower(),
            "data-min-value": str(min_value),
            "data-max-value": str(max_value),
        },
    )


def create_enhanced_card_section(
    title: str,
    content: list[Any],
    icon_class: str = "",
    card_class: str = "",
    header_class: str = "",
    body_class: str = "",
    footer_content: Any | None = None,
) -> Tag:
    """Create enhanced card section with professional styling.

    Args:
        title: Card title text
        content: List of content elements for card body
        icon_class: CSS class for header icon
        card_class: Additional CSS classes for card
        header_class: Additional CSS classes for header
        body_class: Additional CSS classes for body
        footer_content: Optional footer content

    Returns
    -------
        ui.div: Complete enhanced card component

    Raises
    ------
        ValueError: If required parameters are invalid
    """
    # Input validation with early returns
    validated_title = validate_component_label(title, "Card", required=True)

    if not isinstance(content, list):
        raise ValueError("content must be a list of UI elements")

    if len(content) == 0:
        raise ValueError("content list cannot be empty")

    # Build header with optional icon
    header_elements: list[Any] = []
    if icon_class:
        header_elements.append(ui.tags.i(class_=f"{icon_class} me-2"))
    header_elements.append(validated_title)

    # Build card elements
    card_elements = [
        ui.card_header(
            ui.h5(*header_elements, class_="mb-0"),
            class_=f"bg-light {header_class}".strip(),
        ),
        ui.card_body(
            *content,
            class_=body_class,
        ),
    ]

    # Add footer if provided
    if footer_content is not None:
        card_elements.append(
            ui.card_footer(
                footer_content,
                class_="bg-light",
            ),
        )

    return ui.card(
        *card_elements,
        class_=f"shadow-sm {card_class}".strip(),
    )


def create_enhanced_summary_display(
    summary_id: str,
    title: str,
    icon_class: str = "fas fa-info-circle",
    background_class: str = "bg-light",
    text_class: str = "",
    border_variant: ComponentVariant = ComponentVariant.LIGHT,
) -> Tag:
    """Create enhanced summary display component with professional styling.

    Args:
        summary_id: Unique identifier for the summary output
        title: Display title for the summary
        icon_class: CSS class for display icon
        background_class: Background styling class
        text_class: Text styling class
        border_variant: Border variant for styling

    Returns
    -------
        ui.div: Complete enhanced summary display component

    Raises
    ------
        ValueError: If required parameters are invalid
        RuntimeError: If component creation fails
    """
    try:
        # Input validation with early returns
        validated_id = validate_component_identifier(summary_id, "Summary")
        validated_title = validate_component_label(title, "Summary", required=True)

        # Configuration validation
        if not isinstance(border_variant, ComponentVariant):
            border_variant = ComponentVariant.LIGHT

        # Build component classes
        component_classes = [
            "card",
            "text-center",
            "h-100",
            background_class,
            f"border-{border_variant.value}",
        ]

        if text_class:
            component_classes.append(text_class)

        # Create summary display component
        return ui.div(
            ui.div(
                ui.div(
                    ui.tags.i(class_=f"{icon_class} fa-2x mb-2"),
                    ui.h6(validated_title, class_="mb-2"),
                    ui.div(
                        ui.output_text(validated_id),
                        class_="h4 mb-0 font-weight-bold",
                    ),
                    class_="card-body",
                ),
                class_=" ".join(component_classes),
            ),
        )

    except Exception as exc_error:
        raise RuntimeError(f"Unexpected error creating summary display: {exc_error}") from exc_error


def create_enhanced_text_input(
    input_ID: str,
    label_text: str,
    default_value: str = "",
    placeholder_text: str = "",
    help_text: str = "",
    tooltip_text: str = "",
    required_field: bool = False,
    disabled_state: bool = False,
    max_length: int = 255,
    validation_pattern: str = "",
    input_width: str = "100%",
) -> Tag:
    """Create enhanced text input component with professional styling and validation.

    Args:
        input_ID: Unique identifier for the input component
        label_text: Display label for the input field
        default_value: Default text value when component initializes
        placeholder_text: Placeholder text when input is empty
        help_text: Explanatory text displayed below input
        tooltip_text: Tooltip text for additional context
        required_field: Whether input is required for form submission
        disabled_state: Whether input should be disabled
        max_length: Maximum character length for input
        validation_pattern: Regex pattern for input validation
        input_width: CSS width specification for input field

    Returns
    -------
        ui.div: Complete enhanced text input component

    Raises
    ------
        ValueError: If input parameters are invalid
    """
    # Input validation with early returns
    validated_id = validate_component_identifier(input_ID, "Text Input")
    validated_label = validate_component_label(label_text, "Text Input", required=True)

    # Configuration validation
    if not isinstance(max_length, int) or max_length <= 0:
        raise ValueError("max_length must be a positive integer")

    # Create horizontal layout with label and input on same row
    main_content = ui.row(
        ui.column(4, ui.tags.label(validated_label, class_="form-label pt-2")),
        ui.column(
            8,
            ui.input_text(
                id=validated_id,
                label=None,
                value=str(default_value),
                placeholder=str(placeholder_text) if placeholder_text else None,
            ),
        ),
    )

    # Build complete component
    component_elements = [main_content]

    # Add help text if provided using correct Shiny syntax
    if help_text:
        component_elements.append(
            ui.div(
                ui.tags.small(str(help_text), class_="form-text text-muted"),
                class_="mt-1",
            ),
        )

    return ui.div(
        *component_elements,
        class_="form-group mb-3",
        title=str(tooltip_text) if tooltip_text else None,
    )


def create_enhanced_select_input(
    input_ID: str,
    label_text: str,
    choices: dict[str, str],
    default_selection: str = "",
    help_text: str = "",
    tooltip_text: str = "",
    required_field: bool = False,
    disabled_state: bool = False,
    multiple_selection: bool = False,
    input_width: str = "100%",
) -> Tag:
    """Create enhanced select input component with professional styling and validation.

    Args:
        input_ID: Unique identifier for the input component
        label_text: Display label for the select field
        choices: Dictionary of value-label pairs for select options
        default_selection: Default selected value when component initializes
        help_text: Explanatory text displayed below select
        tooltip_text: Tooltip text for additional context
        required_field: Whether selection is required for form submission
        disabled_state: Whether select should be disabled
        multiple_selection: Whether multiple selections are allowed
        input_width: CSS width specification for select field

    Returns
    -------
        ui.div: Complete enhanced select input component

    Raises
    ------
        ValueError: If input parameters are invalid
    """
    # Input validation with early returns
    validated_id = validate_component_identifier(input_ID, "Select Input")
    validated_label = validate_component_label(label_text, "Select Input", required=True)

    # Configuration validation
    if not isinstance(choices, dict) or len(choices) == 0:
        raise ValueError("choices must be a non-empty dictionary")

    # Create horizontal layout with label and input on same row
    main_content = ui.row(
        ui.column(4, ui.tags.label(validated_label, class_="form-label pt-2")),
        ui.column(
            8,
            ui.input_select(
                id=validated_id,
                label=None,
                choices=choices,
                selected=str(default_selection) if default_selection else None,
                multiple=multiple_selection,
            ),
        ),
    )

    # Build complete component
    component_elements = [main_content]

    # Add help text if provided using correct Shiny syntax
    if help_text:
        component_elements.append(
            ui.div(
                ui.tags.small(str(help_text), class_="form-text text-muted"),
                class_="mt-1",
            ),
        )

    return ui.div(
        *component_elements,
        class_="form-group mb-3",
        title=str(tooltip_text) if tooltip_text else None,
    )


def create_enhanced_button(
    button_id: str,
    button_text: str,
    button_variant: ComponentVariant = ComponentVariant.PRIMARY,
    button_size: str = "md",
    disabled_state: bool = False,
    icon_class: str = "",
    button_width: str = "auto",
    onclick_action: str = "",
) -> Tag:
    """Create enhanced button component with professional styling.

    Args:
        button_id: Unique identifier for the button component
        button_text: Display text for the button
        button_variant: Button styling variant
        button_size: Button size (sm, md, lg)
        disabled_state: Whether button should be disabled
        icon_class: CSS class for button icon
        button_width: CSS width specification for button
        onclick_action: JavaScript action for button click

    Returns
    -------
        ui.div: Complete enhanced button component

    Raises
    ------
        ValueError: If input parameters are invalid
    """
    # Input validation with early returns
    validated_id = validate_component_identifier(button_id, "Button")
    validated_text = validate_component_label(button_text, "Button", required=True)

    # Configuration validation
    if not isinstance(button_variant, ComponentVariant):
        button_variant = ComponentVariant.PRIMARY

    valid_sizes = ["sm", "md", "lg"]
    if button_size not in valid_sizes:
        button_size = "md"

    # Build button classes
    button_classes = [
        "btn",
        f"btn-{button_variant.value}",
    ]

    if button_size != "md":
        button_classes.append(f"btn-{button_size}")

    # Build button content
    button_content: list[Any] = []
    if icon_class:
        button_content.append(ui.tags.i(class_=f"{icon_class} me-2"))
    button_content.append(validated_text)

    # Create button element
    button_element = ui.input_action_button(
        id=validated_id,
        label=ui.span(*button_content),
        class_=" ".join(button_classes),
    )

    return ui.div(
        button_element,
        style=f"width: {button_width};" if button_width != "auto" else None,
    )
