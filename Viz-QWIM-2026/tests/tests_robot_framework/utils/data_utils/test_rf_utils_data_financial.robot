*** Settings ***
Documentation
...    Robot Framework test suite for utils_data_financial financial estimator type enums.
...
...    Verifies member counts, classmethod group returns, group disjointness,
...    and full member coverage for:
...      - expected_returns_estimator_type (5 members)
...      - prior_estimator_type (7 members)
...      - distribution_estimator_type (14 members)
...
...    Test Categories:
...    - Enum member counts (smoke)
...    - Classmethod group non-emptiness
...    - Group disjointness
...    - Full member coverage
...
...    Author:     QWIM Development Team
...    Version:    0.1.0
...    Last Modified:    2026-03-01

Library     ${CURDIR}${/}keywords_utils_data_financial.py
Library     Collections

Suite Setup      Log    Starting utils_data_financial Robot Framework Test Suite
Suite Teardown   Log    utils_data_financial Test Suite Complete


*** Variables ***
${EXPECTED_RETURNS_MEMBER_COUNT}    ${5}
${PRIOR_MEMBER_COUNT}               ${7}
${DISTRIBUTION_MEMBER_COUNT}        ${14}


*** Test Cases ***

# ---------------------------------------------------------------------------
# expected_returns_estimator_type — member count
# ---------------------------------------------------------------------------

Expected Returns Estimator Type Has Exactly Five Members
    [Documentation]    expected_returns_estimator_type enum must expose exactly 5 members.
    [Tags]    utils    enums    smoke    expected_returns
    ${count}=    Get Expected Returns Member Count
    Integer Should Equal    ${count}    ${EXPECTED_RETURNS_MEMBER_COUNT}

# ---------------------------------------------------------------------------
# expected_returns_estimator_type — group classmethods
# ---------------------------------------------------------------------------

Expected Returns Historical Methods Returns Non Empty List
    [Documentation]    get_historical_methods must return at least one member.
    [Tags]    utils    enums    expected_returns
    ${methods}=    Get Expected Returns Historical Methods
    List Should Be Non Empty    ${methods}

Expected Returns Model Based Methods Returns Non Empty List
    [Documentation]    get_model_based_methods must return at least one member.
    [Tags]    utils    enums    expected_returns
    ${methods}=    Get Expected Returns Model Based Methods
    List Should Be Non Empty    ${methods}

Expected Returns Historical And Model Based Are Disjoint
    [Documentation]    Historical and model-based groups must not share any member.
    [Tags]    utils    enums    expected_returns
    ${hist}=    Get Expected Returns Historical Methods
    ${model}=   Get Expected Returns Model Based Methods
    Lists Should Be Disjoint    ${hist}    ${model}

Expected Returns All Members Are Covered By At Least One Group
    [Documentation]    Every expected_returns_estimator_type member must appear
    ...    in at least one classmethod group.
    [Tags]    utils    enums    expected_returns
    ${covered}=    Expected Returns All Members Are Covered
    Bool Should Be True    ${covered}

# ---------------------------------------------------------------------------
# prior_estimator_type — member count
# ---------------------------------------------------------------------------

Prior Estimator Type Has Exactly Seven Members
    [Documentation]    prior_estimator_type enum must expose exactly 7 members.
    [Tags]    utils    enums    smoke    prior
    ${count}=    Get Prior Member Count
    Integer Should Equal    ${count}    ${PRIOR_MEMBER_COUNT}

# ---------------------------------------------------------------------------
# prior_estimator_type — group classmethods
# ---------------------------------------------------------------------------

Prior Data Driven Returns Non Empty List
    [Documentation]    get_data_driven must return at least one member.
    [Tags]    utils    enums    prior
    ${methods}=    Get Prior Data Driven
    List Should Be Non Empty    ${methods}

Prior Equilibrium Based Returns Non Empty List
    [Documentation]    get_equilibrium_based must return at least one member.
    [Tags]    utils    enums    prior
    ${methods}=    Get Prior Equilibrium Based
    List Should Be Non Empty    ${methods}

Prior Data Driven And Equilibrium Based Are Disjoint
    [Documentation]    Data-driven and equilibrium-based groups must not share any member.
    [Tags]    utils    enums    prior
    ${dd}=    Get Prior Data Driven
    ${eq}=    Get Prior Equilibrium Based
    Lists Should Be Disjoint    ${dd}    ${eq}

Prior All Members Are Covered By At Least One Group
    [Documentation]    Every prior_estimator_type member must appear
    ...    in at least one classmethod group.
    [Tags]    utils    enums    prior
    ${covered}=    Prior All Members Are Covered
    Bool Should Be True    ${covered}

# ---------------------------------------------------------------------------
# distribution_estimator_type — member count
# ---------------------------------------------------------------------------

Distribution Estimator Type Has Exactly Fourteen Members
    [Documentation]    distribution_estimator_type enum must expose exactly 14 members.
    [Tags]    utils    enums    smoke    distribution
    ${count}=    Get Distribution Member Count
    Integer Should Equal    ${count}    ${DISTRIBUTION_MEMBER_COUNT}

# ---------------------------------------------------------------------------
# distribution_estimator_type — group classmethods
# ---------------------------------------------------------------------------

Distribution Univariate Returns Non Empty List
    [Documentation]    get_univariate must return at least one member.
    [Tags]    utils    enums    distribution
    ${methods}=    Get Distribution Univariate
    List Should Be Non Empty    ${methods}

Distribution Fat Tailed Returns Non Empty List
    [Documentation]    get_fat_tailed_distributions must return at least one member.
    [Tags]    utils    enums    distribution
    ${methods}=    Get Distribution Fat Tailed
    List Should Be Non Empty    ${methods}

Distribution Simulation Ready Returns Non Empty List
    [Documentation]    get_simulation_ready must return at least one member.
    [Tags]    utils    enums    distribution
    ${methods}=    Get Distribution Simulation Ready
    List Should Be Non Empty    ${methods}

Distribution All Members Are Covered By At Least One Group
    [Documentation]    Every distribution_estimator_type member must appear
    ...    in at least one classmethod group.
    [Tags]    utils    enums    distribution
    ${covered}=    Distribution All Members Are Covered
    Bool Should Be True    ${covered}
