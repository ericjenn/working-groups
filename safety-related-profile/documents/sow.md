# Introduction

## Terms

The following terms are used in the document:

-   **Trained ML model**: the conceptual structure of graphs and
    operators that maps a set of input tensors to a set of output
    tensors. The Trained ML Model Description (TMD) is represented using
    the *Trained Model Description Language* (TMDL).

-   **Trained ML Model Description** (TMD: a concrete representation of
    the conceptual trained ML Model that can be interpreted by a human
    or a program.

-   **Trained ML Model Description** **Language** (TMDL): The language
    used to represent a *Trained ML Model* (TMD).

## Contents

This document gives a first definition of the activities to be carried
out by the ONNX safety-related profile workgroup. This "workplan" is
aimed at being presented *and discussed* during the Workgroup kick-off
meeting planned for the end of September.

## Main objectives

Provide a definition of the formalism used to represent a trained ML
model\...

-   *\... that has an understandable and non ambiguous syntax and
    semantics.*

    The description of the ML model expressed using this formalism must
    be a Low level Requirement for the implementation phase in the sense
    that its interpretation and implementation shall not require any
    further information that the one given by the description of the ML
    model.

-   *\... that allows multiple levels of accuracy and precision for the
    description of a given model.*

    The language used to describe the model (i.e., its syntax and
    semantics) must be non ambiguous, but a model may be ambiguous if
    this ambiguity is acceptable or even necessary to leave some freedom
    to the implementer (e.g., for optimization). The objective is to
    identify, control, and possibly remove this ambiguity by an
    appropriate refinement of the ML model description.

# Workgroup Activities

This section presents the different activities of the workgroup. Their
dependencies are expressed via their inputs/outputs.

### A1 - Elicitation of industrial needs

#### Objectives

Elicit end-users needs related to the ONNX format, i.e., What arethe activities using ONNX models?, What are the evidences required by certification authorities that involve the ONNX model[^1]?, How do the ONNX model impact these activities?

#### Rationales {#rationales .unnumbered}

Clarify the expectation of end-users.

#### Inputs

-   Certification standards (e.g., ARP6983, ISO/DPAS 8800,
    ECSS-E-HB-40-02A DIR1, etc.)

-   Company-specific requirements

-   End-user use cases

#### Outputs

-   D1.a.\<x\>: End users needs and requirements for domain \<x\>.
-   D1.b: Consolidated needs for all industrial domains
-   D1.c: Safety-related Profile Scope Definition

#### Detailed activities

The activities defined below are per domain <x>.

##### SC: SR-profile Scope Definition

- SCAct1: Identification/definition of the Safety-related industrial use case reference models

- SCAct2: Extraction of the operators and constructs from the Safety-related industrial use cases reference models

- SSAct3: Consolidation of the set of operators and constructs to be included in the Safety-related profile (from the reference models possibly augmented with necessary additional operators and constructs).

##### EN: End-Users Needs Elicitation

-   ENAct1.\<x\>: Description (overview) of the machine learning development
    process

-   ENAct2.\<x\>: Description of the development process objectives and
    activities that produce the TMD and take it as input

-   ENAct3.\<x\>: Definition of the *Trained Model Description (TMD)* artefact
    (e.g., the Machine Learning Model Description (MLMD) in ARP6983)

-   UENAct4.\<x\>: Description of the development process verification
    objectives and activities that apply to the TMD

-   ENAct5.\<x\>: Expression of constraints on the TDM, that come from the Development and verification activities the Industrial context

-   ENAct6.\<x\>: Expression of the needs

-   ENAct7.\<x\>:Consolidation of industrial needs from all domains

##### *ONNX Requirements Expression*

The activities below take the end-user needs as inputs

-   ORAct1: Definition of the list of the aspects to which the
    requirements for a safety-related ONNX profile will apply, e.g.,
    -   Semantics of the operators
    -   Semantics of the graph
    -   Data types
    -   Metamodel
    -   Concrete syntax (format)
    -   Documentation
    -   Traceability
    -   Versioning
    -   etc.

-   ORAct2: For each aspect of the list, definition of the requirements
    *Examples of requirements that may be expressed:*
    -   The semantics of the Trained Model Description Language (TMDL) used
    to describe the TMD shall be defined both informally (for
    documentation purposes) and formally using a mathematically-grounded
    language. This covers all that is needed for tooled and/or human
    interpretation of any valid TMD described using the TMDL (including,
    (e.g., operators and graphs).
    -   The formal definition of the TMDL shall define precisely and
    accurately the expected results of the interpretation of any valid
    TMDL model. The level of precision and accuracy may be a parameter
    of the description of the semantics.
    -   A reference implementation shall be provided for each operator. The
    reference implementation shall be accompanied with all the necessary
    information describing the execution environment used to validate
    compliance with the formal specification.
    -   In the TMD, it should be possible to indicate the meaning of each
    dimension of the tensors

### A2: Specification of the ONNX SR profile

#### Objectives

Elaborate the list of requirements applicable to the SR profile in order to comply with the end users' needs. Consolidate, filter, and prioritize the needs identified for the different industrial domains in D1.a.<x>. Discriminate requirements aimed at the preservation of the model semantics from requirements aimed at facilitating / supporting other development assurance activities.

#### Rationale

TBD

#### Inputs

-   D1.a.\<x\>: End users needs and requirements for domain \<x\>.
-   D1.b: Consolidated needs for all industrial domains

#### Outputs

-  D2.a: ONNX safety-related Profile Requirements

#### Detailed activities

##### OR: ONNX SR profile requirements specification

-  ORAct1: Definition of the list of the aspects (e.g., accuracry, completeness, traceability, etc.) to which the requirements for a safety-related ONNX profile will apply

-  ORAct2: For each aspect, definition of the requirements applicable to the standard

-  ORAct3: Grouping and prioritization of requirements

### A3: Development of the ONNX SR profile

#### Objectives

Development of the ONNX Safety-related profile (syntax and semantics) in compliance with the ONNX safety-related Profile Requirements (D2.a)

#### Rationale

TBD

#### Inputs

-  D2.a: ONNX safety-related Profile Requirements

#### Outputs

-   D3.a: ONNX Safety-related profile - proof of concept
-   D3.b: ONNX Safety-related profile - graph
-   D3.c: ONNX Safety-related profile - operators
-   D3.d: ONNX Safety-related profile - format
-   D3.e: ONNX Safety-related profile reference implementation

#### Detailed activities

##### POC: Proof of concept

-   PROAct1: Elaborate a first set of (informal + formal) specification guidelines and apply them on a few operators (e.g., conv) and constructs in order to discussed and reviewed by the workgroup. Will serve as a baseline for other operators

##### GR: Graph execution semantics

-   GRAct1: Development of the ONNX SR profile graph semantics in compliance with specification D2.a

##### OP: Operator semantics

-   OPAct1: Development of the ONNX SR profile operators semantics in compliance with specification D2.a

##### FO: Format

-  FOAct1: Development of the ONNX SR profile exchange format in compliance with specification D2.a

##### RI: Reference implementation

-  RIAct1: Development of a reference implemation (on the basis of the existing ref imp.). This covers the development of the graph execution engine and operators.

### A4: V&V of the ONNX Safety-related profile

#### Objectives

Verification (validation) of the ONNX Safety-related profile vs the requirements (resp. needs) expressed in D2.a (resp D1.b and D1.c)

#### Rationales

TBD

#### Inputs

-   D3.b: ONNX Safety-related profile - graph

-   D3.c: ONNX Safety-related profile - operators

-   D3.d: ONNX Safety-related profile - format

#### Outputs

-   D4.a: ONNX Safety-related profile verification report
    Profile
-   D4.b: ONNX Safety-related profile validation report. 

#### Detailed activities

##### VE: Verification

-   VEAct1: Review of the ONNX SR profile against the requirments in D2.a 

##### VA: Validation

-   VAAct1: Validation of the ONNX SR profile via its application to one or several industrial use cases 

##### Analysis of the ONNX standard

-   AnaAct1: Analysis of the compliance of the ONNX standard with
    respect to each of the requirements and identification of
    non-compliances.

-   AnaAct2: Provision of recommendations, solutions, guidance to modify
    the ONNX standard.

### A5: Tooling

#### Objectives

Provision of tooling to support the exploitation of the ONNX SR model. E.g., model inspection and review tool, etc. This 

#### Rationales

TBD

#### Inputs



#### Outputs



#### Detailed activities






