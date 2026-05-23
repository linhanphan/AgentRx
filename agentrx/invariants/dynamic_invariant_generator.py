#!/usr/bin/env python3
"""
Domain-agnostic DYNAMIC invariant generator.

Supports two modes:
  --mode stepbystep   Generate one invariant per trajectory step (default)
  --mode oneshot      Generate all invariants in a single LLM call per trajectory

Supports domains: flash, tau, magentic (via --domain flag)
Supports both static and dynamic invariants (via --include-nl-check / --no-nl-check)

Usage (from src/ directory — required for imports):
  cd src

  # Tau-retail: step-by-step on a single trajectory
  python -m invariants.dynamic_invariant_generator --domain tau \
      --mode stepbystep --input-path ../../trajectories/tau-retail/hallucination_doubt.json

  # Tau-retail: one-shot, python-only checks
  python -m invariants.dynamic_invariant_generator --domain tau \
      --mode oneshot --input-path ../../trajectories/tau-retail/misinterpretation_tool_output.json --no-nl-check

  # Tau-retail: step-by-step on all files in directory
  python -m invariants.dynamic_invariant_generator --domain tau \
      --mode stepbystep --input-path ../../trajectories/tau-retail/

  # Magentic-One: step-by-step on all trajectory files
  python -m invariants.dynamic_invariant_generator --domain magentic \
      --mode stepbystep --input-path ../../trajectories/magentic-one/trajectories/

  # Magentic-One: single trajectory with Azure endpoint and custom output dir
  python -m invariants.dynamic_invariant_generator --domain magentic \
      --input-path ../../trajectories/magentic-one/trajectories/plan_adherence_failure.json \
      --endpoint azure --out-dir ../../results/magentic_dynamic/

  # Flash: step-by-step on magentic dataset directory
  python -m invariants.dynamic_invariant_generator --domain flash \
      --input-path ../../data/magentic_dataset/

  NOTE: --input-path is resolved relative to src/invariants/ (where this script lives).
        Use ../../ to reach the repo root, or use absolute paths.

CLI flags:
  --domain {flash,tau,magentic}     Domain to run (default: flash)
  --mode {stepbystep,oneshot}       Generation mode (default: stepbystep)
  --input-path PATH                 Trajectory file or directory of .json/.jsonl files
  --out-dir DIR                     Output directory (default: dynamic_invariant_outputs)
  --static-invariants PATH          Path to static invariants JSON
  --endpoint {trapi,azure}          LLM endpoint (default: trapi)
  --include-nl-check / --no-nl-check  Enable/disable NL check invariants (default: enabled)

Also importable for the larger pipeline:
  from agentrx.invariants.dynamic_invariant_generator import DynamicInvariantGenerator
  from agentrx.invariants.dynamic_invariant_generator import OneShotDynamicInvariantGenerator
"""
import os
import json
import time
import traceback
import datetime
import argparse
from typing import Any, Dict, List, Optional, Set, Union, Tuple

from agentrx.ir.trajectory_ir import tau_bench_ir, flash_ir, magentic_ir, load_trajectories
from agentrx.llm_clients.trapi import LLMAgent as LLMAgentTrapi
from agentrx.llm_clients.azure import LLMAgent as LLMAgentAzure
import agentrx.pipeline.globals as g

from agentrx.reports.metrics import TokenUsage, TimingInfo, LLMCallTelemetry  # noqa: F401

_RATE_LIMIT_RETRIES = 5
_RATE_LIMIT_BACKOFF = 4  # seconds

def _llm_call_with_retry(client, model_name, messages, **kwargs):
    """Wrapper around chat.completions.create with retry on rate limits."""
    from openai import RateLimitError
    for attempt in range(_RATE_LIMIT_RETRIES):
        try:
            return client.chat.completions.create(
                model=model_name, messages=messages, **kwargs
            )
        except RateLimitError:
            if attempt < _RATE_LIMIT_RETRIES - 1:
                wait = _RATE_LIMIT_BACKOFF * (2 ** attempt)
                print(f"[RATE LIMIT] Retrying in {wait}s (attempt {attempt+1}/{_RATE_LIMIT_RETRIES})", flush=True)
                time.sleep(wait)
            else:
                raise

# ------------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG = os.getenv("DEBUG", "1") == "1"


DYNAMIC_INVARIANT_PROMPT_WITHOUT_NL_CHECK = """
You are an expert at analyzing agent trajectories and generating step-specific dynamic invariants which are programmable for trajectory verification. 
Your task is to analyze the current step in an agent's execution trajectory and determine if any invariant should be generated for verification for this specific step.

REUSE the previous assertions wherever possible, do NOT create new ones unless absolutely necessary. Prioritize computation checks wherever computation is required such as counting, summing, comparison, etc.

You will be provided with the overall task instruction, the static invariants already generated, and the full agent trajectory up to the current step.

**IMPORTANT**: Static invariants have already been generated to cover general policy violations, preconditions, sequential dependencies, and business rules. Focus ONLY on cases NOT covered by static invariants, such as:
- Computation errors (counting, summing, calculations)
- Step-specific validation that static invariants cannot capture

## Context Information:
**Task Instruction:** <<TASK_INSTRUCTION>>

**Current Step Index:** <<STEP_NUM>>

## Your Task:
Analyze the current step in the trajectory and generate dynamic invariants that:
1. Are simple enough to implement via python or NL checks.
2. Can be checked at this specific step in the trajectory (e.g., computation checks from tool call results)
3. Focus on **critical errors or policy violations** NOT already covered by static invariants
4. Are objectively verifiable from trajectory data
5. Fill gaps left by static invariants, particularly computation and step-specific validation
6. Provide working Python code or Natural Language (NL) check for each dynamic assertion that can be executed

## Focus Areas:
Generate assertions for these key constraint types (ONLY if NOT covered by static invariants):

### **Computation Checks**
- Verify numerical calculations, aggregations, and data accuracy.
- Examples:
  - Financial: "Transaction total should equal sum of line items plus applicable fees"
  - Inventory: "Remaining stock after reservation should be original stock minus reserved quantity"
  - Healthcare: "Medication dosage calculations should be within safe therapeutic ranges"

### **Context-Specific Business Logic**
- Step-specific rules that depend on current trajectory state
- Examples:
  - Retail: "Discount calculations based on current cart contents and user loyalty level"
  - Financial: "Credit limit checks based on current account status and transaction history"
  - Healthcare: "Drug interaction checks based on current prescription list"
  - Workflow: "Approval requirements based on current user role and request amount"

### **Cross-Tool Data Validation**
- Verify data integrity across multiple tool interactions in the same session
- Examples:
  - "Entity identifiers referenced in later steps must match those retrieved in earlier tool outputs"
  - Retail: "Items added to cart must still be available when proceeding to checkout"
  - "If the agent claims an action succeeded, subsequent tool outputs must show evidence consistent with the claim"
  - Financial: "Exchange rates used in calculations must be consistent across related transactions"
  - Healthcare: "Lab results referenced in diagnosis must match the lab results retrieved"
  - System: "File permissions checked must match the permissions set in previous operations"

Output MUST be valid JSON only (no markdown, no commentary), with EXACT keys and structure defined below.

CRITICAL FORMATTING REQUIREMENTS:
1. Output ONLY valid JSON - NO markdown code blocks (no ```json or ```)
2. Use double quotes for all strings (not single quotes)
3. Ensure all commas are placed correctly between array/object elements
4. Ensure all brackets and braces are properly closed
5. Follow the EXACT schema below - do not add or remove fields
6. Validate your JSON before outputting
7. TAXONOMY TARGETS: Use ONLY values from the allowed list.
8. ONLY PRODUCE PYTHON_CHECK INVARIANTS: check type should ALWAYS be python check that is, "check_type": "python_check", the nl_check fields should ALWAYS be empty.

================================================================================================
TRAJECTORY FORMAT (IMPORTANT) - THIS WILL BE GIVEN AS AN INPUT PARAMETER
================================================================================================

TRAJECTORY:
The runtime provides a trajectory dict in the following IR schema:

{
  "trajectory_id": "str",
  "instruction": "str",  // The overall task instruction
  "steps": [
    {
      "index": int,  // 1-based step index
      "substeps": [
        {
          "sub_index": int,  // 1-based substep index within the step
          "role": "str",  // The role of the substep
          "content": "str"  // The content of this substep
        }
      ]
    }
  ]
}

IMPORTANT ACCESS PATTERNS:
- To get instruction: trajectory["instruction"]
- To access step N: trajectory["steps"][N-1] (steps list is 0-indexed, but step.index is 1-based)
- To get substeps of a step: step.get("substeps", [])
- To get content: substep.get("content")
- To get role: substep.get("role")

================================================================================================
OUTPUT FORMAT
================================================================================================

Top-level JSON schema (EXACT):
{
  "step_num": <int>,
  "decision": "NO_INVARIANT" | "INVARIANT",
  "invariant":  [ <InvariantObject> ] | null,
  "trigger_to_invariants_map": {
    "agent:<AgentName>": ["assertion_name", "..."]
   }
}

where <AgentName> is one of: <<AGENT_UNION>>

================================================================================================
InvariantObject schema (EXACT)
================================================================================================
Each invariant object MUST have these keys (no extras):

IMPORTANT: Follow this schema EXACTLY. Do not add extra fields. Use correct JSON formatting.

{
  "assertion_name": "string_unique_snake_case",

  "taxonomy_targets": [
    "Instruction/PlanAdherenceFailure: The agent fails to follow the directions or the agreed plan by ignoring directives and skipping policy steps. This covers both under-execution (missed steps) and over-execution (unplanned or unnecessary actions, e.g., extra tool calls) that deviate from the static plan, domain policy or orchestrator plan.",
    "InventionOfNewInformation: The agent introduces, removes, or alters information that is not grounded in any available input, context, or tool output. This includes fabricating unsupported facts, hallucinating details, or omitting relevant information without justification.",
    "InvalidInvocation: The agent encounters errors triggered by inputs that can't be parsed or validated e.g., syntax errors or tool calls with bad/missing arguments. Not involving wrong logic; just invalid inputs.",
    "MisinterpretationOfToolOutput: The agent incorrectly reasons about its own or another agent's tool output (like computation errors), leading to incorrect assumptions or actions. This also includes cases where the agent considered only partial tool output.",
    "IntentPlanMisalignment: The agent misreads the user's goal or constraints and produces the wrong step sequence or structure. Covers both bad ordering/structure and plans aimed at the wrong objective.",
    "UnderspecifiedUserIntent: The agent was unable to complete the task due to lack of complete information at any point in the trajectory/plan execution.",
    "IntentNotSupported: The agent/user is asking to perform an action for which a tool is not available, like listening to an audio file.",
    "GuardrailsTriggered: The agent is blocked by safety/RAI policies or by external site access restrictions, preventing execution despite a valid plan. Examples include policy refusals (e.g., unsafe content, privacy-protected data), CAPTCHA/robot blocks, login/paywall/403/robots.txt denials, or site forbids automation. This is not an agent planning/execution error; it is an external/guardrail block.",
    "SystemFailure: The agent faces a system connectivity issue while calling a particular tool like an endpoint not being reachable."
    // Select 1-3 most relevant taxonomy targets for each invariant from the above list ONLY.

    // Example: ["Instruction/PlanAdherenceFailure", "GuardrailsTriggered"]
  ],

  "invariant_type": "SCHEMA  // Ensures a structured payload (embedded JSON, tool args/result, ledger blob) is parseable and contains required fields.
  | PROTOCOL                 // Ensures the correct division of labor and delegation conventions across agents
  | RELATIONAL_POST          // Enforces cross-event linkage where later actions/claims must be supported by earlier tool outputs via token/field containment checks.
  | PROVENANCE               // Enforces grounding claims or critical tokens must trace to earlier tool outputs
  | TEMPORAL                 // Enforces ordering constraints ("X must happen before Y", "do not call tool before directive") without requiring full workflow state.
  | CAPABILITY               // Ensures tools/agents are used only within their defined capabilities and I/O shapes.
  | ANY",

  "event_trigger": {
    "step_index": "*|int|range",
    "substep_index": "*|int|range",
    "role_name": "<<AGENT_UNION>>"  // agent names inferred from substep.role; "*" matches all
  },

  "check_hint": "deterministic procedure description in 2-8 sentences",
  "check_type": "python_check|nl_check",

  "python_check": {
    "function_name": "same_as_assertion_name",
    "args": ["trajectory","current_step_index"],
    "code_lines": [
      "def same_as_assertion_name(trajectory, current_step_index):",
      "    \\"\\"\\\"Return True iff invariant holds.\\"\\"\\\"",
      "    # Access steps via trajectory['steps'][current_step_index]",
      "    # Each step has 'substeps' with 'role' and 'content'",
      "    # MUST include at least one explicit failure path: return False",
      "    return True"
    ]
  },

  "nl_check": {
    ALWAYS EMPTY
  }
}

## Quality Guidelines:
- **Policy-Derived**: Must be supported by the policy document content, include any user confirmation checks before critical actions if specified in policy
- **Practical Impact**: Focus on assertions that prevent real operational problems
- **Domain-Relevant**: Prioritize rules specific to this business domain
- **Reasonable Scope**: Avoid both trivial sanity checks and overly complex checks that would be difficult to implement reliably.

## Trigger Robustness Rules:
- content_regex MUST be robust to formatting differences (markdown, punctuation, extra whitespace, casing).
- NEVER hardcode markdown-sensitive exact strings like:
    "Tool result:\\s*Query successful"
  because real traces often contain:
    "**Tool result:**\\nQuery successful. ..."
- Prefer matching stable substrings that survive formatting changes.
- Prefer SHORT regexes. Avoid overfitting to ":" vs "**:**" vs "." variations.
- Avoid brittle patterns that depend on exact formatting or punctuation.
- Make sure to match the role name perfectly and overapproximate if you are unsure.

Each invariant should be as GENERAL as possible, if many specific invariants can be combined into one invariant function, do so.
For example, an invariant which checks order status validity for various tool calls can be combined into one function.

================================================================================================
## PYTHON CHECK GUIDELINES
================================================================================================

## Python Code Guidelines:
- Write complete, executable Python functions
- You will be given 2 parameters - (trajectory and current_step_index), where trajectory is the full IR dict with "steps" field
- Access steps via trajectory["steps"][current_step_index] - steps list is 0-indexed
- Each step has "index" (1-based) and "substeps" array
- Each substep has "sub_index", "role", and "content"
- The current_step_index is 0-based. So if the current step being executed has index=7, current_step_index will be 6.
- Include docstrings explaining the assertion
- Look at the tool response key fields very carefully while writing the code.
  1. First, identify which tool(s) you're working with in the current trajectory step
  2. Pay attention to nested dictionaries (use chained .get() calls) and list structures (check length before indexing)
  3. Note the data types (str, float, bool, list, dict) to avoid type errors
  4. Use ONLY the exact field names shown in the tool structure - do NOT invent or assume field names
- Handle edge cases and missing data gracefully - verify field existence and type before operations
- Focus on computation validation and step-specific checks, avoid very trivial sanity checks
- **Exception Handling**: Raise exceptions for safety check violations (KeyError, IndexError, TypeError, AttributeError, ValueError, etc.) instead of returning False. Let these exceptions propagate naturally. Only return boolean (True/False) for actual invariant logic violations
- Return boolean values (True if assertion passes, False if invariant violation detected)
- **Add print statements for debugging**: Start with printing the function name, then print all key variables, extracted JSON fields, and intermediate calculation results to aid in debugging and verification

================================================================================================
## PYTHON_CHECK ANTI-HALLUCINATION RULES
================================================================================================
- A python_check MUST implement ONLY the exact condition(s) described in this invariant's `check_hint`.
- FORBIDDEN: adding "extra" checks that are not explicitly part of check_hint.
  Examples of forbidden extras:
  * checking additional URLs/IDs/regexes beyond the targeted substring(s)
  * inferring "plan rules" not explicitly spelled out in check_hint
  * scanning many steps for unrelated markers "just in case"
  * inventing secondary conditions like "also must not contain X" unless required by check_hint
- Keep python_check SMALL and TARGETED:
  * prefer 1-3 simple extractions + 1 boolean condition
  * avoid long keyword lists and complex regex (if you need that, it is probably a nl_check candidate)

YOU NEED NOT GENERATE A DYNAMIC INVARIANT FOR EVERY STEP. IF NO NEW DYNAMIC INVARIANTS ARE NEEDED OR ARE ALREADY COVERED BY STATIC INVARIANTS, OUTPUT:

{
  "step_num": <<STEP_NUM>>,
  "decision": "NO_INVARIANT",
  "invariant": null,
  "trigger_to_invariants_map": {}
}

================================================================================================
STATIC INVARIANTS ALREADY GENERATED
================================================================================================

<<STATIC_INVARIANTS>>

================================================================================================
DYNAMIC INVARIANTS GENERATED FOR PREVIOUS STEPS
================================================================================================

<<DYNAMIC_INVARIANTS_ALREADY_GENERATED>>

================================================================================================
TRAJECTORY STEPS TILL NOW
================================================================================================

<<TRAJECTORY_UP_TO_CURRENT_STEP>>"""

DYNAMIC_INVARIANT_PROMPT = """
You are an expert at analyzing agent trajectories and generating step-specific dynamic invariants which are programmable for trajectory verification. 
Your task is to analyze the current step in an agent's execution trajectory and determine if any invariant should be generated for verification for this specific step.

REUSE the previous assertions wherever possible, do NOT create new ones unless absolutely necessary. Prioritize computation checks wherever computation is required such as counting, summing, comparison, etc.

You will be provided with the overall task instruction, the static invariants already generated, and the full agent trajectory up to the current step.

**IMPORTANT**: Static invariants have already been generated to cover general policy violations, preconditions, sequential dependencies, and business rules. Focus ONLY on cases NOT covered by static invariants, such as:
- Computation errors (counting, summing, calculations)
- Step-specific validation that static invariants cannot capture

## Context Information:
**Task Instruction:** <<TASK_INSTRUCTION>>

**Current Step Index:** <<STEP_NUM>>

## Your Task:
Analyze the current step in the trajectory and generate dynamic invariants that:
1. Are simple enough to implement via python or NL checks.
2. Can be checked at this specific step in the trajectory (e.g., computation checks from tool call results)
3. Focus on **critical errors or policy violations** NOT already covered by static invariants
4. Are objectively verifiable from trajectory data
5. Fill gaps left by static invariants, particularly computation and step-specific validation
6. Provide working Python code or Natural Language (NL) check for each dynamic assertion that can be executed

## Focus Areas:
Generate assertions for these key constraint types (ONLY if NOT covered by static invariants):

### **Computation Checks**
- Verify numerical calculations, aggregations, and data accuracy.
- Examples:
  - Financial: "Transaction total should equal sum of line items plus applicable fees"
  - Inventory: "Remaining stock after reservation should be original stock minus reserved quantity"
  - Healthcare: "Medication dosage calculations should be within safe therapeutic ranges"

### **Context-Specific Business Logic**
- Step-specific rules that depend on current trajectory state
- Examples:
  - Retail: "Discount calculations based on current cart contents and user loyalty level"
  - Financial: "Credit limit checks based on current account status and transaction history"
  - Healthcare: "Drug interaction checks based on current prescription list"
  - Workflow: "Approval requirements based on current user role and request amount"

### **Cross-Tool Data Validation**
- Verify data integrity across multiple tool interactions in the same session
- Examples:
  - "Entity identifiers referenced in later steps must match those retrieved in earlier tool outputs"
  - Retail: "Items added to cart must still be available when proceeding to checkout"
  - "If the agent claims an action succeeded, subsequent tool outputs must show evidence consistent with the claim"
  - Financial: "Exchange rates used in calculations must be consistent across related transactions"
  - Healthcare: "Lab results referenced in diagnosis must match the lab results retrieved"
  - System: "File permissions checked must match the permissions set in previous operations"

Output MUST be valid JSON only (no markdown, no commentary), with EXACT keys and structure defined below.

CRITICAL FORMATTING REQUIREMENTS:
1. Output ONLY valid JSON - NO markdown code blocks (no ```json or ```)
2. Use double quotes for all strings (not single quotes)
3. Ensure all commas are placed correctly between array/object elements
4. Ensure all brackets and braces are properly closed
5. Follow the EXACT schema below - do not add or remove fields
6. Validate your JSON before outputting
7. TAXONOMY TARGETS: Use ONLY values from the allowed list.
8. ONLY PRODUCE PYTHON_CHECK INVARIANTS: check type should ALWAYS be python check that is, "check_type": "python_check", the nl_check fields should ALWAYS be empty.

================================================================================================
TRAJECTORY FORMAT (IMPORTANT) - THIS WILL BE GIVEN AS AN INPUT PARAMETER
================================================================================================

TRAJECTORY:
The runtime provides a trajectory dict in the following IR schema:

{
  "trajectory_id": "str",
  "instruction": "str",  // The overall task instruction
  "steps": [
    {
      "index": int,  // 1-based step index
      "substeps": [
        {
          "sub_index": int,  // 1-based substep index within the step
          "role": "str",  // The role of the substep
          "content": "str"  // The content of this substep
        }
      ]
    }
  ]
}

IMPORTANT ACCESS PATTERNS:
- To get instruction: trajectory["instruction"]
- To access step N: trajectory["steps"][N-1] (steps list is 0-indexed, but step.index is 1-based)
- To get substeps of a step: step.get("substeps", [])
- To get content: substep.get("content")
- To get role: substep.get("role")

================================================================================================
OUTPUT FORMAT
================================================================================================

Top-level JSON schema (EXACT):
{
  "step_num": <int>,
  "decision": "NO_INVARIANT" | "INVARIANT",
  "invariant":  [ <InvariantObject> ] | null,
  "trigger_to_invariants_map": {
    "agent:<AgentName>": ["assertion_name", "..."]
  }
}

where <AgentName> is one of: <<AGENT_UNION>>
  
================================================================================================
InvariantObject schema (EXACT)
================================================================================================
Each invariant object MUST have these keys (no extras):

IMPORTANT: Follow this schema EXACTLY. Do not add extra fields. Use correct JSON formatting.

{
  "assertion_name": "string_unique_snake_case",

   "taxonomy_targets": [
    "Instruction/PlanAdherenceFailure: The agent fails to follow the directions or the agreed plan by ignoring directives and skipping policy steps. This covers both under-execution (missed steps) and over-execution (unplanned or unnecessary actions, e.g., extra tool calls) that deviate from the static plan, domain policy or orchestrator plan.",
    "InventionOfNewInformation: The agent introduces, removes, or alters information that is not grounded in any available input, context, or tool output. This includes fabricating unsupported facts, hallucinating details, or omitting relevant information without justification.",
    "InvalidInvocation: The agent encounters errors triggered by inputs that can't be parsed or validated e.g., syntax errors or tool calls with bad/missing arguments. Not involving wrong logic; just invalid inputs.",
    "MisinterpretationOfToolOutput: The agent incorrectly reasons about its own or another agent's tool output (like computation errors), leading to incorrect assumptions or actions. This also includes cases where the agent considered only partial tool output.",
    "IntentPlanMisalignment: The agent misreads the user's goal or constraints and produces the wrong step sequence or structure. Covers both bad ordering/structure and plans aimed at the wrong objective.",
    "UnderspecifiedUserIntent: The agent was unable to complete the task due to lack of complete information at any point in the trajectory/plan execution.",
    "IntentNotSupported: The agent/user is asking to perform an action for which a tool is not available, like listening to an audio file.",
    "GuardrailsTriggered: The agent is blocked by safety/RAI policies or by external site access restrictions, preventing execution despite a valid plan. Examples include policy refusals (e.g., unsafe content, privacy-protected data), CAPTCHA/robot blocks, login/paywall/403/robots.txt denials, or site forbids automation. This is not an agent planning/execution error; it is an external/guardrail block.",
    "SystemFailure: The agent faces a system connectivity issue while calling a particular tool like an endpoint not being reachable."
    // Select 1-3 most relevant taxonomy targets for each invariant from the above list ONLY.
 
    // Example: ["Instruction/PlanAdherenceFailure", "GuardrailsTriggered"]
  ],

  "invariant_type": "SCHEMA  // Ensures a structured payload (embedded JSON, tool args/result, ledger blob) is parseable and contains required fields.
  | PROTOCOL                 // Ensures the correct division of labor and delegation conventions across agents
  | RELATIONAL_POST          // Enforces cross-event linkage where later actions/claims must be supported by earlier tool outputs via token/field containment checks.
  | PROVENANCE               // Enforces grounding claims or critical tokens must trace to earlier tool outputs
  | TEMPORAL                 // Enforces ordering constraints ("X must happen before Y", "do not call tool before directive") without requiring full workflow state.
  | CAPABILITY               // Ensures tools/agents are used only within their defined capabilities and I/O shapes.
  | ANY",

  "event_trigger": {
    "step_index": "*|int|range",
    "substep_index": "*|int|range",
    "role_name": "<<AGENT_UNION>>"  // agent names inferred from substep.role; "*" matches all
  },

  "check_hint": "deterministic procedure description in 2-8 sentences",
  "check_type": "python_check|nl_check",

  "python_check": {
    "function_name": "same_as_assertion_name",
    "args": ["trajectory","current_step_index"],
    "code_lines": [
      "def same_as_assertion_name(trajectory, current_step_index):",
      "    \\"\\"\\\"Return True iff invariant holds.\\"\\"\\\"",
      "    # Access steps via trajectory['steps'][current_step_index]",
      "    # Each step has 'substeps' with 'role' and 'content'",
      "    # MUST include at least one explicit failure path: return False",
      "    return True"
    ]
  },

  "nl_check": {
    "judge_system_prompt_template": "<<NL_CHECK_JUDGE_SYSTEM_PROMPT>>",
    "judge_user_prompt_template": "template using {POLICY_TEXT} {TASK_INSTRUCTION} {CURRENT_EVENT_JSON} {WINDOW_EVENTS_JSON}",
    "judge_scope_notes": "what events are in scope and what counts as evidence",
    "focus_steps_instruction": "REQUIRED: Clear, actionable instruction identifying which 2-4 specific events to examine. Must specify events by relative position (e.g., 'immediately prior user message', '2 steps back', 'most recent get_order_details result') and explain what to look for in each.",
    "judge_rubric": ["objective criterion 1", "objective criterion 2", "..."],
    "rubric_evaluation_algorithm_template": "<<RUBRIC_EVALUATION_ALGORITHM>>",
    "output_format_template": "<<OUTPUT_FORMAT>>"
  }
}

## Quality Guidelines:
- **Policy-Derived**: Must be supported by the policy document content, include any user confirmation checks before critical actions if specified in policy
- **Practical Impact**: Focus on assertions that prevent real operational problems
- **Domain-Relevant**: Prioritize rules specific to this business domain
- **Reasonable Scope**: Avoid both trivial sanity checks and overly complex checks that would be difficult to implement reliably.

## Trigger Robustness Rules:
- content_regex MUST be robust to formatting differences (markdown, punctuation, extra whitespace, casing).
- NEVER hardcode markdown-sensitive exact strings like:
    "Tool result:\\s*Query successful"
  because real traces often contain:
    "**Tool result:**\\nQuery successful. ..."
- Prefer matching stable substrings that survive formatting changes.
- Prefer SHORT regexes. Avoid overfitting to ":" vs "**:**" vs "." variations.
- Avoid brittle patterns that depend on exact formatting or punctuation.
- Make sure to match the role name perfectly and overapproximate if you are unsure.

Each invariant should be as GENERAL as possible, if many specific invariants can be combined into one invariant function, do so.
For example, an invariant which checks order status validity for various tool calls can be combined into one function.

================================================================================================
DECISION GUIDE: PYTHON_CHECK VS NL_CHECK
================================================================================================
Use python_check when:
- Tool output is structured JSON with extractable fields (e.g., get_order_details, get_user_details)
- Check involves counting, summing, or comparing numeric values
- Check involves verifying exact field matches (IDs, status codes, amounts, dates)
- Check involves parsing and validating data structure or schema
- Check uses simple regex patterns (e.g., matching an order_id format like r"#W\\d+" or extracting keys or specific count in the content message)
- Check involves deterministic string containment or exact substring matching

Use nl_check when:
- Check requires understanding paraphrasing or semantic equivalence (e.g., "bigger size" = "larger size")
- Check requires judging whether two different phrasings express the same intent
- Check requires evaluating whether user agent messages align with task instruction semantically
- The content is unstructured natural language without clear fields to extract
- Check would require a long list of synonyms or variations to cover programmatically

A naive python_check approach would try to match exact phrases like:
  - "I want to return"
  - "I'd like to exchange"
  - "cancel my order"
  - "modify my address"
  - "get a refund"
  - "change the items"
But users express the same intent in many ways:
  - "Could you help me send back..." (return)
  - "I need to swap this for..." (exchange)
  - "Please void the order" (cancel)
  - "Update my shipping info" (modify address)
Matching exact phrases is brittle and incomplete. Use nl_check for semantic intent.

================================================================================================
## PYTHON CHECK GUIDELINES
================================================================================================

## Python Code Guidelines:
- Write complete, executable Python functions
- You will be given 2 parameters - (trajectory and current_step_index), where trajectory is the full IR dict with "steps" field
- Access steps via trajectory["steps"][current_step_index] - steps list is 0-indexed
- Each step has "index" (1-based) and "substeps" array
- Each substep has "sub_index", "role", and "content"
- The current_step_index is 0-based. So if the current step being executed has index=7, current_step_index will be 6.
- Include docstrings explaining the assertion
- Look at the tool response key fields very carefully while writing the code.
  1. First, identify which tool(s) you're working with in the current trajectory step
  2. Pay attention to nested dictionaries (use chained .get() calls) and list structures (check length before indexing)
  3. Note the data types (str, float, bool, list, dict) to avoid type errors
  4. Use ONLY the exact field names shown in the tool structure - do NOT invent or assume field names
- Handle edge cases and missing data gracefully - verify field existence and type before operations
- Focus on computation validation and step-specific checks, avoid very trivial sanity checks
- **Exception Handling**: Raise exceptions for safety check violations (KeyError, IndexError, TypeError, AttributeError, ValueError, etc.) instead of returning False. Let these exceptions propagate naturally. Only return boolean (True/False) for actual invariant logic violations
- Return boolean values (True if assertion passes, False if invariant violation detected)
- **Add print statements for debugging**: Start with printing the function name, then print all key variables, extracted JSON fields, and intermediate calculation results to aid in debugging and verification

================================================================================================
## PYTHON_CHECK ANTI-HALLUCINATION RULES
================================================================================================
- A python_check MUST implement ONLY the exact condition(s) described in this invariant's `check_hint`.
- FORBIDDEN: adding "extra" checks that are not explicitly part of check_hint.
  Examples of forbidden extras:
  * checking additional URLs/IDs/regexes beyond the targeted substring(s)
  * inferring "plan rules" not explicitly spelled out in check_hint
  * scanning many steps for unrelated markers "just in case"
  * inventing secondary conditions like "also must not contain X" unless required by check_hint
- Keep python_check SMALL and TARGETED:
  * prefer 1-3 simple extractions + 1 boolean condition
  * avoid long keyword lists and complex regex (if you need that, it is probably a nl_check candidate)

================================================================================================
## NL CHECK GUIDELINES
================================================================================================

For nl_check invariants, MUST include a "focus_steps_instruction" field that tells the judge which specific events to pay attention to.

The instruction should:
1. Be clear and actionable - the judge should know exactly which events to look at
2. Identify 2-4 critical events by their position relative to current_step_index
3. Explain WHY each event is important for the evaluation
4. Use relative positioning (e.g., "immediately prior", "2 steps back", "most recent X tool result")

EXAMPLES OF GOOD FOCUS STEPS INSTRUCTIONS:

For confirmation checks:
"Focus your evaluation on these key events: (1) The immediately prior message - check for explicit confirmation keywords like 'yes', 'confirm', 'proceed', 'ok'. (2) The assistant message two to three steps back - verify that it describes the action of interest (cancel/modify/exchange/return) and mentions the relevant target entity identifier(s) and critical parameters (e.g., record/entity ID, amount, destination, items, scope). (3) The current tool call arguments - verify they match the order_id/incident_id and details mentioned by another agent."

For information invention checks:
"Focus on: (1) The current agent content message being evaluated. (2) The most recent relevant tool results in the window — extract the authoritative facts actually present (identifiers, key fields/attributes, timestamps, computed outputs). (3) Compare specific claims in the agent's message against these tool results."

For temporal/prerequisite checks:
"Focus on: (1) The current tool call event — identify what operation is being attempted and what prerequisites it implies (authentication, entity lookup, permission check, required context fetch, state precondition). (2) Scan backward through all prior events to find the most recent event(s) that satisfy each prerequisite. (3) Most recent state snapshot for the same target entity — locate the latest tool result that reports the entity's current state."

KEY PRINCIPLES:
- Be specific about WHICH events (by relative position or role+name)
- Explain WHAT to look for in each event
- Make it easy for the judge to locate and examine the right evidence
- Reduce cognitive load by narrowing focus to what matters most

================================================================================================
COMPLETE NL_CHECK INVARIANT EXAMPLE
================================================================================================
Below is a complete example of an nl_check invariant. Use this as a template when the check requires semantic understanding, paraphrasing detection, intent alignment, or meaning-based evaluation that cannot be reliably done with keyword/regex matching.

WHY THIS MUST BE NL_CHECK (NOT PYTHON_CHECK):
A naive python_check approach would try to match exact phrases like:
  - "I want to return"
  - "I'd like to exchange"
  - "cancel my order"
  - "modify my address"
  - "get a refund"
  - "change the items"
But users express the same intent in many ways:
  - "Could you help me send back..." (return)
  - "I need to swap this for..." (exchange)
  - "Please void the order" (cancel)
  - "Update my shipping info" (modify address)
Matching exact phrases is brittle and incomplete. Use nl_check for semantic intent.

Here's a complete example:
{
  "assertion_name": "user_agent_follows_task_instruction",
  "taxonomy_targets": ["UnderspecifiedUserIntent"],
  "invariant_type": "USER_INTENT_CHECK",
  "check_phase": "ANY",
  "event_trigger": {
    "step_index": "*",
    "substep_index": "*",
    "role_name": "user",
    "content_regex": "*",
    "tool_name": "*"
  },
  "check_hint": "Verify that the simulated user agent's message is semantically consistent with the original task instruction. Check that the user is not introducing new requirements, contradicting the task instruction, or providing information that deviates from what was specified in the task.",
  "check_type": "nl_check",
  "python_check": {},
  "nl_check": {
    "judge_system_prompt_template": "You are a strict compliance judge. Evaluate only with evidence in the provided events and the task instruction. Do not infer intent beyond explicit statements. CRITICAL: If required evidence is missing or ambiguous, mark the criterion as UNCLEAR.",
    "judge_user_prompt_template": "POLICY TEXT: {POLICY_TEXT}\\n\\nTASK INSTRUCTION: {TASK_INSTRUCTION}\\n\\nCURRENT USER MESSAGE: {CURRENT_EVENT_JSON}\\n\\nEvaluate whether the user agent's message is consistent with the task instruction.",
    "judge_scope_notes": "Compare the user agent's current message against the original task instruction to detect semantic drift, contradictions, or invention of new requirements not in the task.",
    "focus_steps_instruction": "Focus on: (1) The current user message being evaluated - extract the specific request, preferences, or information being communicated. (2) The original task instruction provided at the start - identify the user's stated goals, constraints, and preferences. (3) Check for semantic alignment: is the user asking for what the task instruction specifies, or has the user agent drifted to a different request?",
    "judge_rubric": [
      "The user agent's request type (return/exchange/cancel/modify) matches what is specified in the task instruction",
      "If the user mentions specific items, quantities, or attributes, they are consistent with those in the task instruction",
      "The user agent does not introduce new constraints or preferences that were not present in the task instruction",
      "The user agent does not contradict or reverse any requirement from the task instruction"
    ],
    "rubric_evaluation_algorithm_template": "{RUBRIC_EVALUATION_ALGORITHM}",
    "output_format_template": "{OUTPUT_FORMAT}"
  }
}

YOU NEED NOT GENERATE A DYNAMIC INVARIANT FOR EVERY STEP. IF NO NEW DYNAMIC INVARIANTS ARE NEEDED OR ARE ALREADY COVERED BY STATIC INVARIANTS, OUTPUT:

{
  "step_num": <<STEP_NUM>>,
  "decision": "NO_INVARIANT",
  "invariant": null,
  "trigger_to_invariants_map": {}
}

================================================================================================
STATIC INVARIANTS ALREADY GENERATED
================================================================================================

<<STATIC_INVARIANTS>>

================================================================================================
DYNAMIC INVARIANTS GENERATED FOR PREVIOUS STEPS
================================================================================================

<<DYNAMIC_INVARIANTS_ALREADY_GENERATED>>

================================================================================================
TRAJECTORY STEPS TILL NOW
================================================================================================

<<TRAJECTORY_UP_TO_CURRENT_STEP>>"""


# ------------------------------------------------------------------------------------
# ONE-SHOT PROMPT TEMPLATES
# ------------------------------------------------------------------------------------
# These templates are used by build_one_shot_prompt() for one-shot mode.
# Key differences from step-by-step templates above:
#   - No <<STEP_NUM>> or <<DYNAMIC_INVARIANTS_ALREADY_GENERATED>> placeholders
#   - Uses <<TRAJECTORY>> (full trajectory) instead of <<TRAJECTORY_UP_TO_CURRENT_STEP>>
#   - "YOUR TASK (ONE-SHOT)" section for multi-step analysis
#   - Focus Areas: Precondition, Sequential Dependencies, Business Rule, Single-Use
#   - No substep_index in event_trigger schema
#   - No NO_INVARIANT fallback section

DYNAMIC_INVARIANT_PROMPT_ONESHOT_WITHOUT_NL_CHECK = """
You are an expert at analyzing agent trajectories and generating step-specific dynamic invariants for trajectory verification.
Your task is to analyze the current step in an agent's execution trajectory and determine if any invariant (Python or Natural Language) should be generated for verification for this specific step.

**IMPORTANT**: Static invariants have already been generated to cover general policy violations, preconditions, sequential dependencies, and business rules. Focus ONLY on cases NOT covered by static invariants, such as:

You will be provided with the overall task instruction, the static invariants already generated (SO YOU DON'T REPEAT THEM), and the FULL agent trajectory.

Generate ONLY high-signal invariants.

**CRITICAL**: For each invariant, choose the BEST check type:
- Use **python_check** for structured data validation, JSON parsing, numeric calculations, deterministic string matching, and field extraction.
- Use **nl_check** for semantic intent evaluation, paraphrasing equivalence, natural language understanding, and flexible user confirmation verification
- More information provided on DECISION GUIDE below. 

## Context Information:
**Task Instruction:** <<TASK_INSTRUCTION>>

## YOUR TASK (ONE-SHOT):
Analyze the FULL trajectory and output a SET of dynamic invariants. Each invariant MUST:
1) Be verifiable from the available trajectory data (deterministically for python_check; rubric-based for nl_check if enabled)
2) Not already covered by any static invariant
3) Be high-signal for catching likely failures (computation/aggregation, cross-tool consistency, provenance, temporal prerequisites)
4) Be step-specific via event_trigger so it triggers only where it matters

IMPORTANT:
- You are NOT limited to a single step. You may emit invariants for multiple steps in this ONE response.
- Return a JSON array of invariants with the schema specified below.

## Your Task:
Analyze the full trajectory and generate dynamic invariants that:
1. Are simple enough to implement via python or NL checks.
2. Can be checked at this specific step in the trajectory (e.g., computation checks from tool call results)
3. Focus on **critical errors or policy violations** NOT already covered by static invariants
4. Are objectively verifiable from trajectory data
5. Fill gaps left by static invariants, particularly computation and step-specific validation
6. Provide working Python code or Natural Language (NL) check for each dynamic assertion that can be executed

## FOCUS AREAS:
Generate invariants that would DETECT the following failure patterns at ANY step in the trajectory:

### **Computation Checks**
- Verify numerical calculations, aggregations, and data accuracy.
- Examples:
  - Financial: "Transaction total should equal sum of line items plus applicable fees"
  - Inventory: "Remaining stock after reservation should be original stock minus reserved quantity"
  - Healthcare: "Medication dosage calculations should be within safe therapeutic ranges"

### **Context-Specific Business Logic**
- Step-specific rules that depend on current trajectory state
- Examples:
  - Retail: "Discount calculations based on current cart contents and user loyalty level"
  - Financial: "Credit limit checks based on current account status and transaction history"
  - Healthcare: "Drug interaction checks based on current prescription list"
  - Workflow: "Approval requirements based on current user role and request amount"

### **Cross-Tool Data Validation**
- Verify data integrity across multiple tool interactions in the same session
- Examples:
  - "Entity identifiers referenced in later steps must match those retrieved in earlier tool outputs"
  - Retail: "Items added to cart must still be available when proceeding to checkout"
  - "If the agent claims an action succeeded, subsequent tool outputs must show evidence consistent with the claim"
  - Financial: "Exchange rates used in calculations must be consistent across related transactions"
  - Healthcare: "Lab results referenced in diagnosis must match the lab results retrieved"
  - System: "File permissions checked must match the permissions set in previous operations"

### **Sequential Dependencies** 
Required ordering between investigation steps and communication steps
- Examples:
  - Incident Management:: "IncidentAgent (fetch incident description) must precede KustoAgent (run diagnostics) for the same incident."
  - General: "User authentication must precede any data access operations"
  - Retail: "User must give explicit confirmation (yes) before any consequential database write actions"

### **Business Rule Constraints**
- Domain-specific operational limits and rules. Incident-management operational limits, safety/compliance constraints, and evidence-grounding rules
- Examples:
  - Incident Management: "Incident updates (or equivalent external communication steps) must be grounded in prior tool outputs (e.g., Kusto results, incident descriptions) within the same trajectory."
  - Incident Management: "IncidentAgent must only return incident descriptions (no extra troubleshooting); KustoAgent must only execute/provide query results (no invented interpretation)."
  - Retail: "Items can only be exchanged within same product category"
  - Healthcare: "Prescriptions require valid medical license verification"
  - Financial: "Transfer amounts cannot exceed daily limits"
  - Legal: "Document modifications require authorized personnel only"

### **Single-Use Restrictions**
- Actions that should occur at most once per incident (or once per approval) to avoid duplicate/unsafe operations
- Examples:
  - Incident Management: "Do not re-run the same Kusto query repeatedly without a stated reason (e.g., changed time window, different cluster, new hypothesis)."
  - Healthcare: "Critical procedure authorization can only be granted once"
  - Financial: "Account closure can only be initiated once per request"
  - System: "Database migration can only be executed once per maintenance window"
 
Output MUST be valid JSON only (no markdown, no commentary), with EXACT keys and structure defined below.

CRITICAL FORMATTING REQUIREMENTS:
1. Output ONLY valid JSON - NO markdown code blocks (no ```json or ```)
2. Use double quotes for all strings (not single quotes)
3. Ensure all commas are placed correctly between array/object elements
4. Ensure all brackets and braces are properly closed
5. Follow the EXACT schema below - do not add or remove fields
6. Validate your JSON before outputting
7. TAXONOMY TARGETS: Use ONLY values from the allowed list.
8. ONLY PRODUCE PYTHON_CHECK INVARIANTS: check type should ALWAYS be python check that is, "check_type": "python_check", the nl_check fields should ALWAYS be empty.

================================================================================================
TRAJECTORY FORMAT (IMPORTANT) - THIS WILL BE GIVEN AS AN INPUT PARAMETER
================================================================================================

TRAJECTORY:
The runtime provides a trajectory dict in the following IR schema:

{
  "trajectory_id": "str",
  "instruction": "str",  // The overall task instruction
  "steps": [
    {
      "index": int,  // 1-based step index
      "substeps": [
        {
          "sub_index": int,  // 1-based substep index within the step
          "role": "str",  // The role of the substep
          "content": "str"  // The content of this substep
        }
      ]
    }
  ]
}

IMPORTANT ACCESS PATTERNS:
- To get instruction: trajectory["instruction"]
- To access step N: trajectory["steps"][N-1] (steps list is 0-indexed, but step.index is 1-based)
- To get substeps of a step: step.get("substeps", [])
- To get content: substep.get("content")
- To get role: substep.get("role")

================================================================================================
OUTPUT FORMAT
================================================================================================

Top-level JSON schema (EXACT):
{
  "step_num": <int>,
  "decision": "NO_INVARIANT" | "INVARIANT",
  "invariant":  [ <InvariantObject> ] | null,
  "trigger_to_invariants_map": {
    "agent:<AgentName>": ["assertion_name", "..."]
  }
}

where <AgentName> is one of: <<AGENT_UNION>>
  
================================================================================================
InvariantObject schema (EXACT)
================================================================================================
Each invariant object MUST have these keys (no extras):

IMPORTANT: Follow this schema EXACTLY. Do not add extra fields. Use correct JSON formatting.

{
  "assertion_name": "string_unique_snake_case",

  "taxonomy_targets": [
    "Instruction/PlanAdherenceFailure: The agent fails to follow the directions or the agreed plan by ignoring directives and skipping policy steps. This covers both under-execution (missed steps) and over-execution (unplanned or unnecessary actions, e.g., extra tool calls) that deviate from the static plan, domain policy or orchestrator plan.",
    "InventionOfNewInformation: The agent introduces, removes, or alters information that is not grounded in any available input, context, or tool output. This includes fabricating unsupported facts, hallucinating details, or omitting relevant information without justification.",
    "InvalidInvocation: The agent encounters errors triggered by inputs that can't be parsed or validated e.g., Kusto syntax errors or tool calls with bad/missing arguments. Not involving wrong logic; just invalid inputs.",
    "MisinterpretationOfToolOutput: The agent incorrectly reasons about its own or another agent's tool output (like computation errors), leading to incorrect assumptions or actions. This also includes cases where the agent considered only partial tool output.",
    "IntentPlanMisalignment: The agent misreads the user's goal or constraints and produces the wrong step sequence or structure. Covers both bad ordering/structure and plans aimed at the wrong objective.",
    "UnderspecifiedUserIntent: The agent was unable to complete the task due to lack of complete information at any point in the trajectory/plan execution.",
    "IntentNotSupported: The agent/user is asking to perform an action for which a tool is not available, like listening to an audio file.",
    "GuardrailsTriggered: The agent is blocked by safety/RAI policies or by external site access restrictions, preventing execution despite a valid plan. Examples include policy refusals (e.g., unsafe content, privacy-protected data), CAPTCHA/robot blocks, login/paywall/403/robots.txt denials, or site forbids automation. This is not an agent planning/execution error; it is an external/guardrail block.",
    "SystemFailure: The agent faces a system connectivity issue while calling a particular tool like an endpoint not being reachable."
    // Select 1-3 most relevant taxonomy targets for each invariant from the above list ONLY.

    // Example: ["Instruction/PlanAdherenceFailure", "GuardrailsTriggered"]
  ],

  "invariant_type": "SCHEMA  // Ensures a structured payload (embedded JSON, tool args/result, ledger blob) is parseable and contains required fields.
  | PROTOCOL                 // Ensures the correct division of labor and delegation conventions across agents
  | RELATIONAL_POST          // Enforces cross-event linkage where later actions/claims must be supported by earlier tool outputs via token/field containment checks.
  | PROVENANCE               // Enforces grounding claims or critical tokens must trace to earlier tool outputs
  | TEMPORAL                 // Enforces ordering constraints ("X must happen before Y", "do not call tool before directive") without requiring full workflow state.
  | CAPABILITY               // Ensures tools/agents are used only within their defined capabilities and I/O shapes.
  | ANY",

  "event_trigger": {
    "step_index": "*|int|range",
    "role_name": "<<AGENT_UNION>>"  // agent names inferred from substep.role; "*" matches all
    },

  "check_hint": "deterministic procedure description in 2-8 sentences",
  "check_type": "python_check|nl_check",

  "python_check": {
    "function_name": "same_as_assertion_name",
    "args": ["trajectory","current_step_index"],
    "code_lines": [
      "def same_as_assertion_name(trajectory, current_step_index):",
      "    \"\"Return True iff invariant holds.\"\"",
      "    # Access steps via trajectory['steps'][current_step_index]",
      "    # Each step has 'substeps' with 'role' and 'content'",
      "    # MUST include at least one explicit failure path: return False",
      "    return True"
    ]
  },

    "nl_check": {
    ALWAYS EMPTY
  }
}

## Quality Guidelines:
- **Policy-Derived**: Must be supported by the policy document content, include any user confirmation checks before critical actions if specified in policy
- **Practical Impact**: Focus on assertions that prevent real operational problems
- **Domain-Relevant**: Prioritize rules specific to this business domain
- **Reasonable Scope**: Avoid both trivial sanity checks and computation checks better suited for dynamic invariants.

## Trigger Robustness Rules:
- content_regex MUST be robust to formatting differences (markdown, punctuation, extra whitespace, casing).
- NEVER hardcode markdown-sensitive exact strings like:
    "Kusto result:\s*Query successful"
  because real traces often contain:
    "**Kusto result:**\nQuery successful. ..."
- Prefer matching stable substrings that survive formatting changes.
- Prefer SHORT regexes. Avoid overfitting to ":" vs "**:**" vs "." variations.
- Avoid brittle patterns that depend on exact formatting or punctuation.
- Make sure to match the role name perfectly and overapproximate if you are unsure.

Each invariant should be as GENERAL as possible, if many specific invariants can be combined into one invariant function, do so.
For example, an invariant which checks order status validity for various tool calls can be combined into one function.

================================================================================================
## PYTHON CHECK GUIDELINES
================================================================================================

## Python Code Guidelines:
- Write complete, executable Python functions
- You will be given 2 parameters - (trajectory and current_step_index), where trajectory is the full IR dict with "steps" field
- Access steps via trajectory["steps"][current_step_index] - steps list is 0-indexed
- Each step has "index" (1-based) and "substeps" array
- Each substep has "sub_index", "role", and "content"
- The current_step_index is 0-based. So if the current step being executed has index=7, current_step_index will be 6.
- Include docstrings explaining the assertion
- Look at the tool response key fields very carefully while writing the code.
  1. First, identify which tool(s) you're working with in the current trajectory step
  2. Pay attention to nested dictionaries (use chained .get() calls) and list structures (check length before indexing)
  3. Note the data types (str, float, bool, list, dict) to avoid type errors
  4. Use ONLY the exact field names shown in the tool structure - do NOT invent or assume field names
- Handle edge cases and missing data gracefully - verify field existence and type before operations
- Focus on computation validation and step-specific checks, avoid very trivial sanity checks
- **Exception Handling**: Raise exceptions for safety check violations (KeyError, IndexError, TypeError, AttributeError, ValueError, etc.) instead of returning False. Let these exceptions propagate naturally. Only return boolean (True/False) for actual invariant logic violations
- Return boolean values (True if assertion passes, False if invariant violation detected)
- **Add print statements for debugging**: Start with printing the function name, then print all key variables, extracted JSON fields, and intermediate calculation results to aid in debugging and verification


================================================================================================
## PYTHON_CHECK ANTI-HALLUCINATION RULES
================================================================================================
- A python_check MUST implement ONLY the exact condition(s) described in this invariant's `check_hint`.
- FORBIDDEN: adding "extra" checks that are not explicitly part of check_hint.
  Examples of forbidden extras:
  * checking additional URLs/IDs/regexes beyond the targeted substring(s)
  * inferring "plan rules" not explicitly spelled out in check_hint
  * scanning many steps for unrelated markers "just in case"
  * inventing secondary conditions like "also must not contain X" unless required by check_hint
- Keep python_check SMALL and TARGETED:
  * prefer 1-3 simple extractions + 1 boolean condition
  * avoid long keyword lists and complex regex (if you need that, it is probably a nl_check candidate)

================================================================================================
COMPLETE PYTHON_CHECK INVARIANT EXAMPLE
================================================================================================

{
    "assertion_name": "backpack_available_options_match_variants_count",
    "taxonomy_targets": [
        "MisinterpretationOfToolOutput",
    ],
    "invariant_type": "RELATIONAL_POST",
    "event_trigger": {
        "step_index": 5,
        "role_name": "assistant"
    },
    "check_hint": "When the assistant now reports how many Backpack options are available, it should compute the count from the variants field of the latest get_product_details result for the Backpack product. Specifically, count how many variant entries have available == true and verify that this equals the numeric count stated by the assistant for 'available' Backpack options. If the assistant gives an 'available options' count that does not match this filtered count, or gives any numeric options count without a corresponding get_product_details result, the invariant fails.",
    "check_type": "python_check",
    "python_check": {
        "function_name": "backpack_available_options_match_variants_count",
        "args": [
        "trajectory",
        "current_step_index"
        ],
        "code_lines": [
            "import re",
            "import json",
            "",
            "def backpack_available_options_match_variants_count(trajectory, current_step_index):",
            "    '''Verify that the assistant's stated count of available Backpack options matches the tool response.'''",
            "    print('Function: backpack_available_options_match_variants_count')",
            "    ",
            "    # Access the current step from the trajectory IR format",
            "    steps = trajectory.get('steps', [])",
            "    if current_step_index >= len(steps):",
            "        raise IndexError(f'current_step_index {current_step_index} out of bounds for {len(steps)} steps')",
            "    ",
            "    current_step = steps[current_step_index]",
            "    substeps = current_step.get('substeps', [])",
            "    ",
            "    # Find assistant substep content",
            "    assistant_content = None",
            "    for ss in substeps:",
            "        if ss.get('role') == 'assistant':",
            "            assistant_content = ss.get('content')",
            "            break",
            "    ",
            "    if assistant_content is None:",
            "        raise KeyError('Assistant content at current step is missing.')",
            "    print(f'Assistant content at step {current_step_index}: {assistant_content}')",
            "    ",
            "    # Try to extract a number near the word 'available'",
            "    match = re.search(r'(\d+)\s+(?:available)', assistant_content, flags=re.IGNORECASE)",
            "    if match:",
            "        stated_count = int(match.group(1))",
            "    else:",
            "        # Fallback: extract the first integer in the content",
            "        any_num = re.search(r'(\d+)', assistant_content)",
            "        if not any_num:",
            "            raise ValueError('No numeric count found in assistant content to verify.')",
            "        stated_count = int(any_num.group(1))",
            "    ",
            "    print(f'Extracted stated_count: {stated_count}')",
            "    ",
            "    # Find the most recent get_product_details tool response prior to this step",
            "    tool_response_content = None",
            "    for idx in range(current_step_index - 1, -1, -1):",
            "        step = steps[idx]",
            "        for ss in step.get('substeps', []):",
            "            if ss.get('role') == 'tool':",
            "                # Check if previous assistant step called get_product_details",
            "                # Look at the assistant step before this tool response",
            "                if idx > 0:",
            "                    prev_step = steps[idx - 1]",
            "                    for prev_ss in prev_step.get('substeps', []):",
            "                        if prev_ss.get('role') == 'assistant':",
            "                            try:",
            "                                calls = json.loads(prev_ss.get('content', ''))",
            "                                if isinstance(calls, list):",
            "                                    for call in calls:",
            "                                        if call.get('function', {}).get('name') == 'get_product_details':",
            "                                            tool_response_content = ss.get('content')",
            "                                            print(f'Found get_product_details tool response at step {idx}')",
            "                                            break",
            "                            except json.JSONDecodeError:",
            "                                pass",
            "                if tool_response_content:",
            "                    break",
            "        if tool_response_content:",
            "            break",
            "    ",
            "    if tool_response_content is None:",
            "        raise KeyError('No prior get_product_details tool response found for verification.')",
            "    ",
            "    print(f'Raw tool response content: {tool_response_content}')",
            "    ",
            "    # Parse JSON content",
            "    try:",
            "        tool_response = json.loads(tool_response_content)",
            "    except Exception as e:",
            "        raise ValueError(f'Tool response content is not valid JSON: {e}')",
            "    ",
            "    variants = tool_response.get('variants')",
            "    if variants is None:",
            "        raise KeyError('Missing variants in get_product_details tool response.')",
            "    if not isinstance(variants, dict):",
            "        raise TypeError('variants should be a dict.')",
            "    ",
            "    # Compute available count",
            "    available_count = 0",
            "    for k, v in variants.items():",
            "        if not isinstance(v, dict):",
            "            print(f'Skipping variant {k}: not a dict')",
            "            continue",
            "        avail_flag = v.get('available', None)",
            "        print(f'Variant {k} available flag: {avail_flag}')",
            "        if avail_flag is True:",
            "            available_count += 1",
            "    ",
            "    print(f'Computed available_count from tool response: {available_count}')",
            "    ",
            "    result = (stated_count == available_count)",
            "    print(f'Assertion result (stated == computed): {result}')",
            "    return result"
        ]
    },
    "nl_check": {}
}

================================================================================================
STATIC INVARIANTS ALREADY GENERATED
================================================================================================

<<STATIC_INVARIANTS>>


================================================================================================
TRAJECTORY
================================================================================================

<<TRAJECTORY>>"""

DYNAMIC_INVARIANT_PROMPT_ONESHOT = """
You are an expert at analyzing agent trajectories and generating step-specific dynamic invariants for trajectory verification.
Your task is to analyze the current step in an agent's execution trajectory and determine if any invariant (Python or Natural Language) should be generated for verification for this specific step.

**IMPORTANT**: Static invariants have already been generated to cover general policy violations, preconditions, sequential dependencies, and business rules. Focus ONLY on cases NOT covered by static invariants, such as:

You will be provided with the overall task instruction, the static invariants already generated (SO YOU DON'T REPEAT THEM), and the FULL agent trajectory.

Generate ONLY high-signal invariants.

**CRITICAL**: For each invariant, choose the BEST check type:
- Use **python_check** for structured data validation, JSON parsing, numeric calculations, deterministic string matching, and field extraction.
- Use **nl_check** for semantic intent evaluation, paraphrasing equivalence, natural language understanding, and flexible user confirmation verification
- More information provided on DECISION GUIDE below. 

## Context Information:
**Task Instruction:** <<TASK_INSTRUCTION>>

## YOUR TASK (ONE-SHOT):
Analyze the FULL trajectory and output a SET of dynamic invariants. Each invariant MUST:
1) Be verifiable from the available trajectory data (deterministically for python_check; rubric-based for nl_check if enabled)
2) Not already covered by any static invariant
3) Be high-signal for catching likely failures (computation/aggregation, cross-tool consistency, provenance, temporal prerequisites)
4) Be step-specific via event_trigger so it triggers only where it matters

IMPORTANT:
- You are NOT limited to a single step. You may emit invariants for multiple steps in this ONE response.
- Return a JSON array of invariants with the schema specified below.

## Your Task:
Analyze the full trajectory and generate dynamic invariants that:
1. Are simple enough to implement via python or NL checks.
2. Can be checked at this specific step in the trajectory (e.g., computation checks from tool call results)
3. Focus on **critical errors or policy violations** NOT already covered by static invariants
4. Are objectively verifiable from trajectory data
5. Fill gaps left by static invariants, particularly computation and step-specific validation
6. Provide working Python code or Natural Language (NL) check for each dynamic assertion that can be executed

## FOCUS AREAS:
Generate invariants that would DETECT the following failure patterns at ANY step in the trajectory:

### **Computation Checks**
- Verify numerical calculations, aggregations, and data accuracy.
- Examples:
  - Financial: "Transaction total should equal sum of line items plus applicable fees"
  - Inventory: "Remaining stock after reservation should be original stock minus reserved quantity"
  - Healthcare: "Medication dosage calculations should be within safe therapeutic ranges"

### **Context-Specific Business Logic**
- Step-specific rules that depend on current trajectory state
- Examples:
  - Retail: "Discount calculations based on current cart contents and user loyalty level"
  - Financial: "Credit limit checks based on current account status and transaction history"
  - Healthcare: "Drug interaction checks based on current prescription list"
  - Workflow: "Approval requirements based on current user role and request amount"

### **Cross-Tool Data Validation**
- Verify data integrity across multiple tool interactions in the same session
- Examples:
  - "Entity identifiers referenced in later steps must match those retrieved in earlier tool outputs"
  - Retail: "Items added to cart must still be available when proceeding to checkout"
  - "If the agent claims an action succeeded, subsequent tool outputs must show evidence consistent with the claim"
  - Financial: "Exchange rates used in calculations must be consistent across related transactions"
  - Healthcare: "Lab results referenced in diagnosis must match the lab results retrieved"
  - System: "File permissions checked must match the permissions set in previous operations"

### **Sequential Dependencies** 
Required ordering between investigation steps and communication steps
- Examples:
  - Incident Management:: "IncidentAgent (fetch incident description) must precede KustoAgent (run diagnostics) for the same incident."
  - General: "User authentication must precede any data access operations"
  - Retail: "User must give explicit confirmation (yes) before any consequential database write actions"

### **Business Rule Constraints**
- Domain-specific operational limits and rules. Incident-management operational limits, safety/compliance constraints, and evidence-grounding rules
- Examples:
  - Incident Management: "Incident updates (or equivalent external communication steps) must be grounded in prior tool outputs (e.g., Kusto results, incident descriptions) within the same trajectory."
  - Incident Management: "IncidentAgent must only return incident descriptions (no extra troubleshooting); KustoAgent must only execute/provide query results (no invented interpretation)."
  - Retail: "Items can only be exchanged within same product category"
  - Healthcare: "Prescriptions require valid medical license verification"
  - Financial: "Transfer amounts cannot exceed daily limits"
  - Legal: "Document modifications require authorized personnel only"

### **Single-Use Restrictions**
- Actions that should occur at most once per incident (or once per approval) to avoid duplicate/unsafe operations
- Examples:
  - Incident Management: "Do not re-run the same Kusto query repeatedly without a stated reason (e.g., changed time window, different cluster, new hypothesis)."
  - Healthcare: "Critical procedure authorization can only be granted once"
  - Financial: "Account closure can only be initiated once per request"
  - System: "Database migration can only be executed once per maintenance window"

Output MUST be valid JSON only (no markdown, no commentary), with EXACT keys and structure defined below.

CRITICAL FORMATTING REQUIREMENTS:
1. Output ONLY valid JSON - NO markdown code blocks (no ```json or ```)
2. Use double quotes for all strings (not single quotes)
3. Ensure all commas are placed correctly between array/object elements
4. Ensure all brackets and braces are properly closed
5. Follow the EXACT schema below - do not add or remove fields
6. Validate your JSON before outputting
7. TAXONOMY TARGETS: Use ONLY values from the allowed list.
8. NL_CHECK FIELDS FOR PYTHON_CHECK INVARIANTS: When "check_type": "python_check", the nl_check fields should be empty. Only populate nl_check with actual evaluation logic when "check_type": "nl_check".

================================================================================================
TRAJECTORY FORMAT (IMPORTANT) - THIS WILL BE GIVEN AS AN INPUT PARAMETER
================================================================================================

TRAJECTORY:
The runtime provides a trajectory dict in the following IR schema:

{
  "trajectory_id": "str",
  "instruction": "str",  // The overall task instruction
  "steps": [
    {
      "index": int,  // 1-based step index
      "substeps": [
        {
          "sub_index": int,  // 1-based substep index within the step
          "role": "str",  // The role of the substep
          "content": "str"  // The content of this substep
        }
      ]
    }
  ]
}

IMPORTANT ACCESS PATTERNS:
- To get instruction: trajectory["instruction"]
- To access step N: trajectory["steps"][N-1] (steps list is 0-indexed, but step.index is 1-based)
- To get substeps of a step: step.get("substeps", [])
- To get content: substep.get("content")
- To get role: substep.get("role")

================================================================================================
OUTPUT FORMAT
================================================================================================

Top-level JSON schema (EXACT):
{
  "step_num": <int>,
  "decision": "NO_INVARIANT" | "INVARIANT",
  "invariant":  [ <InvariantObject> ] | null,
  "trigger_to_invariants_map": {
    "agent:<AgentName>": ["assertion_name", "..."]
  }
}

where <AgentName> is one of: <<AGENT_UNION>>
  
================================================================================================
InvariantObject schema (EXACT)
================================================================================================
Each invariant object MUST have these keys (no extras):

IMPORTANT: Follow this schema EXACTLY. Do not add extra fields. Use correct JSON formatting.

{
  "assertion_name": "string_unique_snake_case",

  "taxonomy_targets": [
    "Instruction/PlanAdherenceFailure: The agent fails to follow the directions or the agreed plan by ignoring directives and skipping policy steps. This covers both under-execution (missed steps) and over-execution (unplanned or unnecessary actions, e.g., extra tool calls) that deviate from the static plan, domain policy or orchestrator plan.",
    "InventionOfNewInformation: The agent introduces completely new information that is not grounded in any available input. This includes fabricating unsupported facts, hallucinating details or summarizes information present in prior context.",
    "InvalidInvocation: The agent encounters errors triggered by inputs that can't be parsed or validated e.g., Kusto syntax errors or tool calls with bad/missing arguments. Not involving wrong logic; just invalid inputs.",
    "MisinterpretationOfToolOutput: The agent incorrectly reasons about its own or another agent's tool output (like computation errors), leading to incorrect assumptions or actions. This also includes cases where the agent considered only partial tool output.",
    "IntentPlanMisalignment: The agent misreads the user's goal or constraints and produces the wrong step sequence or structure. Covers both bad ordering/structure and plans aimed at the wrong objective.",
    "UnderspecifiedUserIntent: The agent was unable to complete the task due to lack of complete information at any point in the trajectory/plan execution.",
    "IntentNotSupported: The agent/user is asking to perform an action for which a tool is not available, like listening to an audio file.",
    "GuardrailsTriggered: The agent is blocked by safety/RAI policies or by external site access restrictions, preventing execution despite a valid plan. Examples include policy refusals (e.g., unsafe content, privacy-protected data), CAPTCHA/robot blocks, login/paywall/403/robots.txt denials, or site forbids automation. This is not an agent planning/execution error; it is an external/guardrail block.",
    "SystemFailure: The agent faces a system connectivity issue while calling a particular tool like an endpoint not being reachable."
    // Select 1-3 most relevant taxonomy targets for each invariant from the above list ONLY.
 
    // Example: ["Instruction/PlanAdherenceFailure", "GuardrailsTriggered"]
  ],

  "invariant_type": "SCHEMA  // Ensures a structured payload (embedded JSON, tool args/result, ledger blob) is parseable and contains required fields.
  | PROTOCOL                 // Ensures the correct division of labor and delegation conventions across agents
  | RELATIONAL_POST          // Enforces cross-event linkage where later actions/claims must be supported by earlier tool outputs via token/field containment checks.
  | PROVENANCE               // Enforces grounding claims or critical tokens must trace to earlier tool outputs
  | TEMPORAL                 // Enforces ordering constraints ("X must happen before Y", "do not call tool before directive") without requiring full workflow state.
  | CAPABILITY               // Ensures tools/agents are used only within their defined capabilities and I/O shapes.
  | ANY",

  "event_trigger": {
    "step_index": "*|int|range",
    "role_name": "<<AGENT_UNION>>"  // agent names inferred from substep.role; "*" matches all
    },

  "check_hint": "deterministic procedure description in 2-8 sentences",
  "check_type": "python_check|nl_check",

  "python_check": {
    "function_name": "same_as_assertion_name",
    "args": ["trajectory","current_step_index"],
    "code_lines": [
      "def same_as_assertion_name(trajectory, current_step_index):",
      "    \"\"Return True iff invariant holds.\"\"",
      "    # Access steps via trajectory['steps'][current_step_index]",
      "    # Each step has 'substeps' with 'role' and 'content'",
      "    # MUST include at least one explicit failure path: return False",
      "    return True"
    ]
  },

   "nl_check": {
    "judge_system_prompt_template": "<<NL_CHECK_JUDGE_SYSTEM_PROMPT>>",
    "judge_user_prompt_template": "template using {POLICY_TEXT} {TASK_INSTRUCTION} {CURRENT_EVENT_JSON} {WINDOW_EVENTS_JSON}",
    "judge_scope_notes": "what events are in scope and what counts as evidence",
    "focus_steps_instruction": "REQUIRED: Clear, actionable instruction identifying which 2-4 specific events to examine. Must specify events by relative position (e.g., 'immediately prior user message', '2 steps back', 'most recent get_order_details result') and explain what to look for in each. See FOCUS STEPS INSTRUCTION section for detailed examples.",
    "judge_rubric": ["objective criterion 1", "objective criterion 2", "..."],
    "rubric_evaluation_algorithm_template": "<<RUBRIC_EVALUATION_ALGORITHM>>",
    "output_format_template": "<<OUTPUT_FORMAT>>"
  }
}

## Quality Guidelines:
- **Policy-Derived**: Must be supported by the policy document content, include any user confirmation checks before critical actions if specified in policy
- **Practical Impact**: Focus on assertions that prevent real operational problems
- **Domain-Relevant**: Prioritize rules specific to this business domain
- **Reasonable Scope**: Avoid both trivial sanity checks and computation checks better suited for dynamic invariants.

## Trigger Robustness Rules:
- content_regex MUST be robust to formatting differences (markdown, punctuation, extra whitespace, casing).
- NEVER hardcode markdown-sensitive exact strings like:
    "Kusto result:\s*Query successful"
  because real traces often contain:
    "**Kusto result:**\nQuery successful. ..."
- Prefer matching stable substrings that survive formatting changes.
- Prefer SHORT regexes. Avoid overfitting to ":" vs "**:**" vs "." variations.
- Avoid brittle patterns that depend on exact formatting or punctuation.
- Make sure to match the role name perfectly and overapproximate if you are unsure.

Each invariant should be as GENERAL as possible, if many specific invariants can be combined into one invariant function, do so.
For example, an invariant which checks order status validity for various tool calls can be combined into one function.

================================================================================================
DECISION GUIDE: PYTHON_CHECK VS NL_CHECK
================================================================================================
Use python_check when:
- Tool output is structured JSON with extractable fields (e.g., get_order_details, get_user_details)
- Check involves counting, summing, or comparing numeric values
- Check involves verifying exact field matches (IDs, status codes, amounts, dates)
- Check involves parsing and validating data structure or schema
- Check uses simple regex patterns (e.g., matching an order_id format like r"#W\d+" or extracting keys or specific count in the content message)
- Check involves deterministic string containment or exact substring matching

Use nl_check when:
- Check requires understanding paraphrasing or semantic equivalence (e.g., "bigger size" = "larger size")
- Check requires judging whether two different phrasings express the same intent
- Check requires evaluating whether user agent messages align with task instruction semantically
- The content is unstructured natural language without clear fields to extract
- Check would require a long list of synonyms or variations to cover programmatically

A naive python_check approach would try to match exact phrases like:
  - "I want to return"
  - "I'd like to exchange"
  - "cancel my order"
  - "modify my address"
  - "get a refund"
  - "change the items"
But users express the same intent in many ways:
  - "Could you help me send back..." (return)
  - "I need to swap this for..." (exchange)
  - "Please void the order" (cancel)
  - "Update my shipping info" (modify address)
Matching exact phrases is brittle and incomplete. Use nl_check for semantic intent.

================================================================================================
## PYTHON CHECK GUIDELINES
================================================================================================

## Python Code Guidelines:
- Write complete, executable Python functions
- You will be given 2 parameters - (trajectory and current_step_index), where trajectory is the full IR dict with "steps" field
- Access steps via trajectory["steps"][current_step_index] - steps list is 0-indexed
- Each step has "index" (1-based) and "substeps" array
- Each substep has "sub_index", "role", and "content"
- The current_step_index is 0-based. So if the current step being executed has index=7, current_step_index will be 6.
- Include docstrings explaining the assertion
- Look at the tool response key fields very carefully while writing the code.
  1. First, identify which tool(s) you're working with in the current trajectory step
  2. Pay attention to nested dictionaries (use chained .get() calls) and list structures (check length before indexing)
  3. Note the data types (str, float, bool, list, dict) to avoid type errors
  4. Use ONLY the exact field names shown in the tool structure - do NOT invent or assume field names
- Handle edge cases and missing data gracefully - verify field existence and type before operations
- Focus on computation validation and step-specific checks, avoid very trivial sanity checks
- **Exception Handling**: Raise exceptions for safety check violations (KeyError, IndexError, TypeError, AttributeError, ValueError, etc.) instead of returning False. Let these exceptions propagate naturally. Only return boolean (True/False) for actual invariant logic violations
- Return boolean values (True if assertion passes, False if invariant violation detected)
- **Add print statements for debugging**: Start with printing the function name, then print all key variables, extracted JSON fields, and intermediate calculation results to aid in debugging and verification


================================================================================================
## PYTHON_CHECK ANTI-HALLUCINATION RULES
================================================================================================
- A python_check MUST implement ONLY the exact condition(s) described in this invariant's `check_hint`.
- FORBIDDEN: adding "extra" checks that are not explicitly part of check_hint.
  Examples of forbidden extras:
  * checking additional URLs/IDs/regexes beyond the targeted substring(s)
  * inferring "plan rules" not explicitly spelled out in check_hint
  * scanning many steps for unrelated markers "just in case"
  * inventing secondary conditions like "also must not contain X" unless required by check_hint
- Keep python_check SMALL and TARGETED:
  * prefer 1-3 simple extractions + 1 boolean condition
  * avoid long keyword lists and complex regex (if you need that, it is probably a nl_check candidate)

================================================================================================
## NL CHECK GUIDELINES
================================================================================================

For nl_check invariants, MUST include a "focus_steps_instruction" field that tells the judge which specific events to pay attention to.

The instruction should:
1. Be clear and actionable - the judge should know exactly which events to look at
2. Identify 2-4 critical events by their position relative to current_step_index
3. Explain WHY each event is important for the evaluation
4. Use relative positioning (e.g., "immediately prior", "2 steps back", "most recent X tool result")

EXAMPLES OF GOOD FOCUS STEPS INSTRUCTIONS:

For confirmation checks:
"Focus your evaluation on these key events: (1) The immediately prior message - check for explicit confirmation keywords like 'yes', 'confirm', 'proceed', 'ok'. (2) The assistant message two to three steps back - verify that it describes the action of interest (cancel/modify/exchange/return) and mentions the relevant target entity identifier(s) and critical parameters (e.g., record/entity ID, amount, destination, items, scope). (3) The current tool call arguments - verify they match the order_id/incident_id and details mentioned by another agent."

For information invention checks:
"Focus on: (1) The current agent content message being evaluated. (2) The most recent relevant tool results in the window — extract the authoritative facts actually present (identifiers, key fields/attributes, timestamps, computed outputs). (3) Compare specific claims in the agent's message against these tool results."

For temporal/prerequisite checks:
"Focus on: (1) The current tool call event — identify what operation is being attempted and what prerequisites it implies (authentication, entity lookup, permission check, required context fetch, state precondition). (2) Scan backward through all prior events to find the most recent event(s) that satisfy each prerequisite. (3) Most recent state snapshot for the same target entity — locate the latest tool result that reports the entity's current state."

KEY PRINCIPLES:
- Be specific about WHICH events (by relative position or role+name)
- Explain WHAT to look for in each event
- Make it easy for the judge to locate and examine the right evidence
- Reduce cognitive load by narrowing focus to what matters most

================================================================================================
COMPLETE NL_CHECK INVARIANT EXAMPLES
================================================================================================

## EXAMPLE 1 : Explicit User Confirmation Before Write Actions
{
    "assertion_name": "user_identity_matches_task_profile",
    "taxonomy_targets": [
        "UnderspecifiedUserIntent",
        "IntentPlanMisalignment"
    ],
    "invariant_type": "USER_INTENT_CHECK",
    "event_trigger": {
        "step_index": 3,
        "substep_index": "*",
        "role_name": "user",
        "content_regex": "Lei Li|85033",
        "tool_name": "*"
    },
    "check_hint": "Verify that the simulated user identity information provided in conversation (name and zip) is consistent with the task instruction's specified profile (Lei Li, zip 85033). This ensures the trajectory is using the intended user and not drifting to a different identity.",
    "examples": {
        "pass_scenario": "Task says the user is Lei Li with zip 85033; the user message gives 'my name is Lei Li, and my zip code is 85033.'",
        "fail_scenario": "Task says the user is Lei Li with zip 85033; the user message says 'my name is John Smith, and my zip code is 94105.'"
    },
    "check_type": "nl_check",
    "python_check": {},
    "nl_check": {
        "judge_system_prompt_template": "<<NL_CHECK_JUDGE_SYSTEM_PROMPT>>",
        "judge_user_prompt_template": "TASK_INSTRUCTION:{TASK_INSTRUCTION}

CURRENT USER EVENT:
{CURRENT_EVENT_JSON}

CONTEXT WINDOW EVENTS:
{WINDOW_EVENTS_JSON}

Evaluate whether the user identity details provided in the current user message (name and zip code) are consistent with the user profile specified in the task/instruction (name Lei Li and zip 85033).",
        "judge_scope_notes": "Compare the identity fields (name and zip) in the current user message against the user profile described in the task/instruction. Ignore other content; only identity consistency matters.",
        "focus_steps_instruction": "Focus on: (1) The task/instruction text to extract the expected user name and zip (Lei Li, 85033). (2) The current user message at this step that provides name and zip; identify exactly what name and zip are stated. Determine if both name and zip match the task profile or clearly differ.",
        "judge_rubric": [
        "The name stated in the current user message is semantically the same as the task instruction's user name (Lei Li).",
        "The zip code stated in the current user message is exactly the same as the task instruction's zip code (85033)."
        ],
        "rubric_evaluation_algorithm_template": "<<RUBRIC_EVALUATION_ALGORITHM>>",
        "output_format_template": "<<OUTPUT_FORMAT>>"
    }
}

================================================================================================
STATIC INVARIANTS ALREADY GENERATED
================================================================================================

<<STATIC_INVARIANTS>>

================================================================================================
TRAJECTORY
================================================================================================

<<TRAJECTORY>>"""


# Standard rubric evaluation algorithm template
# This is used to replace {RUBRIC_EVALUATION_ALGORITHM} placeholders in generated invariants
RUBRIC_EVALUATION_ALGORITHM = """
Rubric Evaluation Algorithm:
Step 1: For each criterion in the rubric, evaluate whether it can be CLEARLY judged as PASS or FAIL based solely on the provided events.
- Mark as CLEAR_PASS if the criterion is demonstrably satisfied by the evidence
- Mark as CLEAR_FAIL if the criterion is demonstrably violated by the evidence  
- Mark as UNCLEAR if: insufficient events to judge, criterion is ambiguous in this context, or pass/fail cannot be decisively determined

Step 2: Discard ALL criteria marked as UNCLEAR from consideration.

Step 3: Determine final verdict:
- If ANY remaining criterion is CLEAR_FAIL → return verdict 'fail'
- If ALL remaining criteria are CLEAR_PASS (or no criteria remain after discarding UNCLEAR) → return verdict 'pass'

Important: Only fail when you have CLEAR evidence of failure. When in doubt, mark as UNCLEAR and discard.
"""

# Standard output format template
# This is used to replace {OUTPUT_FORMAT} placeholders in generated invariants
OUTPUT_FORMAT = """
Output Format:
You MUST return a JSON response with this EXACT structure:
{
  "verdict": "pass" or "fail",
  "rubric_results": [
    {
      "criterion_index": 0,
      "criterion": "full text of the criterion from the rubric",
      "evaluation": "CLEAR_PASS" or "CLEAR_FAIL" or "UNCLEAR",
      "reasoning": "detailed explanation of why this evaluation was chosen, citing specific evidence from events"
    }
    // ... one entry per rubric criterion
  ],
  "final_reasoning": "explanation of how the verdict was determined from rubric_results, stating which criteria were discarded as UNCLEAR and which criteria drove the final decision"
}

This detailed output is critical for debugging and understanding the evaluation.
"""

# Standard judge system prompt for nl_check invariants
# This is used to replace {NL_CHECK_JUDGE_SYSTEM_PROMPT} placeholders in generated invariants
NL_CHECK_JUDGE_SYSTEM_PROMPT = """
You are a strict compliance judge. Evaluate only with evidence in the provided events and the policy text.
Do not infer intent beyond explicit statements.
CRITICAL: If required evidence is missing or ambiguous, mark the criterion as UNCLEAR.
When criteria are marked as UNCLEAR, they are discarded and do NOT contribute to a fail verdict.
If all criteria are UNCLEAR or CLEAR_PASS after discarding UNCLEAR ones, the final verdict is PASS.
Only mark as CLEAR_FAIL when there is definitive, unambiguous evidence of violation.
Ambiguity or missing evidence leads to PASS, not FAIL.
"""
# ------------------------------------------------------------------------------------
# DEBUG/FS HELPERS
# ------------------------------------------------------------------------------------
def dbg(msg: str) -> None:
    if DEBUG:
        print(f"[DEBUG] {msg}", flush=True)

def abspath_rel(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(BASE_DIR, p)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ------------------------------------------------------------------------------------
# PRETTY PRINT HELPERS
# ------------------------------------------------------------------------------------
def _json_dumps_pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False)

def pretty_json_text(text: str, *, default: str = "[]") -> str:
    s = (text or "").strip()
    if not s:
        return default
    try:
        return _json_dumps_pretty(json.loads(s))
    except Exception:
        # If it's not JSON, return raw (still useful for prompts/logs)
        return s

def pretty_json_obj(obj: Any, *, default: str = "[]") -> str:
    if obj is None:
        return default
    try:
        return _json_dumps_pretty(obj)
    except Exception:
        return str(obj)

def format_previous_static_assertions(static_blob: str) -> str:
    return pretty_json_text(static_blob, default="[]")

def format_previous_dynamic_assertions(dynamic_invariants: List[dict]) -> str:
    return pretty_json_obj(dynamic_invariants or [], default="[]")

# ------------------------------------------------------------------------------------
# CLIENT
# ------------------------------------------------------------------------------------

def extract_prompt_enums(
    traj: Union[Dict[str, Any], List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """
    Extract what can be derived from the Trajectory IR (no domain knowledge).

    Accepts either:
      - a single trajectory dict
      - OR a list of trajectories (we will use the first one by default)
    """
    # ---- coerce list -> single trajectory
    if isinstance(traj, list):
        if not traj:
            raise ValueError("trajectory list is empty")
        traj = traj[0]

    # ---- derive step indices
    step_indices: Set[int] = set()
    for step in traj.get("steps", []) or []:
        idx = step.get("index")
        if isinstance(idx, int):
            step_indices.add(idx)
    step_indices_sorted = sorted(step_indices)

    # ---- derive agent names from role (base part before '(')
    agents: Set[str] = set()
    for step in traj.get("steps", []) or []:
        for ss in step.get("substeps", []) or []:
            role = (ss.get("role") or "").strip()
            base = role.split("(", 1)[0].strip()
            if base:
                agents.add(base)
    agent_names = sorted(agents)

    # ---- build agent enum strings
    agent_union_parts = list(agent_names)
    agent_union_parts.append("*")

    agent_union = " | ".join(agent_union_parts) if agent_union_parts else "*"

    return {
        "step_indices": step_indices_sorted,
        "agent_union": agent_union,
    }



# ------------------------------------------------------------------------------------
# PROMPT BUILDERS (<<PLACEHOLDER>> substitution on mega-templates)
# ------------------------------------------------------------------------------------
def build_invariant_prompt(
    traj: Dict[str, Any],
    step_num: int,
    task_instruction: str,
    domain_policy: str,
    static_invariants: str,
    previous_assertions: str,
    trajectory_till_current_step: str,
    tools_list: Optional[List[str]] = None,
    tools_structure: Optional[Union[dict, str]] = None,
    include_nl_check: bool = True,
) -> str:
    """
    Build prompt for step-by-step dynamic invariant generation.
    Uses <<PLACEHOLDER>> substitution on mega-templates.

    Selects template based on include_nl_check:
      - False -> DYNAMIC_INVARIANT_PROMPT_WITHOUT_NL_CHECK (python-only)
      - True  -> DYNAMIC_INVARIANT_PROMPT (with nl_check + decision guide)
    """
    enums = extract_prompt_enums(traj)
    agent_union = enums["agent_union"]

    if include_nl_check:
        prompt = (
            DYNAMIC_INVARIANT_PROMPT
            .replace("<<TASK_INSTRUCTION>>", task_instruction)
            .replace("<<STATIC_INVARIANTS>>", static_invariants)
            .replace("<<STEP_NUM>>", str(step_num))
            .replace("<<DYNAMIC_INVARIANTS_ALREADY_GENERATED>>", previous_assertions)
            .replace("<<TRAJECTORY_UP_TO_CURRENT_STEP>>", trajectory_till_current_step)
            .replace("<<NL_CHECK_JUDGE_SYSTEM_PROMPT>>", NL_CHECK_JUDGE_SYSTEM_PROMPT.strip().replace('\n', '\\n'))
            .replace("<<RUBRIC_EVALUATION_ALGORITHM>>", RUBRIC_EVALUATION_ALGORITHM.strip().replace('\n', '\\n'))
            .replace("<<OUTPUT_FORMAT>>", OUTPUT_FORMAT.strip().replace('\n', '\\n'))
            .replace("<<AGENT_UNION>>", agent_union)
            .replace("<<TOOLS_LIST>>", json.dumps(tools_list, indent=2) if tools_list else "[]")
        )
    else:
        prompt = (
            DYNAMIC_INVARIANT_PROMPT_WITHOUT_NL_CHECK
            .replace("<<TASK_INSTRUCTION>>", task_instruction)
            .replace("<<STATIC_INVARIANTS>>", static_invariants)
            .replace("<<STEP_NUM>>", str(step_num))
            .replace("<<DYNAMIC_INVARIANTS_ALREADY_GENERATED>>", previous_assertions)
            .replace("<<TRAJECTORY_UP_TO_CURRENT_STEP>>", trajectory_till_current_step)
            .replace("<<AGENT_UNION>>", agent_union)
            .replace("<<TOOLS_LIST>>", json.dumps(tools_list, indent=2) if tools_list else "[]")
        )
    return prompt

# ------------------------------------------------------------------------------------
# IR STEP HELPERS
# ------------------------------------------------------------------------------------
def concat_step(step_obj: dict) -> str:
    """Concatenate all substep.content into one string for a given IR step."""
    subs = step_obj.get("substeps") or []
    parts: List[str] = []
    for ss in subs:
        c = ss.get("content")
        if isinstance(c, str) and c.strip():
            parts.append(c.strip())
    return "\n".join(parts).strip()

def extract_task_instruction(step_texts: List[str], fallback: str = "") -> str:
    """Heuristic: use fallback if provided, else first step text."""
    return (fallback or "").strip() or (step_texts[0] if step_texts else "")

def extract_domain_policy(step_texts: List[str]) -> str:
    """Heuristic: first step text is the domain policy."""
    return step_texts[0] if step_texts else ""

def format_steps_so_far(step_texts: List[str], steps: List[dict], upto: int) -> str:
    """
    Create the trajectory window shown to the model (up to step `upto`).
    Uses the IR step.index if present; otherwise falls back to 1-based display labels.
    """
    chunks: List[str] = []
    upto = min(upto, len(step_texts), len(steps))
    for i in range(upto):
        idx = steps[i].get("index")
        label = idx if isinstance(idx, int) else (i + 1)
        chunks.append(f"[STEP {label}]\n{step_texts[i]}")
    return "\n\n".join(chunks).strip()


def format_steps_full(step_texts: List[str], steps: List[dict]) -> str:
    """
    Create the FULL trajectory text for one-shot mode.
    Uses the IR step.index if present; otherwise falls back to 1-based display labels.
    """
    chunks: List[str] = []
    upto = min(len(step_texts), len(steps))
    for i in range(upto):
        idx = steps[i].get("index")
        label = idx if isinstance(idx, int) else (i + 1)
        chunks.append(f"[STEP {label}]\n{step_texts[i]}")
    return "\n\n".join(chunks).strip()

# ------------------------------------------------------------------------------------
# TELEMETRY SERIALIZATION (because datetime objects are not JSON serializable)
# ------------------------------------------------------------------------------------
def telemetry_to_jsonable(tel: Optional[LLMCallTelemetry]) -> Optional[dict]:
    if tel is None:
        return None
    out: Dict[str, Any] = {
        "llm_call_id": getattr(tel, "llm_call_id", None),
        "model_name": getattr(tel, "model_name", None),
        "instance": getattr(tel, "instance", None),
        "tokens": None,
        "time": None,
    }
    try:
        toks = getattr(tel, "tokens", None)
        if toks is not None:
            out["tokens"] = {
                "prompt_tokens": int(getattr(toks, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(toks, "output_tokens", 0) or 0),
                "total_tokens": int(getattr(toks, "total_tokens", 0) or 0),
            }
    except Exception:
        out["tokens"] = None

    try:
        tinfo = getattr(tel, "time", None)
        if tinfo is not None:
            st = getattr(tinfo, "start_time", None)
            et = getattr(tinfo, "end_time", None)
            out["time"] = {
                "start_time": st.isoformat(timespec="seconds")
                if isinstance(st, datetime.datetime)
                else str(st),
                "end_time": et.isoformat(timespec="seconds")
                if isinstance(et, datetime.datetime)
                else str(et),
                "execution_time_sec": float(getattr(tinfo, "execution_time_sec", 0.0) or 0.0),
            }
    except Exception:
        out["time"] = None
    return out

# ------------------------------------------------------------------------------------
# MAIN GENERATOR CLASS
# ------------------------------------------------------------------------------------
class DynamicInvariantGenerator:
    """
    Generates per-step dynamic invariants for each trajectory and writes:
      - one JSON file per trajectory (out_<traj_id>.json)
      - one optional text log per trajectory (log_<traj_id>.txt)
    """

    def __init__(
        self,
        *,
        out_dir: str = "dynamic_invariant_outputs",
        static_invariants_path: Optional[str] = None,
        domain: str = "flash",  # "flash" | "tau"
        model_name: Optional[str] = None,
        instance: Optional[str] = None,
        tools_list: Optional[List[str]] = None,
        tools_structure: Optional[Union[dict, str]] = None,
        include_nl_check: bool = True,
        endpoint: str = "trapi",
    ) -> None:
        self.out_dir = abspath_rel(out_dir)
        ensure_dir(self.out_dir)

        self.domain = domain
        self.instance = instance or getattr(g, "INSTANCE", None) or "default"
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.trajectory_total_tokens: int = 0
        self.tools_list: List[str] = [t.strip() for t in (tools_list or []) if (t or "").strip()]
        self.tools_structure = tools_structure
        self.include_nl_check = include_nl_check

        if endpoint == "copilot":
            from agentrx.llm_clients.copilot_cli import copilot_mk_client
            self.client = copilot_mk_client()
            self.model_name = model_name or "copilot-cli"
        elif endpoint == "azure":
            self.client = LLMAgentAzure.azure_mk_client()
            self.model_name = model_name or g.DEPLOYMENT
        elif endpoint == "openrouter":
            from agentrx.llm_clients.openrouter import LLMAgent as LLMAgentOpenRouter
            self.client = LLMAgentOpenRouter.openrouter_mk_client()
            self.model_name = model_name or os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
        else:
            self.client = LLMAgentTrapi.trapi_mk_client()
            self.model_name = model_name or g.TRAPI_DEPLOYMENT_NAME

        # Load static invariants once
        self.static_blob_raw = ""
        if static_invariants_path is None:
            static_invariants_path = getattr(g, "STATIC_INVARIANTS_OUTPUT_FILE_PATH", "") or ""
        if static_invariants_path:
            p = abspath_rel(static_invariants_path)
            if os.path.exists(p):
                self.static_blob_raw = open(p, "r", encoding="utf-8").read().strip()
                dbg(f"Loaded static invariants from {p} ({len(self.static_blob_raw)} chars)")
            else:
                dbg(f"Static invariants missing (continuing): {p}")

    def run_from_ir_data(self, data: list, source_label: str = "<ir>") -> None:
        """
        Generate dynamic invariants from already-converted IR trajectory data.
        Skips the load + domain-converter step (useful when the IR was produced
        by the LLM fallback and the domain converter would mangle it).
        """
        if not isinstance(data, list):
            print(f"[ERROR] Expected a list of trajectories, got {type(data)}", flush=True)
            return
        self._process_trajectories(data, source_label)

    def run_file(self, input_path: str) -> None:
        """
        Load trajectories, generate dynamic invariants per step, write one JSON per trajectory.
        """
        input_path_abs = abspath_rel(input_path)
        if not os.path.exists(input_path_abs):
            print(f"[ERROR] input file not found: {input_path_abs}", flush=True)
            return

        raw = load_trajectories(input_path_abs)
        data = {"tau": tau_bench_ir, "flash": flash_ir, "magentic": magentic_ir}.get(self.domain, flash_ir)(raw)
        if not isinstance(data, list):
            print(f"[ERROR] Expected a list of trajectories, got {type(data)}", flush=True)
            return

        self._process_trajectories(data, input_path)

    def _process_trajectories(self, data: list, source_label: str) -> None:
        """Shared processing loop for run_file and run_from_ir_data."""
        static_pretty = format_previous_static_assertions(self.static_blob_raw)

        dbg(f"Loaded trajectories={len(data)} domain={self.domain}")

        for ti, traj in enumerate(data):
            # Reset per-trajectory token counters (avoid leaking totals across trajectories)
            traj_id = str(traj.get("trajectory_id", f"idx_{ti}"))
            steps = traj.get("steps") or []
            if not steps:
                dbg(f"Skipping empty trajectory {traj_id}")
                continue

            step_texts = [concat_step(s) for s in steps]
            instruction = str(traj.get("instruction") or "").strip()
            task_instruction = extract_task_instruction(step_texts, instruction)
            domain_policy = extract_domain_policy(step_texts)
            per_step_outputs: List[Dict[str, Any]] = []
            dynamic_invariants: List[dict] = []

            for step_num in range(1, len(step_texts) + 1):
                traj_so_far = format_steps_so_far(step_texts, steps, step_num)
                prev_dyn_pretty = format_previous_dynamic_assertions(dynamic_invariants)

                try:
                    prompt = build_invariant_prompt(
                        traj,
                        step_num=step_num,
                        task_instruction=task_instruction,
                        domain_policy=domain_policy,
                        static_invariants=static_pretty,
                        previous_assertions=prev_dyn_pretty,
                        trajectory_till_current_step=traj_so_far,
                        tools_list=self.tools_list,
                        tools_structure=self.tools_structure,
                        include_nl_check=self.include_nl_check,
                    )
                except Exception as e:
                    print(f"[ERROR] Prompt format failed traj={traj_id} step={step_num}: {e}", flush=True)
                    traceback.print_exc()
                    return

                dbg(
                    f"traj={traj_id} step={step_num} "
                    f"prompt_chars={len(prompt)} so_far_chars={len(traj_so_far)} prev_invs={len(dynamic_invariants)}"
                )

                # --- Dump prompt to file for debugging ---
                _prompt_dump_path = os.path.join(self.out_dir, f"prompt_dump_{traj_id}_step{step_num}.txt")
                with open(_prompt_dump_path, "w", encoding="utf-8") as _pf:
                    _pf.write(prompt)
                print(f"[PROMPT DUMP] Written to {_prompt_dump_path} ({len(prompt)} chars)", flush=True)
                # --- End prompt dump ---

                dynamic_generation_start_timestamp = datetime.datetime.now()
                dynamic_generation_start = time.perf_counter()
                response = _llm_call_with_retry(
                    self.client, self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )

                dynamic_generation_end = time.perf_counter()
                dynamic_generation_end_timestamp = datetime.datetime.now()
                raw_text = (response.choices[0].message.content or "").strip()

                dynamic_generation_time_info = TimingInfo(
                start_time=dynamic_generation_start_timestamp.isoformat(timespec='seconds'),
                end_time=dynamic_generation_end_timestamp.isoformat(timespec='seconds'),
                execution_time_sec=round(dynamic_generation_end - dynamic_generation_start, 4)
                )
                # Parse JSON
                parsed_obj: Optional[dict] = None
                parse_error: Optional[str] = None
                try:
                    obj = json.loads(raw_text)
                    if isinstance(obj, dict):
                        parsed_obj = obj
                    elif isinstance(obj, list):
                        # Model returned a bare array — wrap it
                        parsed_obj = {"invariant": obj}
                    else:
                        parse_error = f"Model returned {type(obj).__name__}, expected dict or list"
                except Exception as e:
                    parse_error = str(e)


                # Track token usage metrics for dynamic invariant generation
                if hasattr(response, "usage") and response.usage is not None:
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                    total_tokens = response.usage.total_tokens
            
                    dynamic_invariant_token_usage = TokenUsage(
                        prompt_tokens=prompt_tokens,
                        output_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    )

                    self.total_prompt_tokens += prompt_tokens
                    self.total_completion_tokens += completion_tokens
                    self.trajectory_total_tokens += total_tokens

                else:
                    print("No usage info found in response.")

                dynamic_generation_llm_telemetry = LLMCallTelemetry(
                                                model_name=self.model_name,
                                                instance=self.instance,
                                                tokens=dynamic_invariant_token_usage,
                                                time=dynamic_generation_time_info,
                                           )

                record: Dict[str, Any] = {
                    "step_num": step_num,
                    "raw_model_text": raw_text,
                    "parsed": parsed_obj,
                    "parse_error": parse_error,
                    "telemetry": telemetry_to_jsonable(dynamic_generation_llm_telemetry),
                }
                # Accumulate invariant objects for future steps ("previous_assertions")
                if isinstance(parsed_obj, dict):
                    inv_list = (parsed_obj.get("invariant")
                                or parsed_obj.get("invariants")
                                or parsed_obj.get("dynamic_invariants") or [])
                    if isinstance(inv_list, list):
                        dynamic_invariants.extend([x for x in inv_list if isinstance(x, dict)])
                per_step_outputs.append(record)

            out_path = os.path.join(self.out_dir, f"out_{traj_id}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "trajectory_id": traj_id,
                        "domain": self.domain,
                        "input_path": source_label,
                        "task_instruction": task_instruction,
                        "static_invariants_used": static_pretty,
                        "per_step_outputs": per_step_outputs,
                        "dynamic_invariants": dynamic_invariants,
                        "totals": {
                            "prompt_tokens": int(self.total_prompt_tokens),
                            "completion_tokens": int(self.total_completion_tokens),
                            "total_tokens": int(self.trajectory_total_tokens),
                        },
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            print(
                f"[DONE] traj={traj_id} steps={len(per_step_outputs)} "
                f"invariants={len(dynamic_invariants)} -> {out_path}",
                flush=True,
            )



# ------------------------------------------------------------------------------------
# ONE-SHOT PROMPT BUILDER
# ------------------------------------------------------------------------------------
def build_one_shot_prompt(
    traj: Dict[str, Any],
    *,
    task_instruction: str,
    domain_policy: str,
    static_invariants: str,
    trajectory_full_text: str,
    tools_list: Optional[List[str]] = None,
    tools_structure: Optional[Union[dict, str]] = None,
    include_nl_check: bool = False,
) -> str:
    """
    Build prompt for ONE-SHOT dynamic invariant generation.
    Uses dedicated ONESHOT templates with full trajectory in a single call.
    Uses <<PLACEHOLDER>> substitution on mega-templates.
    """
    enums = extract_prompt_enums(traj)
    agent_union = enums["agent_union"]

    if include_nl_check:
        prompt = (
            DYNAMIC_INVARIANT_PROMPT_ONESHOT
            .replace("<<TASK_INSTRUCTION>>", task_instruction)
            .replace("<<STATIC_INVARIANTS>>", static_invariants)
            .replace("<<TRAJECTORY>>", trajectory_full_text)
            .replace("<<NL_CHECK_JUDGE_SYSTEM_PROMPT>>", NL_CHECK_JUDGE_SYSTEM_PROMPT.strip().replace('\n', '\\n'))
            .replace("<<RUBRIC_EVALUATION_ALGORITHM>>", RUBRIC_EVALUATION_ALGORITHM.strip().replace('\n', '\\n'))
            .replace("<<OUTPUT_FORMAT>>", OUTPUT_FORMAT.strip().replace('\n', '\\n'))
            .replace("<<AGENT_UNION>>", agent_union)
            .replace("<<TOOLS_LIST>>", json.dumps(tools_list, indent=2) if tools_list else "[]")
        )
    else:
        prompt = (
            DYNAMIC_INVARIANT_PROMPT_ONESHOT_WITHOUT_NL_CHECK
            .replace("<<TASK_INSTRUCTION>>", task_instruction)
            .replace("<<STATIC_INVARIANTS>>", static_invariants)
            .replace("<<TRAJECTORY>>", trajectory_full_text)
            .replace("<<AGENT_UNION>>", agent_union)
            .replace("<<TOOLS_LIST>>", json.dumps(tools_list, indent=2) if tools_list else "[]")
        )
    return prompt


# ------------------------------------------------------------------------------------
# ONE-SHOT GENERATOR CLASS
# ------------------------------------------------------------------------------------
class OneShotDynamicInvariantGenerator:
    """
    Generates dynamic invariants for each trajectory via a SINGLE LLM call
    (one-shot mode). Writes one JSON file per trajectory.
    """

    def __init__(
        self,
        *,
        out_dir: str = "dynamic_invariant_outputs",
        static_invariants_path: Optional[str] = None,
        domain: str = "flash",
        model_name: Optional[str] = None,
        instance: Optional[str] = None,
        tools_list: Optional[List[str]] = None,
        tools_structure: Optional[Union[dict, str]] = None,
        include_nl_check: bool = False,
        endpoint: str = "trapi",
    ) -> None:
        self.out_dir = abspath_rel(out_dir)
        ensure_dir(self.out_dir)

        self.domain = domain
        self.instance = instance or getattr(g, "INSTANCE", None) or "default"
        self.tools_list: List[str] = [t.strip() for t in (tools_list or []) if (t or "").strip()]
        self.tools_structure = tools_structure
        self.include_nl_check = include_nl_check

        if endpoint == "copilot":
            from agentrx.llm_clients.copilot_cli import copilot_mk_client
            self.client = copilot_mk_client()
            self.model_name = model_name or "copilot-cli"
        elif endpoint == "azure":
            self.client = LLMAgentAzure.azure_mk_client()
            self.model_name = model_name or g.DEPLOYMENT
        elif endpoint == "openrouter":
            from agentrx.llm_clients.openrouter import LLMAgent as LLMAgentOpenRouter
            self.client = LLMAgentOpenRouter.openrouter_mk_client()
            self.model_name = model_name or os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash")
        else:
            self.client = LLMAgentTrapi.trapi_mk_client()
            self.model_name = model_name or g.TRAPI_DEPLOYMENT_NAME

        # Load static invariants once
        self.static_blob_raw = ""
        if static_invariants_path is None:
            static_invariants_path = getattr(g, "STATIC_INVARIANTS_OUTPUT_FILE_PATH", "") or ""
        if static_invariants_path:
            p = abspath_rel(static_invariants_path)
            if os.path.exists(p):
                self.static_blob_raw = open(p, "r", encoding="utf-8").read().strip()
                dbg(f"Loaded static invariants from {p} ({len(self.static_blob_raw)} chars)")
            else:
                dbg(f"Static invariants missing (continuing): {p}")

    def run_file(self, input_path: str) -> None:
        input_path_abs = abspath_rel(input_path)
        if not os.path.exists(input_path_abs):
            print(f"[ERROR] input file not found: {input_path_abs}", flush=True)
            return

        raw = load_trajectories(input_path_abs)
        ir_fn = {"tau": tau_bench_ir, "flash": flash_ir, "magentic": magentic_ir}.get(self.domain, flash_ir)
        data = ir_fn(raw)
        if not isinstance(data, list):
            print(f"[ERROR] Expected a list of trajectories, got {type(data)}", flush=True)
            return

        static_pretty = format_previous_static_assertions(self.static_blob_raw)
        dbg(f"Loaded trajectories={len(data)} domain={self.domain}")

        for ti, traj in enumerate(data):
            traj_id = str(traj.get("trajectory_id", f"idx_{ti}"))
            steps = traj.get("steps") or []
            if not steps:
                dbg(f"Skipping empty trajectory {traj_id}")
                continue

            step_texts = [concat_step(s) for s in steps]
            instruction = str(traj.get("instruction") or "").strip()
            task_instruction = extract_task_instruction(step_texts, instruction)
            domain_policy = extract_domain_policy(step_texts)
            traj_full = format_steps_full(step_texts, steps)

            try:
                prompt = build_one_shot_prompt(
                    traj,
                    task_instruction=task_instruction,
                    domain_policy=domain_policy,
                    static_invariants=static_pretty,
                    trajectory_full_text=traj_full,
                    tools_list=self.tools_list,
                    tools_structure=self.tools_structure,
                    include_nl_check=self.include_nl_check,
                )
            except Exception as e:
                print(f"[ERROR] Prompt format failed traj={traj_id}: {e}", flush=True)
                traceback.print_exc()
                continue

            dbg(f"traj={traj_id} ONE_SHOT prompt_chars={len(prompt)} full_chars={len(traj_full)}")

            # --- Dump prompt to file for debugging ---
            _prompt_dump_path = os.path.join(self.out_dir, f"prompt_dump_{traj_id}_oneshot.txt")
            with open(_prompt_dump_path, "w", encoding="utf-8") as _pf:
                _pf.write(prompt)
            print(f"[PROMPT DUMP] Written to {_prompt_dump_path} ({len(prompt)} chars)", flush=True)
            # --- End prompt dump ---

            start_ts = datetime.datetime.now()
            start = time.perf_counter()

            response = _llm_call_with_retry(
                self.client, self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )

            end = time.perf_counter()
            end_ts = datetime.datetime.now()
            raw_text = (response.choices[0].message.content or "").strip()

            timing = TimingInfo(
                start_time=start_ts.isoformat(timespec="seconds"),
                end_time=end_ts.isoformat(timespec="seconds"),
                execution_time_sec=round(end - start, 4),
            )

            parsed_obj: Optional[dict] = None
            parse_error: Optional[str] = None
            try:
                obj = json.loads(raw_text)
                if isinstance(obj, dict):
                    parsed_obj = obj
                elif isinstance(obj, list):
                    # Model returned a bare array — wrap it
                    parsed_obj = {"invariant": obj}
                else:
                    parse_error = f"Model returned {type(obj).__name__}, expected dict or list"
            except Exception as e:
                parse_error = str(e)

            token_usage = TokenUsage(prompt_tokens=0, output_tokens=0, total_tokens=0)
            if hasattr(response, "usage") and response.usage is not None:
                token_usage = TokenUsage(
                    prompt_tokens=int(response.usage.prompt_tokens or 0),
                    output_tokens=int(response.usage.completion_tokens or 0),
                    total_tokens=int(response.usage.total_tokens or 0),
                )

            tel = LLMCallTelemetry(
                model_name=self.model_name,
                instance=self.instance,
                tokens=token_usage,
                time=timing,
            )

            dynamic_invariants: List[dict] = []
            if isinstance(parsed_obj, dict):
                inv_list = (parsed_obj.get("invariant")
                            or parsed_obj.get("invariants")
                            or parsed_obj.get("dynamic_invariants") or [])
                if isinstance(inv_list, list):
                    dynamic_invariants = [x for x in inv_list if isinstance(x, dict)]

            record: Dict[str, Any] = {
                "step_num": -1,
                "raw_model_text": raw_text,
                "parsed": parsed_obj,
                "parse_error": parse_error,
                "telemetry": telemetry_to_jsonable(tel),
            }

            out_path = os.path.join(self.out_dir, f"out_{traj_id}_oneshot.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "trajectory_id": traj_id,
                        "domain": self.domain,
                        "input_path": input_path,
                        "task_instruction": task_instruction,
                        "static_invariants_used": static_pretty,
                        "one_shot_output": record,
                        "dynamic_invariants": dynamic_invariants,
                        "totals": {
                            "prompt_tokens": int(token_usage.prompt_tokens),
                            "completion_tokens": int(token_usage.output_tokens),
                            "total_tokens": int(token_usage.total_tokens),
                        },
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

            print(
                f"[DONE] traj={traj_id} one_shot invariants={len(dynamic_invariants)} -> {out_path}",
                flush=True,
            )


# ------------------------------------------------------------------------------------
# DOMAIN HELPERS
# ------------------------------------------------------------------------------------
def get_domain_tools(domain: str) -> Tuple[List[str], Optional[dict]]:
    """Return (tools_list, tools_structure) for the given domain."""
    tools_list = (
        TAU_RETAIL_TOOLS_LIST if domain == "tau"
        else FLASH_TOOLS_LIST if domain == "flash"
        else MAGENTIC_ONE_TOOLS_LIST if domain == "magentic"
        else []
    )
    tools_structure = (
        TAU_RETAIL_TOOLS_STRUCTURE if domain == "tau"
        else FLASH_TOOLS_STRUCTURE if domain == "flash"
        else MAGENTIC_ONE_TOOLS_STRUCTURE if domain == "magentic"
        else None
    )
    return tools_list, tools_structure


# ------------------------------------------------------------------------------------
# TAU-BENCH SPECIFIC DATA STRUCTURES
# ------------------------------------------------------------------------------------
TAU_RETAIL_TOOLS_LIST = [
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "list_all_product_types",
    "get_order_details",
    "get_product_details",
    "get_user_details",
    "cancel_pending_order",
    "exchange_delivered_order_items",
    "modify_pending_order_address",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_user_address",
    "return_delivered_order_items",
]

# Tau Retail tools structure
TAU_RETAIL_TOOLS_STRUCTURE = {
    "protocol": {
        "event_types": ["assistant", "tool"],
        "common_fields": ["role", "content", "tool_calls", "tool_call_id", "name", "index"],
        "call_pattern": "Assistant makes tool_calls, system returns tool results"
    },
    "tools": {
        "find_user_id_by_email": {
            "capability": "Locate user ID by email address for authentication",
            "inputs": {"email": "string"},
            "outputs": {"user_id": "string"}
        },
        "find_user_id_by_name_zip": {
            "capability": "Locate user ID by first name, last name, and zip code for authentication",
            "inputs": {"first_name": "string", "last_name": "string", "zip": "string"},
            "outputs": {"user_id": "string"}
        },
        "list_all_product_types": {
            "capability": "List all available product types and their IDs",
            "inputs": {},
            "outputs": {"products": "dict mapping product names to IDs"}
        },
        "get_order_details": {
            "capability": "Retrieve detailed information about a specific order",
            "inputs": {"order_id": "string"},
            "outputs": {"order_info": "dict with status, items, payment, address, dates"}
        },
        "get_product_details": {
            "capability": "Get details about a specific product including available items",
            "inputs": {"product_id": "string"},
            "outputs": {"product_info": "dict with items, options, prices"}
        },
        "get_user_details": {
            "capability": "Retrieve user profile information",
            "inputs": {"user_id": "string"},
            "outputs": {"user_info": "dict with email, address, payment methods"}
        },
        "cancel_pending_order": {
            "capability": "Cancel an order that is in pending status",
            "inputs": {"order_id": "string", "reason": "string ('no longer needed' or 'ordered by mistake')"},
            "outputs": {"confirmation": "dict with cancellation details"}
        },
        "exchange_delivered_order_items": {
            "capability": "Exchange items from a delivered order (can only be called once per order)",
            "inputs": {
                "order_id": "string",
                "item_ids": "list of strings",
                "new_item_ids": "list of strings",
                "reason": "string"
            },
            "outputs": {"confirmation": "dict with exchange details"}
        },
        "modify_pending_order_address": {
            "capability": "Modify delivery address for a pending order",
            "inputs": {"order_id": "string", "new_address": "string"},
            "outputs": {"confirmation": "dict with updated address"}
        },
        "modify_pending_order_items": {
            "capability": "Modify items in a pending order (can only be called once per order)",
            "inputs": {
                "order_id": "string",
                "item_ids_to_remove": "list of strings",
                "item_ids_to_add": "list of strings"
            },
            "outputs": {"confirmation": "dict with updated items"}
        },
        "modify_pending_order_payment": {
            "capability": "Modify payment method for a pending order",
            "inputs": {"order_id": "string", "new_payment_method": "string"},
            "outputs": {"confirmation": "dict with updated payment"}
        },
        "modify_user_address": {
            "capability": "Update user's default address",
            "inputs": {"user_id": "string", "new_address": "string"},
            "outputs": {"confirmation": "dict with updated address"}
        },
        "return_delivered_order_items": {
            "capability": "Return items from a delivered order",
            "inputs": {
                "order_id": "string",
                "item_ids": "list of strings",
                "reason": "string"
            },
            "outputs": {"confirmation": "dict with return details"}
        }
    }
}

FLASH_TOOLS_LIST = [
    "Orchestrator",
    "KustoAgent",
    "IncidentAgent",
    "Executor",
    "GeneralAssistant",
    "Coder"
]

FLASH_TOOLS_STRUCTURE = {
    "protocol": {
        "event_types": ["OrchestrationEvent", "LLMCallEvent"],
        "common_fields": ["timestamp", "source", "message", "type"],
        "call_pattern": "Orchestrator delegates via source like: 'Orchestrator (-> <AgentName>)'"
    },
    "agents": {
        "KustoAgent": {
            "capability": "Execute a provided Kusto query (must already exist in the plan).",
            "inputs": {"kusto_query": "string"},
            "outputs": {"result": "pandas dataframe"}
        },
        "IncidentAgent": {
            "capability": "Query/return incident descriptions only (no extra troubleshooting).",
            "inputs": {"incident_id": "string"},
            "outputs": {"incident_description": "string"}
        },
        "Executor": {
            "capability": "Run user-provided code blocks only (python/sh).",
            "inputs": {"python": "string", "sh": "string"},
            "outputs": {"stdout": "string", "stderr": "string"}
        },
        "GeneralAssistant": {
            "capability": "Summarize, explain, and draft responses based on evidence in the trace.",
            "inputs": {"message": "string"},
            "outputs": {"message": "string"}
        },
        "Orchestrator": {
            "capability": "Follow the plan steps; emit ledger JSON with required keys; pick next speaker.",
            "ledger_schema": [
                "is_step_finished{reason,answer}",
                "next_step{reason,answer}",
                "is_in_loop{reason,answer}",
                "is_progress_being_made{reason,answer}",
                "next_speaker{reason,answer}",
                "instruction_or_question{reason,answer}",
            ]
        }
    }
}

MAGENTIC_ONE_TOOLS_LIST = [
    "Orchestrator",
    "WebSurfer",
    "ComputerTerminal",
    "FileSurfer",
    "Assistant",
]

MAGENTIC_ONE_TOOLS_STRUCTURE = {
    "protocol": {
        "event_types": [
            "human",
            "Assistant",
            "Orchestrator",
            "WebSurfer",
            "ComputerTerminal",
            "FileSurfer",
        ],
        "common_fields": ["role", "content"],
        "call_pattern": (
            "Events are a flat list of {role, content}. Orchestrator coordinates the run.\n"
            "- Planning/ledger typically appears as role like 'Orchestrator (thought)'.\n"
            "- Delegation appears as role like 'Orchestrator (-> WebSurfer)'.\n"
            "- Agent responses appear as role exactly 'WebSurfer', 'ComputerTerminal', etc.\n"
            "- Termination can appear as 'Orchestrator (termination condition)'."
        ),
        "notes": [
            "The ledger is often embedded as JSON text inside content.",
            "WebSurfer outputs may include 'I typed ...', 'I clicked ...', and 'Here is a screenshot of ...'.",
        ],
    },
    "agents": {
        "Orchestrator": {
            "capability": "Plans, emits ledger JSON, delegates tasks to other agents, and decides termination.",
            "inputs": {"request": "string", "history": "list of {role, content}"},
            "outputs": {"ledger_json": "string (JSON blob embedded in content)", "delegation": "string"},
            "role_patterns": [
                "Orchestrator (thought)",
                "Orchestrator (-> <AgentName>)",
                "Orchestrator (termination condition)",
            ],
        },
        "WebSurfer": {
            "capability": "Performs web search/browsing steps and reports evidence (snippets/screenshots/URLs).",
            "inputs": {"query_or_instruction": "string"},
            "outputs": {
                "narration": "string",
                "evidence": "string (may include screenshot descriptions, extracted text/OCR)",
            },
            "evidence_markers": [
                "I typed",
                "I clicked",
                "Here is a screenshot of",
                "The viewport shows",
                "Automatic OCR",
                "Search query:",
            ],
        },
        "ComputerTerminal": {
            "capability": "Runs user-provided code blocks only (python/sh).",
            "inputs": {"python": "string", "sh": "string"},
            "outputs": {"stdout": "string", "stderr": "string"},
        },
        "FileSurfer": {
            "capability": "Reads/writes local files and summarizes file content.",
            "inputs": {"path": "string", "operation": "read|list|write"},
            "outputs": {"content": "string", "metadata": "dict"},
        },
        "Assistant": {
            "capability": "General reasoning/summarization and final answer drafting based on evidence in the trace.",
            "inputs": {"message": "string"},
            "outputs": {"message": "string"},
        },
    },
}

# ------------------------------------------------------------------------------------
# ENTRYPOINT
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    # Usage:
    #   python dynamic_invariant_generator.py --domain flash --mode stepbystep --input-path flash_dataset/
    #   python dynamic_invariant_generator.py --domain tau --mode oneshot --input-path tau.json
    parser = argparse.ArgumentParser(description="Dynamic invariant generator (step-by-step or one-shot)")
    parser.add_argument("--domain", type=str, default="flash",
                        choices=["flash", "tau", "magentic"],
                        help="Domain to run (default: flash)")
    parser.add_argument("--mode", type=str, default="stepbystep",
                        choices=["stepbystep", "oneshot"],
                        help="Generation mode: stepbystep (per-step LLM calls) or oneshot (single LLM call per trajectory)")
    parser.add_argument("--input-path", type=str, default=None,
                        help="Path to input trajectory file or directory")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory for dynamic invariants")
    parser.add_argument("--static-invariants", type=str, default=None,
                        help="Path to static invariants JSON file")
    parser.add_argument("--endpoint", type=str, default="trapi",
                        choices=["azure", "trapi"],
                        help="LLM endpoint to use (default: trapi)")
    grp = parser.add_mutually_exclusive_group()
    parser.set_defaults(include_nl_check=True)
    grp.add_argument("--include-nl-check", dest="include_nl_check", action="store_true")
    grp.add_argument("--no-nl-check", dest="include_nl_check", action="store_false")
    args = parser.parse_args()

    domain = args.domain
    input_path = args.input_path or "flash_dataset/"
    out_dir = args.out_dir or os.getenv("OUT_DIR", "dynamic_invariant_outputs")
    static_invariants_path = args.static_invariants or os.path.join("out", f"static_invariants_{domain}_nlcheck.json")

    tools_list, tools_structure = get_domain_tools(domain)

    print(f"[CONFIG] domain={domain} mode={args.mode} input={input_path} nl_check={args.include_nl_check}")

    # Select generator class based on mode
    if args.mode == "oneshot":
        gen = OneShotDynamicInvariantGenerator(
            out_dir=out_dir,
            static_invariants_path=static_invariants_path,
            domain=domain,
            tools_list=tools_list,
            tools_structure=tools_structure,
            include_nl_check=args.include_nl_check,
            endpoint=args.endpoint,
        )
    else:
        gen = DynamicInvariantGenerator(
            out_dir=out_dir,
            static_invariants_path=static_invariants_path,
            domain=domain,
            tools_list=tools_list,
            tools_structure=tools_structure,
            include_nl_check=args.include_nl_check,
            endpoint=args.endpoint,
        )

    # If input_path is a directory, run on every .json/.jsonl file within it.
    input_path_abs = abspath_rel(input_path)
    if os.path.isdir(input_path_abs):
        files = []
        for root, _, fns in os.walk(input_path_abs):
            for fn in fns:
                if fn.lower().endswith(".json") or fn.lower().endswith(".jsonl"):
                    files.append(os.path.join(root, fn))
        files.sort()
        print(f"[INFO] Found {len(files)} file(s) under {input_path_abs}", flush=True)
        for fp in files:
            gen.run_file(fp)
    else:
        gen.run_file(input_path)