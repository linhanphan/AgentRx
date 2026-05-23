import os
from dotenv import load_dotenv

load_dotenv()

CLIENT_ID = os.environ.get("AGENT_VERIFY_CLIENT_ID", "")# Managed Identity client ID if applicable
ENDPOINT = os.environ.get("AGENT_VERIFY_ENDPOINT", "")
API_VERSION = os.environ.get("AGENT_VERIFY_API_VERSION", "2024-12-01-preview")
DEPLOYMENT = os.environ.get("AGENT_VERIFY_DEPLOYMENT", "gpt-5-mini")
MODEL_NAME = os.environ.get("AGENT_VERIFY_MODEL_NAME", "gpt-5-mini")
EMBEDDING_MODEL_NAME = os.environ.get("AGENT_VERIFY_EMBEDDING_MODEL_NAME", "text-embedding-3-small")
INSTANCE = os.environ.get("AGENT_VERIFY_INSTANCE", "default")
RUN_DIR = None

MAGENTIC_TASK_IDS = [
    "5f982798-16b9-4051-ab57-cfc7ebdb2a91",
    "c7afe00869f98cf363fd83677ac41757ed5e57f03eacc3d1304feb0a92084bd1",
    "ebbc1f13-d24d-40df-9068-adcf735b4240",
    "52f7224e9c79431e7926afe317782711a0028750693e7456cde22ef6f4bd8bd5",
    "3af8028c2a59e28ca88baff0e6d91f2a9f170c5ef91003f1c8406755a2760ad4",
    "e6bc98089608217e45b6956a46518fe3cce64a799b3ac43c6974c449ae14c408",
    "a1e91b78-d3d8-4675-bb8d-62741b4b68a6",
    "08cae58d-4084-4616-b6dd-dd6534e4825b",
    "ccec2229ced20a4b0cb4897e3a99120a3017ea030903e01c9bda6b13d40b0b14",
    "6b06d186921b8b390c65aebd0d16f09f60a47d2f1288ebe36953f734e84c0a3c",
    "8b3379c0-0981-4f5b-8407-6444610cb212",
    "840bfca7-4f7b-481a-8794-c560c340185d",
    "8ad84bd6fe38481ba49e7ad1f6fbd43219a999074e5c6fc940003281f55ec65b",
    "1f975693-876d-457b-a649-393859e79bf3",
    "114d5fd0-e2ae-4b6d-a65a-870da2d19c08",
    "2ddae3b7a208e3c25f14d82d7a1faaaa1832fbf950b4dac345e755c4c361f294",
    "0ec4371851b96837b0a81b3dd3df401415061bb532fbafeb4609f3337c358508",
    "5d0080cb-90d7-4712-bc33-848150e917d3",
    "6e3be83d1949fa52cba03fb1ce4b5b3bf7e37a83fd7d67694b10b2e439d90cf8",
    "797f7a5b65ca28b7e7156e7db1e9f117bd4a021de0cd512bfdbb0be897d89eab",
    "624cbf11-6a41-4692-af9c-36b3e5ca3130",
    "16d825ff-1623-4176-a5b5-42e0f5c2b0ac",
    "55f4258484c5b398956133128a50462a767da211f8f72aa5ac5bbffb9bcbba1a",
    "557e78eceec08ca8b0da5f9fdaca6e1c7ec6140a8ce600983ee716327dab005e",
    "72c06643-a2fa-4186-aa5c-9ec33ae9b445",
    "2dfc4c37-fec1-4518-84a7-10095d30ad75",
    "a3fbeb63-0e8c-4a11-bff6-0e3b484c3e9c",
    "929b45f34805280d77c61d1e093e3d4e551d77ddb6ecd73552b12b1af286388d",
    "42576abe-0deb-4869-8c63-225c2d75a95a",
    "14569e28-c88c-43e4-8c32-097d35b9a67d",
    "2aa5dd83fbcd0dce9a3dd4592106e5b5edf738008d932e357d477bba80e59ccf",
    "42d4198c-5895-4f0a-b0c0-424a66465d83",
    "72e110e7-464c-453c-a309-90a95aed6538",
    "748899d9d70c09beb3bd48ac8a3658bdcfd2f9114fe6dc4c4b8d2f9541ef4607",
    "9baaa267c95f9d8b75741ee9169c50563d297cfa592c20deaffd30dbc5984c74",
    "b36ef2d8f2643b80e74a44ce3403f674ecb2aed7fd36afeaa289061a59feef92",
    "73c1b9fe-ee1d-4cf4-96ca-35c08f97b054",
    "0a65cb96-cb6e-4a6a-8aae-c1084f613456",
    "56137764-b4e0-45b8-9c52-1866420c3df5",
    "9e31099fffa6a3891c94934fd4fc2f3f522d51c1904ff3561f3a10e4bf245821",
    "b816bfce-3d80-4913-a07d-69b752ce6377",
    "57d9dc6935e8a40b02e7f8ec81768fe70e68a0c05f6866927c9fda38db38a486",
    "a9074997e698f912b9e751779ea19c1e92fa148404e90e0ae997acea3f9559b0",
    "f88066d274e265edd6cd9d61cd80a41accb3a14baf2297652fdd05cdf716d455",
]

DEFAULT_ENDPOINT = os.environ.get("AGENT_VERIFY_ENDPOINT_TYPE", "copilot")  # "copilot", "azure", "trapi", or "openrouter"
if DEFAULT_ENDPOINT not in ("copilot", "azure", "trapi", "openrouter"):
    raise ValueError(f"AGENT_VERIFY_ENDPOINT_TYPE must be 'copilot', 'azure', 'trapi', or 'openrouter', got: {DEFAULT_ENDPOINT!r}")

TRAPI_INSTANCE = os.environ.get("AGENT_VERIFY_TRAPI_INSTANCE", "")  # See https://aka.ms/trapi/models for the instance name
TRAPI_ENDPOINT_PREFIX = os.environ.get("AGENT_VERIFY_TRAPI_ENDPOINT_PREFIX", "https://trapi.research.microsoft.com/")
TRAPI_API_VERSION = os.environ.get("AGENT_VERIFY_TRAPI_API_VERSION", "2025-03-01-preview")  # See: https://learn.microsoft.com/en-us/azure/ai-services/openai/api-version-deprecation#latest-ga-api-release
TRAPI_MODEL_NAME = os.environ.get("AGENT_VERIFY_TRAPI_MODEL_NAME", "gpt-5")
TRAPI_MODEL_VERSION = os.environ.get("AGENT_VERIFY_TRAPI_MODEL_VERSION", "2025-04-16")
TRAPI_DEPLOYMENT_NAME = os.environ.get("AGENT_VERIFY_TRAPI_DEPLOYMENT_NAME", "")  # See: https://aka.ms/trapi/models

TOOLS_LIST = [
              'find_user_id_by_email',
              'find_user_id_by_name_zip',
              'list_all_product_types',
              'get_order_details',
              'get_product_details',
              'get_user_details',
              'cancel_pending_order',
              'exchange_delivered_order_items',
              'modify_pending_order_address',
              'modify_pending_order_items',
              'modify_pending_order_payment',
              'modify_user_address',
              'return_delivered_order_items'
             ]

FILTERED_TASKS_MAGENTIC = [
    "0ec4371851b96837b0a81b3dd3df401415061bb532fbafeb4609f3337c358508",
    "114d5fd0-e2ae-4b6d-a65a-870da2d19c08",
    "14569e28-c88c-43e4-8c32-097d35b9a67d",
    "291b53e665b4dd4365cde995042db4a6f6fecef3fe3a6f4482f23d61bd673918",
    "2aa5dd83fbcd0dce9a3dd4592106e5b5edf738008d932e357d477bba80e59ccf",
    "2dfc4c37-fec1-4518-84a7-10095d30ad75",
    "3af8028c2a59e28ca88baff0e6d91f2a9f170c5ef91003f1c8406755a2760ad4",
    "42576abe-0deb-4869-8c63-225c2d75a95a",
    "42d4198c-5895-4f0a-b0c0-424a66465d83",
    "4dbedc5e1a0205e14b7ff3ba89bce3060dab15d0ada3b7e1351a6f2aa8287aec",
    "52f7224e9c79431e7926afe317782711a0028750693e7456cde22ef6f4bd8bd5",
    "557e78eceec08ca8b0da5f9fdaca6e1c7ec6140a8ce600983ee716327dab005e",
    "56137764-b4e0-45b8-9c52-1866420c3df5",
    "57d9dc6935e8a40b02e7f8ec81768fe70e68a0c05f6866927c9fda38db38a486",
    "6b06d186921b8b390c65aebd0d16f09f60a47d2f1288ebe36953f734e84c0a3c",
    "6e3be83d1949fa52cba03fb1ce4b5b3bf7e37a83fd7d67694b10b2e439d90cf8",
    "72c06643-a2fa-4186-aa5c-9ec33ae9b445",
    "72e110e7-464c-453c-a309-90a95aed6538",
    "73c1b9fe-ee1d-4cf4-96ca-35c08f97b054",
    "7673d772-ef80-4f0f-a602-1bf4485c9b43",
    "840bfca7-4f7b-481a-8794-c560c340185d",
    "8ad84bd6fe38481ba49e7ad1f6fbd43219a999074e5c6fc940003281f55ec65b",
    "8d46b8d6-b38a-47ff-ac74-cda14cf2d19b",
    "929b45f34805280d77c61d1e093e3d4e551d77ddb6ecd73552b12b1af286388d",
    "a0c07678-e491-4bbc-8f0b-07405144218f",
    "a1e91b78-d3d8-4675-bb8d-62741b4b68a6",
    "a3fbeb63-0e8c-4a11-bff6-0e3b484c3e9c",
    "b36ef2d8f2643b80e74a44ce3403f674ecb2aed7fd36afeaa289061a59feef92",
    "b816bfce-3d80-4913-a07d-69b752ce6377",
    "cca4776df3c73e7f9430a2e624aafad056b14322a0b7ca6c0c22b7e7f3f0890a",
    "ccec2229ced20a4b0cb4897e3a99120a3017ea030903e01c9bda6b13d40b0b14",
    "db4fd70a-2d37-40ea-873f-9433dc5e301f",
    "e6bc98089608217e45b6956a46518fe3cce64a799b3ac43c6974c449ae14c408",
    "ed58682d-bc52-4baa-9eb0-4eb81e1edacc",
    "f88066d274e265edd6cd9d61cd80a41accb3a14baf2297652fdd05cdf716d455",
]

DOCUMENT_POLICY_PATH = 'policy_documents/retail_policy.txt'

TAU_RETAIL_TRAJECTORY_PATH = 'agent_trajectory_tau_retail.json'
TAU_RETAIL_POLICY_PATH = 'policy_documents/tau_retail_policy.txt'

TOOLS_FORMAT_PATH = 'tools_structure.json'

# Parent directory for all violation results - will be timestamped at runtime
VIOLATION_RESULTS_BASE_DIR = None  # Will be set dynamically in main.py

# Subdirectory names (relative paths within the timestamped parent directory)
INVARIANT_OUTPUTS_SUBDIR = 'invariant_outputs'
INVARIANTS_MODULE_SUBDIR = 'invariants_module'
METRICS_OUTPUT_SUBDIR = 'metrics_output'
JUDGE_CONTEXT_SUBDIR = 'judge_context'
DEDUPLICATED_VIOLATIONS_SUBDIR = 'deduplicated_violations'

# Full paths - will be set dynamically based on VIOLATION_RESULTS_BASE_DIR
STATIC_INVARIANTS_OUTPUT_FILE_PATH = 'invariant_outputs/static_invariants_output.txt'
STATIC_INVARIANTS_MODULE_FILE_PATH = 'invariants_module/static_invariants_module.py'
DYNAMIC_INVARIANTS_OUTPUT_PATH = 'invariant_outputs/dynamic_invariants_output'
DYNAMIC_INVARIANTS_MODULE_PATH = 'invariants_module/dynamic_invariants_module'
METRICS_OUTPUT_DIR_PATH = 'metrics_output'
JUDGE_CONTEXT_DIR_PATH = 'judge_context'
DEDUPLICATED_VIOLATIONS_DIR_PATH = 'deduplicated_violations'

COPILOT_ENDPOINT = "copilot"
AZURE_ENDPOINT = "azure"
TRAPI_ENDPOINT = "trapi"

def initialize_results_directory(timestamp=None):
    """
    Initialize a timestamped parent directory for all violation results.
    Updates all global path variables to use subdirectories within the timestamped parent.
    
    Args:
        timestamp (str, optional): Custom timestamp string. If None, generates current timestamp.
    
    Returns:
        str: The created parent directory path
    """
    global VIOLATION_RESULTS_BASE_DIR
    global STATIC_INVARIANTS_OUTPUT_FILE_PATH, STATIC_INVARIANTS_MODULE_FILE_PATH
    global DYNAMIC_INVARIANTS_OUTPUT_PATH, DYNAMIC_INVARIANTS_MODULE_PATH
    global METRICS_OUTPUT_DIR_PATH, JUDGE_CONTEXT_DIR_PATH, DEDUPLICATED_VIOLATIONS_DIR_PATH
    
    import os
    from datetime import datetime as dt
    
    if timestamp is None:
        timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
    
    # Set the parent directory with timestamp
    VIOLATION_RESULTS_BASE_DIR = f'violation_results_{timestamp}'
    
    # Update all paths to be relative to the parent directory
    STATIC_INVARIANTS_OUTPUT_FILE_PATH = os.path.join(VIOLATION_RESULTS_BASE_DIR, INVARIANT_OUTPUTS_SUBDIR, 'static_invariants_output.txt')
    STATIC_INVARIANTS_MODULE_FILE_PATH = os.path.join(VIOLATION_RESULTS_BASE_DIR, INVARIANTS_MODULE_SUBDIR, 'static_invariants_module.py')
    DYNAMIC_INVARIANTS_OUTPUT_PATH = os.path.join(VIOLATION_RESULTS_BASE_DIR, INVARIANT_OUTPUTS_SUBDIR, 'dynamic_invariants_output')
    DYNAMIC_INVARIANTS_MODULE_PATH = os.path.join(VIOLATION_RESULTS_BASE_DIR, INVARIANTS_MODULE_SUBDIR, 'dynamic_invariants_module')
    METRICS_OUTPUT_DIR_PATH = os.path.join(VIOLATION_RESULTS_BASE_DIR, METRICS_OUTPUT_SUBDIR)
    JUDGE_CONTEXT_DIR_PATH = os.path.join(VIOLATION_RESULTS_BASE_DIR, JUDGE_CONTEXT_SUBDIR)
    DEDUPLICATED_VIOLATIONS_DIR_PATH = os.path.join(VIOLATION_RESULTS_BASE_DIR, DEDUPLICATED_VIOLATIONS_SUBDIR)
    
    return VIOLATION_RESULTS_BASE_DIR

STATIC_INVARIANT_PROMPT = """You are an expert at analyzing domain policies to generate meaningful static assertions for agent trajectory verification. Your task is to identify important rules from policy documents that prevent policy violations, errors, and ensure proper operation.
You must extract checkable rules from policy documents that can be programmatically verified when specific tool calls are made during an agent's execution.

## Your Task:
Analyze the provided policy document and generate a focused set of static assertions that:
1. Are simple enough to implement programmatically and are objectively verifiable from trajectory data
2. Can be checked at specific tool call triggers
3. Focus on critical policy violations that must be prevented
4. Include both critical business logic AND important error checking
5. Avoid trivial input checks, like assertions that check field completeness
6. Provide working Python code for each assertion that can be executed

## Focus Areas:
Generate assertions for these key constraint types:

## Focus Areas - Generate assertions for:
### **Precondition Checks**
- Required state/status before tool execution
- Examples:
  - Retail: "Order status must be 'pending' before cancellation"
  - Healthcare: "Patient must be 'checked-in' before treatment actions"
  - Financial: "Account must be 'active' before transaction processing"
  - System Admin: "Service must be 'stopped' before configuration changes"

### **Sequential Dependencies** 
- Required action ordering between tool calls
- Examples:
  - General: "User authentication must precede any data access operations"
  - Retail: "User must give explicit confirmation (yes) before any consequential database write actions"

### **Business Rule Constraints**
- Domain-specific operational limits and rules
- Examples:
  - Retail: "Items can only be exchanged within same product category"
  - Healthcare: "Prescriptions require valid medical license verification"
  - Financial: "Transfer amounts cannot exceed daily limits"
  - Legal: "Document modifications require authorized personnel only"

### **Single-Use Restrictions**
- Tools that can only be called once per entity/session
- Examples:
  - Retail: "Order modification tool can only be used once per order"
  - Healthcare: "Critical procedure authorization can only be granted once"
  - Financial: "Account closure can only be initiated once per request"
  - System: "Database migration can only be executed once per maintenance window"

## Output Format:
For each assertion, provide:
1. **Assertion Name**: Simple descriptive name (e.g., check_order_status_for_cancellation)
2. **Tool Trigger**: Specific tool call(s) that should trigger this check
3. **Check Logic**: Clear, specific condition to verify
4. **Violation Condition**: When this assertion fails
5. **Python Code**: Complete Python function that implements this assertion check

## Example Output:
```
Assertion Name: check_order_status_for_cancellation
Tool Trigger: cancel_pending_order
Check Logic: Verify order.status == 'pending' before allowing cancellation
Violation Condition: Attempting to cancel order when status is not 'pending'
Python Code: 
    def check_order_status_for_cancellation(trajectory_step, tool_call_params, tool_response):
        # Check if order status is 'pending' before cancellation
        if tool_response and 'order' in tool_response:
            order_status = tool_response['order'].get('status')
            return order_status == 'pending'
        return False

Assertion Name: check_user_authentication_before_data_access
Tool Trigger: get_user_profile, get_order_details, modify_user_data
Check Logic: Verify user authentication step completed earlier in trajectory
Violation Condition: Accessing user data without prior authentication
Python Code:
    def check_user_authentication_before_data_access(trajectory, current_step_index):
        # Check if user authentication occurred before data access
        for i in range(current_step_index):
            step = trajectory[i]
            if step.get('tool_name') in ['find_user_id_by_email', 'find_user_id_by_name_zip']:
                return True
        return False

## Quality Guidelines:
- **Policy-Derived**: Must be supported by the policy document content, include any user confirmation checks before critical actions if specified in policy
- **Practical Impact**: Focus on assertions that prevent real operational problems
- **Domain-Relevant**: Prioritize rules specific to this business domain
- **Reasonable Scope**: Avoid both trivial sanity checks and computation checks better suited for dynamic invariants

## Instructions:
1. Read the policy document thoroughly
2. Identify important constraints, rules, and requirements
3. Include critical business logic, essential error checking, and policy compliance
4. Focus on assertions that ensure proper operation and prevent violations
5. Generate a comprehensive but focused set of static assertions
6. AVOID assertions that check:
   - Whether all required fields are filled/provided
   - Whether information was provided earlier

Each assertion generated should be mapped to a tool call.
Each assertion should be as GENERAL as possible, if many specific assertions can be combined into one assertion function, do so.
For example, an assertion which checks order status validity for various tool calls can be combined into one function.

## Python Code Guidelines:
- Write complete, executable Python functions
- You will be given 2 parameters - (trajectory and current_step_index), you can extract all the info required such as arguments for the tool call from the current trajectory step.
- Include docstrings explaining the assertion
- Handle edge cases and missing data gracefully - verify field existence before access
- **Exception Handling**: Raise exceptions for safety check violations (KeyError, IndexError, TypeError, AttributeError, etc.) instead of returning False. Only return boolean (True/False) for actual invariant logic violations
- Return boolean values (True if assertion passes, False if invariant violation detected)
- **Add print statements for debugging**: Start with printing the function name, then print all key variables and extracted JSON fields to aid in debugging and verification

The tools available to the LLM are:

{tools_list}

**IMPORTANT: Review this JSON structure carefully before writing any code. Use ONLY the field names that appear in this structure:**

{tools_structure}

Please analyze the following policy document and generate practical static assertions:

{policy_document}"""

STATIC_INVARIANT_PROMPT_FLASH = r"""You are an expert at analyzing domain policies to generate meaningful static assertions for agent trajectory verification. Your task is to identify important rules from policy documents that prevent policy violations, errors, and ensure proper operation.
You must extract checkable rules from that can be programmatically verified when specific tool calls are made during an agent's execution.

## Your Task:
Analyze the provided policy document and generate a focused set of static assertions that:
1. Are simple enough to implement programmatically and are objectively verifiable from trajectory data
2. Can be checked at specific tool call triggers
3. Focus on critical policy violations that must be prevented
4. Include both critical business logic AND important error checking
5. Avoid trivial input checks, like assertions that check field completeness
6. Provide working Python code for each assertion that can be executed

## Focus Areas:
Generate assertions for these key constraint types:

## Focus Areas - Generate assertions for:
### **Precondition Checks**
- Required state/status before agent execution
- Examples:
  - Incident Management: "An incident_id must be identifiedbefore any incident-scoped investigation or communication."
  - Incident Management: "Kusto queries must be non-empty and must not contain placeholders like TODO, TBD, or <CLUSTER> before execution."
  - Retail: "Order status must be 'pending' before cancellation"
  - Healthcare: "Patient must be 'checked-in' before treatment actions"
  - Financial: "Account must be 'active' before transaction processing"
  - System Admin: "Service must be 'stopped' before configuration changes"

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
  
## Output Format:
For each assertion, provide:
1. **Assertion Name**: Simple descriptive name (e.g., check_order_status_for_cancellation)
2. **Tool Trigger**: Specific tool call(s) that should trigger this check
3. **Check Logic**: Clear, specific condition to verify
4. **Violation Condition**: When this assertion fails
5. **Python Code**: Complete Python function that implements this assertion check

## Example Output:
```
Assertion Name: check_user_authentication_before_data_access
Agent Trigger: IncidentAgent
Check Logic:  Extract setting token from "[SettingDrift] <SettingName> is drifted" in instruction and ensure Step-1's claimed setting equals it.
Violation Condition: Step-1 extracted setting name differs from incident description setting token.
Python Code: 
def drifted_setting_name_matches_incident_description(trajectory_step, tool_call_params, tool_response):
    import re
    import json
    print("Function: drifted_setting_name_matches_incident_description")
    traj = trajectory_step.get("_trajectory", {}) or {}
    instruction = traj.get("instruction", "") or ""
    print("instruction_length", len(instruction))

    # Canonical incident token comes from instruction (Flash format)
    m_inc = re.search(r"\[SettingDrift\]\s*(\S+)\s+is drifted", instruction)
    if not m_inc:
        raise ValueError("Could not parse setting name from instruction incident description")
    incident_setting = m_inc.group(1)
    print("incident_setting", incident_setting)

    # Step-1 content text
    substeps = trajectory_step.get("substeps", []) or []
    step_text = "\n".join((s.get("content", "") or "") for s in substeps)
    if not step_text:
        step_text = trajectory_step.get("instruction", "") or trajectory_step.get("content", "") or ""
    print("step_text_length", len(step_text))

    # Try to parse ledger JSON to find is_step_finished.reason with quoted setting name
    ledger_setting = None
    json_snippets = re.findall(r"{.*}", step_text, flags=re.DOTALL)
    for snip in json_snippets[::-1]:
        try:
            cand = json.loads(snip)
            if isinstance(cand, dict) and "is_step_finished" in cand:
                reason = (cand.get("is_step_finished", {}) or {}).get("reason", "") or ""
                print("is_step_finished_reason_length", len(reason))
                m_reason = re.search(r"drifted setting name as '([^']+)'", reason)
                if m_reason:
                    ledger_setting = m_reason.group(1)
                else:
                    # fallback: match incident-style token in reason
                    m_reason2 = re.search(r"\[SettingDrift\]\s*(\S+)\s+is drifted", reason)
                    if m_reason2:
                        ledger_setting = m_reason2.group(1)
                break
        except Exception:
            continue

    # Fallback: parse directly from step text if ledger parsing fails
    if not ledger_setting:
        m_fallback = re.search(r"drifted setting name as '([^']+)'", step_text)
        if m_fallback:
            ledger_setting = m_fallback.group(1)

    print("ledger_setting", ledger_setting)
    if not ledger_setting:
        raise ValueError("Could not parse drifted setting name from Step-1")

    if ledger_setting != incident_setting:
        print("Violation: ledger_setting != incident_setting")
        return False

    return True


## Quality Guidelines:
- **Practical Impact**: Focus on assertions that prevent real operational problems
- **Domain-Relevant**: Prioritize rules specific to this business domain
- **Reasonable Scope**: Avoid both trivial sanity checks and computation checks better suited for dynamic invariants

## Instructions:
1. Identify important constraints, rules, and requirements
2. Include critical business logic, essential error checking, and policy compliance
3. Focus on assertions that ensure proper operation and prevent violations
4. Generate a comprehensive but focused set of static assertions
5. AVOID assertions that check:
   - Whether all required fields are filled/provided
   - Whether information was provided earlier

Each assertion generated should be mapped to a tool call.
Each assertion should be as GENERAL as possible, if many specific assertions can be combined into one assertion function, do so.
For example, an assertion which checks order status validity for various tool calls can be combined into one function.

## Python Code Guidelines:
- Write complete, executable Python functions
- You will be given 2 parameters - (trajectory and current_step_index), you can extract all the info required such as arguments for the tool call from the current trajectory step.
- Include docstrings explaining the assertion
- Handle edge cases and missing data gracefully - verify field existence before access
- **Exception Handling**: Raise exceptions for safety check violations (KeyError, IndexError, TypeError, AttributeError, etc.) instead of returning False. Only return boolean (True/False) for actual invariant logic violations
- Return boolean values (True if assertion passes, False if invariant violation detected)
- **Add print statements for debugging**: Start with printing the function name, then print all key variables and extracted JSON fields to aid in debugging and verification

The tools available to the LLM are:

{tools_list}

**IMPORTANT: Review this JSON structure carefully before writing any code. Use ONLY the field names that appear in this structure:**

{tools_structure}"""

DYNAMIC_INVARIANT_PROMPT_FLASH = """You are an expert at analyzing **Flash incident-management** agent trajectories and generating **step-specific dynamic invariants** (programmable checks) for trajectory verification.
Your task is to analyze the current step in an agent's execution trajectory and determine if any invariants should be generated for verification for this specific step.

Generate ONLY ONE (or two if absolutely necessary) invariant and ONLY when they are necessary and valuable for detecting potential errors or policy violations at THIS step.
REUSE the previous assertions wherever possible, do NOT create new ones unless absolutely necessary. Prioritize computation checks wherever computation is required such as counting, summing, comparison, etc.

You will be provided with the overall task instruction, the domain policy, the static invariants already generated using the domain policy, and the full agent trajectory up to the current step.

**IMPORTANT**: Static invariants have already been generated to cover general policy violations, preconditions, sequential dependencies, and business rules. Focus ONLY on cases NOT covered by static invariants, such as:
- Computation errors (counting, summing, calculations)
- Step-specific validation that static invariants cannot capture

## Context Information:
**Task Instruction:** {task_instruction}
**Current Step:** {step_num}

Static invariants already cover most general policy violations (preconditions, sequencing, non-placeholder queries, grounding / evidence rules, tool-role separation, etc.).
Dynamic invariants are ONLY for gaps static invariants cannot capture, especially:
- **Computation / aggregation correctness** (sums, counts, dcount, max/min, thresholds)
- **Step-specific numeric interpretation** (e.g., “0 clusters”, “N impacted tenants”, “total tasks = …”)
- **Cross-checking an LLM-stated number against the tool output at (or before) this step**

You will be provided with:
- the overall task instruction,
- the static invariants already generated,
- the full trajectory up to the current step (step-indexed),
- and the dynamic assertions generated so far.

**IMPORTANT**: Do NOT generate invariants that are already covered by static invariants or earlier dynamic invariants.
If this step is already covered, respond with **NO INVARIANT NEEDED**.

================================================================================
## Context Information
**Task Instruction:** {task_instruction}

**Flash Domain Policy:**
{domain_policy}

**Current Step:** {step_num}
================================================================================

## Flash-Specific Guidance
Flash trajectories typically involve roles like:
- **IncidentAgent**: fetches incident metadata/description and identifiers
- **KustoAgent**: executes Kusto queries and returns query outputs (tables / time-series / counts)
- **Orchestrator / GeneralAssistant**: interprets tool outputs and proposes conclusions/actions

Dynamic invariants for Flash should focus on:
1) **Kusto result interpretation correctness**:
   - If the agent claims “0 clusters” or “no impacted tenants”, verify the Kusto output supports it (e.g., dcount == 0).
   - If the agent claims a numeric total (e.g., total tasks), verify it equals the sum/aggregation from the returned series/rows.
2) **Threshold / comparison correctness** (only if policy defines thresholds and the tool output contains comparable values):
   - If the agent claims “spike above baseline” or “threshold exceeded”, verify the numeric comparison is correct.
3) **Cross-step numeric consistency**:
   - If step {step_num} references counts from an earlier tool step, verify the referenced number matches that earlier tool output.

================================================================================
## Your Task:
Analyze the current step in the trajectory and generate dynamic invariants that:
1. Are simple enough to implement programmatically
2. Can be checked at this specific step in the trajectory (e.g., computation checks from tool call results)
3. Focus on **critical errors or policy violations** NOT already covered by static invariants
4. Are objectively verifiable from trajectory data
5. Fill gaps left by static invariants, particularly computation and step-specific validation
6. Provide working Python code for each dynamic assertion that can be executed
================================================================================

## Required Output Format
If you generate an invariant, you MUST output it in EXACTLY this shape:

Step Number: <step_num>
Assertion Name: <snake_case_name>
Reasoning: <explain what you are checking and exactly which step indices/fields you read>
Primary Step Number: <step_num_or_prior_step_int>
Check Logic: <one clear condition>
Violation Condition: <when it fails>
Python Code:
    def <assertion_name>(trajectory_step, tool_call_params, tool_response):
        ...

If NO invariant is needed, output EXACTLY:
Step Number: {step_num}
NO INVARIANT NEEDED
Reason: <brief explanation why no check is necessary or it is already covered>

================================================================================
## Required Output Format

Step Number: <step_num>
Assertion Name: kusto_dcount_interpretation_correct
Reasoning: The assistant claims “0 impacted tenants” at step <step_num>. Verify this matches the Kusto query output in trajectory[<step_num>]['tool_response'] by extracting the dcount result (e.g., dcount(serviceId) or similar) from the returned table/rows and checking it equals 0.
Primary Step Number: <step_num>
Check Logic: If the step claims “0 impacted tenants/traffic”, then dcount(serviceId) from the tool response must be 0.
Violation Condition: Assistant claims 0 impacted tenants but Kusto output shows non-zero dcount.
Python Code:
    def kusto_dcount_interpretation_correct(trajectory_step, tool_call_params, tool_response):
        print("Function: kusto_dcount_interpretation_correct")
        print("tool_response_type", type(tool_response))
        if not isinstance(tool_response, dict):
            raise TypeError("tool_response must be a dict for KustoAgent outputs")

        # NOTE: The runner will provide the exact schema; inspect keys defensively.
        print("tool_response_keys", list(tool_response.keys()))

        # Example extraction pattern (adjust to real schema via tool_structure):
        # - tool_response may contain "tables": [{"columns": [...], "rows": [[...], ...]}]
        tables = tool_response.get("tables")
        if not isinstance(tables, list) or not tables:
            raise KeyError("Expected non-empty 'tables' in tool_response")

        first = tables[0] or {}
        cols = first.get("columns") or []
        rows = first.get("rows") or []
        if not isinstance(cols, list) or not isinstance(rows, list):
            raise TypeError("'columns' and 'rows' must be lists")
        if not cols or not rows:
            raise ValueError("No data rows returned for dcount interpretation check")

        col_names = [c.get("name") for c in cols if isinstance(c, dict)]
        print("col_names", col_names)

        # Try common dcount column names; final choice must match actual tool_structure.
        candidates = ["dcount_serviceId", "dcount(serviceId)", "dcount"]
        idx = None
        for name in candidates:
            if name in col_names:
                idx = col_names.index(name)
                break
        if idx is None:
            raise KeyError("Could not find a dcount column in the returned columns")

        val = rows[0][idx]
        print("dcount_val_raw", val)
        try:
            dcount_val = int(val)
        except Exception as e:
            raise TypeError("dcount value is not int-like") from e

        # Extract the assistant claim text from the current step (optional; schema-dependent).
        # If you can’t reliably detect the claim text, skip claim parsing and just validate the number itself.
        claim_zero = True  # conservative default for the example
        print("dcount_val", dcount_val, "claim_zero", claim_zero)

        if claim_zero:
            return dcount_val == 0
        return True


Step Number: <step_num>
Assertion Name: kusto_time_series_total_matches_summary
Reasoning: Step <step_num> summarizes a “total task count” from a Kusto time-series result. Verify the stated total equals the sum over the per-bin counts in trajectory[<step_num>]['tool_response'] (same step tool response). This detects arithmetic / aggregation mistakes in the assistant’s summary.
Primary Step Number: <step_num>
Check Logic: total_count == sum(per_bin_counts)
Violation Condition: Total count mismatch between summarized total and computed sum from the tool output series.
Python Code:
    def kusto_time_series_total_matches_summary(trajectory_step, tool_call_params, tool_response):
        print("Function: kusto_time_series_total_matches_summary")
        if not isinstance(tool_response, dict):
            raise TypeError("tool_response must be a dict")
        print("tool_response_keys", list(tool_response.keys()))

        # Example only; MUST align with actual tool_structure.
        total = tool_response.get("total_count")
        series = tool_response.get("counts_by_bin")
        print("total_raw", total, "series_type", type(series))
        if total is None or series is None:
            raise KeyError("Expected 'total_count' and 'counts_by_bin' in tool_response")
        if not isinstance(series, list):
            raise TypeError("'counts_by_bin' must be a list of numeric bin counts")

        try:
            total_int = int(total)
        except Exception as e:
            raise TypeError("'total_count' is not int-like") from e

        calc = 0
        for x in series:
            try:
                calc += int(x)
            except Exception as e:
                raise TypeError("Bin count not int-like") from e
        print("total_int", total_int, "calculated_total", calc)
        return total_int == calc

================================================================================
## Implementation Guidelines:
# 1. Avoid Duplication: DO NOT create invariants already covered by static invariants. Focus on computation errors and step-specific validation. 
# 2. Simple and Focused: Exactly one concrete condition which is practical and programmable, the assertion logic should be as GENERAL as possible which can also be reused later on. Keep the assertion name simple.
# 3. Step-verifiable: Check can be performed now using current + prior trajectory data (tool name, params, response, prior states). Each assertion should clearly apply to the current trajectory step
# 4. Impact: Target computation errors, data accuracy issues, and step-specific violations NOT covered by static invariants.
# 5. Practical: Better to have an implementable assertion than a complex one, expressible as a boolean, required fields are observable in trajectory/tool outputs.
# 6. Objective: Avoid **subjective judgments** or very complex reasoning requirements

## Python Code Guidelines:
- Write complete, executable Python functions
- You will be given 2 parameters - (trajectory and current_step_index), you can extract all the info required such as arguments for the tool call and tool results from the trajectory steps till now. Current trajectory step is trajectory[current_step_index]
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

## Output Format:
For each assertion, provide:
1. **Step Number**: Current step number ({step_num})
2. **Assertion Name**: Simple descriptive name (e.g., check_tshirt_count_accuracy)
3. **Reasoning**: Explain what logic you are checking and where in the trajectory (specific step indices) you are looking to extract the necessary information for this assertion
4. **Primary Step Number**: Single step number (integer) that is the main/primary step you are examining in the trajectory based on your reasoning above
5. **Check Logic**: Clear, specific condition to verify
6. **Violation Condition**: When this assertion fails
7. **Python Code**: Complete Python function that implements this assertion check

**Static Invariants Already Generated:** 
{static_invariants}

## Dynamic Assertions Generated for Previous Steps:
{previous_assertions}

You NEED NOT GENERATE AN INVARIANT for every step IF PREVIOUSLY COVERED, AVOID DUPLICATION. Only generate new ones when absolutely necessary and when the case is NOT covered by static invariants.

1. Check if the current step is already covered by a previous invariant whether static or dynamic
2. If covered, respond with "NO INVARIANT NEEDED" and a brief explanation
3. If not covered, generate a new invariant

If no invariant is needed for this step or is already covered by a previous invariant generated, simply respond with: 
```
Step Number: {step_num}
NO INVARIANT NEEDED
Reason: [Brief explanation why no check is necessary for this step]
```

## Trajectory Steps (up to current step):
{trajectory_till_current_step}
"""

DYNAMIC_INVARIANT_PROMPT = """You are an expert at analyzing agent trajectories and generating step-specific dynamic invariants which are programmable for trajectory verification. 
Your task is to analyze the current step in an agent's execution trajectory and determine if any invariants should be generated for verification for this specific step.

Generate ONLY ONE (or two if absolutely necessary) invariant and ONLY when they are necessary and valuable for detecting potential errors or policy violations at THIS step.
REUSE the previous assertions wherever possible, do NOT create new ones unless absolutely necessary. Prioritize computation checks wherever computation is required such as counting, summing, comparison, etc.

You will be provided with the overall task instruction, the domain policy, the static invariants already generated using the domain policy, and the full agent trajectory up to the current step.

**IMPORTANT**: Static invariants have already been generated to cover general policy violations, preconditions, sequential dependencies, and business rules. Focus ONLY on cases NOT covered by static invariants, such as:
- Computation errors (counting, summing, calculations)
- Step-specific validation that static invariants cannot capture

## Context Information:
**Task Instruction:** {task_instruction}

**Domain Policy:** 
{domain_policy}

**Current Step:** {step_num}

## Your Task:
Analyze the current step in the trajectory and generate dynamic invariants that:
1. Are simple enough to implement programmatically
2. Can be checked at this specific step in the trajectory (e.g., computation checks from tool call results)
3. Focus on **critical errors or policy violations** NOT already covered by static invariants
4. Are objectively verifiable from trajectory data
5. Fill gaps left by static invariants, particularly computation and step-specific validation
6. Provide working Python code for each dynamic assertion that can be executed

## Focus Areas:
Generate assertions for these key constraint types (ONLY if NOT covered by static invariants):

### **Computation Checks**
- Verify numerical calculations, aggregations, and data accuracy
- Examples:
  - Retail: "Total available t-shirts count in response given by LLM should match the json output of the tool call"
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
- Verify data integrity across multiple tool interactions within the same session
- Examples:
  - Retail: "Items added to cart must still be available when proceeding to checkout"
  - Financial: "Exchange rates used in calculations must be consistent across related transactions"
  - Healthcare: "Lab results referenced in diagnosis must match the lab results retrieved"
  - System: "File permissions checked must match the permissions set in previous operations"

## Example Output:
```
Step Number: <step_num>
Assertion Name: count_verification
Reasoning: Checking if the total count stated in the tool response at step <step_num> matches the sum of individual size counts from the same response. Looking at trajectory[<step_num>]['tool_response'] to extract both total_count and size_counts fields.
Primary Step Number: <step_num>
Check Logic: Verify total available t-shirts equals sum of counts across all sizes in JSON response
Violation Condition: Total count mismatch between aggregated value and individual size counts
Python Code:
    def count_verification(trajectory_step, tool_call_params, tool_response):
        # Verify total count equals sum of individual counts
        if tool_response and 'total_count' in tool_response:
            total = tool_response['total_count']
            size_counts = tool_response['size_counts']
            calculated_total = sum(size_counts.values())
            return total == calculated_total
        return False

Step Number: <step_num>
Assertion Name: check_price_calculation_accuracy
Reasoning: Verifying the price calculation accuracy by comparing the stated total_price with manually calculated total from item quantities and unit prices. Examining trajectory[<step_num>]['tool_response'] which contains 'items' array and 'total_price' field.
Primary Step Number: <step_num>
Check Logic: Verify calculated total price matches sum of (quantity * unit_price) for all items
Violation Condition: Price calculation error in order total computation
Python Code:
    def check_price_calculation_accuracy(trajectory_step, tool_call_params, tool_response):
        # Verify price calculation accuracy
        if tool_response and 'items' in tool_response and 'total_price' in tool_response:
            items = tool_response['items']
            stated_total = tool_response['total_price']
            calculated_total = sum(item.get('quantity', 0) * item.get('unit_price', 0) for item in items)
            return abs(stated_total - calculated_total) < 0.01
        return False
```

## Implementation Guidelines:
# 1. Avoid Duplication: DO NOT create invariants already covered by static invariants. Focus on computation errors and step-specific validation. 
# 2. Simple and Focused: Exactly one concrete condition which is practical and programmable, the assertion logic should be as GENERAL as possible which can also be reused later on. Keep the assertion name simple.
# 3. Step-verifiable: Check can be performed now using current + prior trajectory data (tool name, params, response, prior states). Each assertion should clearly apply to the current trajectory step
# 4. Impact: Target computation errors, data accuracy issues, and step-specific violations NOT covered by static invariants.
# 5. Practical: Better to have an implementable assertion than a complex one, expressible as a boolean, required fields are observable in trajectory/tool outputs.
# 6. Objective: Avoid **subjective judgments** or very complex reasoning requirements

## Python Code Guidelines:
- Write complete, executable Python functions
- You will be given 2 parameters - (trajectory and current_step_index), you can extract all the info required such as arguments for the tool call and tool results from the trajectory steps till now. Current trajectory step is trajectory[current_step_index]
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

## Output Format:
For each assertion, provide:
1. **Step Number**: Current step number ({step_num})
2. **Assertion Name**: Simple descriptive name (e.g., check_tshirt_count_accuracy)
3. **Reasoning**: Explain what logic you are checking and where in the trajectory (specific step indices) you are looking to extract the necessary information for this assertion
4. **Primary Step Number**: Single step number (integer) that is the main/primary step you are examining in the trajectory based on your reasoning above
5. **Check Logic**: Clear, specific condition to verify
6. **Violation Condition**: When this assertion fails
7. **Python Code**: Complete Python function that implements this assertion check

**Static Invariants Already Generated:** 
{static_invariants}

## Dynamic Assertions Generated for Previous Steps:
{previous_assertions}

You NEED NOT GENERATE AN INVARIANT for every step IF PREVIOUSLY COVERED, AVOID DUPLICATION. Only generate new ones when absolutely necessary and when the case is NOT covered by static invariants.

1. Check if the current step is already covered by a previous invariant whether static or dynamic
2. If covered, respond with "NO INVARIANT NEEDED" and a brief explanation
3. If not covered, generate a new invariant

If no invariant is needed for this step or is already covered by a previous invariant generated, simply respond with: 
```
Step Number: {step_num}
NO INVARIANT NEEDED
Reason: [Brief explanation why no check is necessary for this step]
```

## Trajectory Steps (up to current step):
{trajectory_till_current_step}
"""

REFLECTION_PROMPT = """You are an expert Python programmer specializing in debugging and fixing code for trajectory verification assertions. 
An assertion function which returns a boolean has failed during execution, and your task is to analyze the error and generate a corrected version of the code.

## Context:
An assertion function designed to verify agent trajectory steps encountered an execution error on the latest step of the trajectory.
You need to:
1. Understand the original intent of the assertion
2. Analyze the error that occurred
3. Generate a fixed, working version of the assertion code

## Original Assertion Details:

**Assertion Logic:**
{assertion_logic}

**Original Python Code:**
```python
{original_code}
```

**Execution Error Description:**
{error_description}

## Trajectory Context (Last 10 Steps Before Failure):
{trajectory_context}

## Your Task:
Analyze the error and generate a corrected version of the assertion function that:
1. **Fixes the Root Cause**: Address the specific error that occurred
2. **Maintains Original Intent**: Keep the same verification logic and purpose
3. **Handles Edge Cases**: Add proper error handling for missing/null data, type mismatches, and unexpected formats
4. **Robust Implementation**: Use defensive programming practices (type checks, None checks, .get() for dictionaries)
5. **Clear and Readable**: Include comments explaining fixes made

## Common Error Patterns to Consider:
- **KeyError/AttributeError**: Missing keys in dictionaries or attributes in objects
  - **ROOT CAUSE**: Often caused by using incorrect field names or assuming fields exist
  - **FIX**: Use `.get()` method with defaults: `data.get('field_name', default_value)`
  - **VERIFY**: Check the actual tool response structure in trajectory context to confirm exact field names
  - Always check if attribute/key exists before accessing
  
- **TypeError**: Incorrect data types or operations on None values
  - **ROOT CAUSE**: Accessing nested fields without checking parent field exists, or wrong data type assumptions
  - **FIX**: Chain .get() calls for nested access: `data.get('outer', {}).get('inner')`
  - Validate data types before operations: `isinstance(value, expected_type)`
  - Handle None/null values explicitly before any operations
  
- **IndexError**: List/array index out of bounds
  - **ROOT CAUSE**: Assuming list has elements without checking length
  - **FIX**: Check list length before indexing: `if len(list) > index:`
  - Use safe iteration patterns with bounds checking
  
- **ValueError**: Invalid conversions or unexpected data formats
  - **ROOT CAUSE**: Attempting type conversions on incompatible data or unexpected formats
  - **FIX**: Validate data format before conversion
  - Use try-except for risky conversions with appropriate fallback values
  
- **Logical Errors**: Incorrect conditions or comparisons
  - **ROOT CAUSE**: Misunderstanding the data structure or using wrong field names
  - **FIX**: Re-examine the trajectory context to see actual field names and values
  - Review comparison operators and boolean logic
  - Ensure conditions match intended assertion logic

**DEBUGGING STRATEGY**: When fixing errors, first examine the trajectory context to see the ACTUAL structure and field names in the tool response, then adjust the code to match that exact structure.

## Output Format:
Provide your corrected code in while keeping the same intent and function signature as the original.
"""
