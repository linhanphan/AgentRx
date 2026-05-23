import os
import json
from datetime import datetime
import agentrx.pipeline.globals as g

def serialize_custom(obj):
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        def _to_dict(item):
            if hasattr(item, "__dict__"):
                return {k: _to_dict(v) for k, v in item.__dict__.items()}
            elif isinstance(item, list):
                return [_to_dict(x) for x in item]
            elif isinstance(item, dict):
                return {k: _to_dict(v) for k, v in item.items()}
            else:
                return item
        return _to_dict(obj)
    elif isinstance(obj, list):
        return [serialize_custom(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: serialize_custom(v) for k, v in obj.items()}
    else:
        try:
            # Test if it is JSON-serializable as is
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)

def dump_call(request_payload, response_obj):
    try:
        # Create directory inside the current run directory, or fallback to project root
        base_dir = getattr(g, "RUN_DIR", None) or os.getcwd()
        call_raw_dir = os.path.join(base_dir, "call_raw")
        os.makedirs(call_raw_dir, exist_ok=True)
        
        # Generate high-precision datetime string
        dt_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        req_filename = f"{dt_str}_request.json"
        resp_filename = f"{dt_str}_response.json"
        
        req_path = os.path.join(call_raw_dir, req_filename)
        resp_path = os.path.join(call_raw_dir, resp_filename)
        
        # Write request
        with open(req_path, "w", encoding="utf-8") as f:
            json.dump(serialize_custom(request_payload), f, ensure_ascii=False, indent=2)
            
        # Write response
        with open(resp_path, "w", encoding="utf-8") as f:
            json.dump(serialize_custom(response_obj), f, ensure_ascii=False, indent=2)
            
    except Exception as e:
        print(f"[DUMP_CALL_ERROR] Failed to dump LLM call: {e}")
