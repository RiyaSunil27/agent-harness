"""
HR database and tool definitions used by Exercise 3.
Not part of the exercise — provided so exercise_3.py stays readable.
"""

import json

EMPLOYEE_DB = {
    "EMP001": {"name": "Alice Johnson",  "department": "Engineering", "leave_casual": 8,  "leave_sick": 5,  "leave_earned": 12, "manager": "Carol White"},
    "EMP002": {"name": "Bob Smith",      "department": "HR",          "leave_casual": 6,  "leave_sick": 3,  "leave_earned": 15, "manager": "David Lee"},
    "EMP003": {"name": "Priya Nair",     "department": "Finance",     "leave_casual": 10, "leave_sick": 7,  "leave_earned": 8,  "manager": "Carol White"},
    "EMP004": {"name": "James Okafor",   "department": "Engineering", "leave_casual": 5,  "leave_sick": 2,  "leave_earned": 20, "manager": "Carol White"},
    "EMP005": {"name": "Sarah Chen",     "department": "Marketing",   "leave_casual": 9,  "leave_sick": 4,  "leave_earned": 6,  "manager": "David Lee"},
}

DEPARTMENT_DB = {
    "Engineering": {"headcount": 24, "manager": "Carol White", "location": "Floor 3"},
    "HR":          {"headcount": 8,  "manager": "David Lee",   "location": "Floor 1"},
    "Finance":     {"headcount": 12, "manager": "Ravi Menon",  "location": "Floor 2"},
    "Marketing":   {"headcount": 10, "manager": "David Lee",   "location": "Floor 2"},
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_leave_balance",
            "description": "Get remaining leave days for an employee by employee ID",
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_id": {"type": "string", "description": "Employee ID, e.g. EMP001"}
                },
                "required": ["employee_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_department_info",
            "description": "Get headcount, manager, and office location for a department",
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {"type": "string", "description": "Department name, e.g. Engineering"}
                },
                "required": ["department"],
            },
        },
    },
]


def execute_tool(name: str, args: dict) -> str:
    """Call the right function and return a JSON string result."""
    if name == "get_leave_balance":
        emp = EMPLOYEE_DB.get(args.get("employee_id", ""))
        if not emp:
            return json.dumps({"error": "Employee not found"})
        return json.dumps({**emp, "employee_id": args["employee_id"]})
    if name == "get_department_info":
        dept = DEPARTMENT_DB.get(args.get("department", ""))
        if not dept:
            return json.dumps({"error": "Department not found"})
        return json.dumps({**dept, "department": args["department"]})
    return json.dumps({"error": f"Unknown tool: {name}"})
