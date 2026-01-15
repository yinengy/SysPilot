from __future__ import annotations

from dataclasses import dataclass

import psutil


@dataclass
class ProcessInfo:
    pid: int
    name: str
    cmdline: list[str]
    status: str
    username: str


def find_process_by_name(name: str, exact: bool = False) -> list[ProcessInfo]:
    results = []
    name_lower = name.lower()
    
    for proc in psutil.process_iter(["pid", "name", "cmdline", "status", "username"]):
        try:
            proc_name = proc.info["name"] or ""
            
            if exact:
                match = proc_name.lower() == name_lower
            else:
                match = name_lower in proc_name.lower()
                if not match and proc.info["cmdline"]:
                    cmdline_str = " ".join(proc.info["cmdline"]).lower()
                    match = name_lower in cmdline_str
            
            if match:
                results.append(ProcessInfo(
                    pid=proc.info["pid"],
                    name=proc_name,
                    cmdline=proc.info["cmdline"] or [],
                    status=proc.info["status"],
                    username=proc.info["username"] or "",
                ))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    return results


def get_process_tree(pid: int) -> dict:
    try:
        proc = psutil.Process(pid)
        
        tree = {
            "pid": pid,
            "name": proc.name(),
            "status": proc.status(),
            "children": [],
        }
        
        for child in proc.children(recursive=False):
            try:
                tree["children"].append(get_process_tree(child.pid))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        return tree
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {"pid": pid, "error": "Process not accessible"}
