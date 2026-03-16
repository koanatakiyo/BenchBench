"""
Mappings for CSBench subdomains → CS.Core.* super parents.
"""

from typing import Dict, List, Set

CS_SUPER_PARENTS = {
    "CS.Core.DataStructuresAlgorithms": [
        "data structure", "data structures", "linear list", "linked list",
        "stack", "queue", "tree", "graph", "graph theory",
        "sorting", "searching", "algorithm", "dynamic programming",
        "hash table", "recursion", "栈", "队列", "树", "图", "排序", "查找", "算法", "递归"
    ],
    "CS.Core.ComputerOrganization": [
        "computer architecture", "computer organization", "cpu architecture",
        "instruction cycle", "pipeline", "memory hierarchy", "cache",
        "register", "assembly", "计算机组成原理", "指令周期", "流水线", "寄存器", "主存", "cache"
    ],
    "CS.Core.ComputerNetwork": [
        "computer network", "network", "tcp/ip", "network layer",
        "transport layer", "application layer", "data link layer",
        "网络层", "传输层", "应用层", "数据链路层", "计算机网络", "通信", "路由"
    ],
    "CS.Core.OperatingSystem": [
        "operating system", "process management", "memory management",
        "storage system", "file system", "process scheduling",
        "concurrency", "synchronization", "io management",
        "输入输出管理", "内存管理", "进程与线程", "操作系统", "存储系统", "文件系统", "调度"
    ],
}


def map_cs_parents(subdomains: List[str]) -> List[str]:
    tags: Set[str] = set()
    for sub in subdomains:
        if not isinstance(sub, str):
            continue
        norm = sub.lower()
        for super_parent, keywords in CS_SUPER_PARENTS.items():
            if any(keyword.lower() in norm for keyword in keywords):
                tags.add(super_parent)
    return sorted(tags)

