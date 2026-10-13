"""Compute budgets; these are thread limits, not a CPU utilization promise."""
import os


def high_speed_threads(logical_threads=None):
    logical = max(1, int(logical_threads or os.cpu_count() or 1))
    return max(1, logical * 85 // 100)
