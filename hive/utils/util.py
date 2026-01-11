def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))
