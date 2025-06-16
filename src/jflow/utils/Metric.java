package jflow.utils;

public record Metric(String name, double value, boolean isPercentage, boolean improved) {}