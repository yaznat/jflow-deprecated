package jflow.model;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;

import jflow.utils.AnsiCodes;

/**
 * Tracks training and validation metrics across epochs and manages model checkpointing
 * based on improvements in a selected metric.
 * <p>
 * Stores metric values for all epochs to enable plotting or analysis after training.
 * </p>
 */
public class ModelCheckpoint {
    private final MetricType metric;
    private final String savePath;
    private boolean improved = true;

    private final Map<MetricType, List<Double>> history = new EnumMap<>(MetricType.class);
    private final Map<MetricType, Double> bestValues = new EnumMap<>(MetricType.class);

    private enum MetricType {
        TRAIN_ACCURACY("Training accuracy", true, Double.NEGATIVE_INFINITY),
        TRAIN_LOSS("Training loss", false, Double.POSITIVE_INFINITY),
        VAL_ACCURACY("Validation accuracy", true, Double.NEGATIVE_INFINITY),
        VAL_LOSS("Validation loss", false, Double.POSITIVE_INFINITY);

        final String description;
        final boolean higherIsBetter;
        final double initialBest;

        MetricType(String description, boolean higherIsBetter, double initialBest) {
            this.description = description;
            this.higherIsBetter = higherIsBetter;
            this.initialBest = initialBest;
        }

        public boolean isImproved(double current, double best) {
            return higherIsBetter ? current > best : current < best;
        }

        public static MetricType fromString(String name) {
            return switch (name) {
                case "train_accuracy" -> TRAIN_ACCURACY;
                case "train_loss"     -> TRAIN_LOSS;
                case "val_accuracy"   -> VAL_ACCURACY;
                case "val_loss"       -> VAL_LOSS;
                default -> throw new IllegalArgumentException("Unknown metric: " + name);
            };
        }
    }

    /**
     * Passes data to the train function to faciliate the saving of model checkpoints.
     * @param metric                            the metric to track for improvement. Supported: 
     *                                            <ul> <li> val_loss <li> val_accuracy 
     *                                                 <li> train_loss <li> train_accuracy </ul>
     * @param savePath                          the path to save checkpoints to.
     */
    public ModelCheckpoint(String metricName, String savePath) {
        this.metric = MetricType.fromString(metricName);
        this.savePath = savePath;

        for (MetricType type : MetricType.values()) {
            history.put(type, new ArrayList<>(List.of(type.initialBest)));
            bestValues.put(type, type.initialBest);
        }
    }

    protected boolean improved() {
        return improved;
    }

    protected String getSavePath() {
        return savePath;
    }

    /**
     * Updates metric trackers with new values and prints a detailed callback denoting if 
     * the selected metric has improved.
     * @param trainAccuracy                 The training accuracy from this epoch.
     * @param trainLoss                     The training loss from this epoch.
     * @param valAccuracy                   The validation accuracy from this epoch.
     * @param valLoss                       The validation loss from this epoch.
     */
    protected void updateAndPrintCallback(
        double trainAccuracy, 
        double trainLoss,
        double valAccuracy,
        double valLoss
    ) {
        // Update all histories
        history.get(MetricType.TRAIN_ACCURACY).add(trainAccuracy);
        history.get(MetricType.TRAIN_LOSS).add(trainLoss);
        history.get(MetricType.VAL_ACCURACY).add(valAccuracy);
        history.get(MetricType.VAL_LOSS).add(valLoss);

        // Determine improvement
        double current = switch (metric) {
            case TRAIN_ACCURACY -> trainAccuracy;
            case TRAIN_LOSS     -> trainLoss;
            case VAL_ACCURACY   -> valAccuracy;
            case VAL_LOSS       -> valLoss;
        };

        double best = bestValues.get(metric);
        improved = metric.isImproved(current, best);

        // Format and print
        String oldValStr = formatValue(best, metric);
        String newValStr = formatValue(current, metric);
        String label = metric.description;

        if (improved) {
            bestValues.put(metric, current);
            System.out.println(AnsiCodes.WHITE + label +
                " improved from " + AnsiCodes.BLUE + oldValStr +
                AnsiCodes.WHITE + " to " + AnsiCodes.BLUE + newValStr +
                AnsiCodes.WHITE + ". Saving model to " + AnsiCodes.BLUE +
                savePath + AnsiCodes.RESET);
        } else {
            System.out.println(AnsiCodes.WHITE + label +
                " did not improve from " + AnsiCodes.BLUE + oldValStr +
                AnsiCodes.RESET);
        }
    }

    private String formatValue(double val, MetricType type) {
        if (Double.isInfinite(val)) return String.valueOf(val);

        int decimals = type.higherIsBetter ? 5 : 8;

        if(type.higherIsBetter) {
            val *= 100;
        }

        String valAsString = String.valueOf(val);
        // Pad with trailing zeros if too short
        while (valAsString.length() < decimals) {
            valAsString += "0";
        }
        // Truncate if too long
        valAsString = valAsString.substring(0, decimals);

        if(type.higherIsBetter) {
            valAsString += "%";
        }
        return valAsString;
    }
}
