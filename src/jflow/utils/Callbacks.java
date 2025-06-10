package jflow.utils;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A compilation of useful printouts for custom train steps.
 */
public class Callbacks {
    private final static String BLUE = "\033[94m";
    private final static String TEAL = "\033[38;2;0;153;153;1m";
    private final static String ORANGE = "\033[38;2;255;165;1m";
    private final static String WHITE = "\033[37m";
    private final static String GREEN = "\033[38;2;0;204;0m";
    private final static String RED = "\033[38;2;255;0;0m";
    private final static String YELLOW = "\033[38;2;255;255;0m";
    private final static String RESET = "\033[0m";
    private final static String BOLD = "\033[1m";
    private final static String SEPARATOR = TEAL + " | " + RESET;
    /**
     * Prints a formatted header using ANSI styling, indicating that training has begun.
     * @param name The name of the model or setup that is undergoing training.
     */
    public static void printTrainingHeader(String name) {
        System.out.println(
            BLUE + "=================== " +
            BOLD + "Training: " + name + RESET +
            BLUE + " ==================" + RESET
        );
    }
    /**
     * Prints a formatted, real-time training status line to the console using ANSI styling and carriage return (`\r`),
     * allowing the output to update in place during training.
     * <p>
     * This method is designed for use in nested training loops, such as epochs containing batches or steps.
     * It displays progress for both outer and inner loop labels, along with optional loss metrics and an estimated
     * time remaining (ETA) for completing the current outer loop unit (e.g., epoch).
     * 
     * @param outerLabel                A descriptive label for the outer training loop, e.g., "Epoch".    
     * @param outerIndex                The current index (1-based) of progress in the outer loop.
     * @param outerTotal                The total number of iterations in the outer loop.
     * @param innerLabel                A descriptive label for the inner training loop, e.g., "Batch" or "Step".
     * @param innerIndex                The current index (0-based) of progress in the inner loop.
     * @param innerTotal                The total number of iterations in the inner loop.
     * @param elapsedTime               The elapsed time in nanoseconds since the start of the current outer loop iteration.
     *                                  Used to calculate and display an ETA.
     */
    public static void printProgressCallback(
        String outerLabel, int currentMeasurement1, 
        int totalMeasurement1, String measurement2, 
        int currentMeasurement2, int totalMeasurement2,
        long elapsedTime) {
        
        doProgressCallback(outerLabel, currentMeasurement1, totalMeasurement1, 
        measurement2, currentMeasurement2, totalMeasurement2, elapsedTime, null);
    }

    /**
     * Prints a formatted, real-time training status line to the console using ANSI styling and carriage return (`\r`),
     * allowing the output to update in place during training.
     * <p>
     * This method is designed for use in nested training loops, such as epochs containing batches or steps.
     * It displays progress for both outer and inner loop labels, along with optional loss metrics and an estimated
     * time remaining (ETA) for completing the current outer loop unit (e.g., epoch).
     * 
     * @param outerLabel                A descriptive label for the outer training loop, e.g., "Epoch".    
     * @param outerIndex                The current index (1-based) of progress in the outer loop.
     * @param outerTotal                The total number of iterations in the outer loop.
     * @param innerLabel                A descriptive label for the inner training loop, e.g., "Batch" or "Step".
     * @param innerIndex                The current index (0-based) of progress in the inner loop.
     * @param innerTotal                The total number of iterations in the inner loop.
     * @param elapsedTime               The elapsed time in nanoseconds since the start of the current outer loop iteration.
     *                                  Used to calculate and display an ETA.
     * @param losses                    A map of loss names to their current values, which will be displayed in the output.
     */
    public static void printProgressCallback(
        String outerLabel, int currentMeasurement1, 
        int totalMeasurement1, String measurement2, 
        int currentMeasurement2, int totalMeasurement2,
        long elapsedTime, 
        LinkedHashMap<String, Double> losses) {
        
        doProgressCallback(outerLabel, currentMeasurement1, totalMeasurement1, 
        measurement2, currentMeasurement2, totalMeasurement2, elapsedTime, losses);
    }


    /**
     * Prints a formatted report of training metrics using ANSI styling. Designed to follow Callbacks.printProgressCallback()
     * when the inner training loop is complete.
     * @param metrics A map of metric names to their values.
     */
    public static void printMetricCallback(LinkedHashMap<String, Double> metrics) {
        doMetricCallback(metrics, null);
    }

    /**
     * Prints a formatted report of training metrics using ANSI styling. Designed to follow Callbacks.printProgressCallback()
     * when the inner training loop is complete.
     * @param metrics                       A map of metric names to their values.
     * @param improvement                   An array of booleans in order with metrics, 
     *                                      denoting if each value has improved (true), or worsened (false).
     *                                      Improved metrics will be colored GREEN. Worsened metrics will be colored RED. 
     */
    public static void printMetricCallback(LinkedHashMap<String, Double> metrics, boolean[] improvement) {
        doMetricCallback(metrics, improvement);
    }

    private static void doMetricCallback(
        LinkedHashMap<String, Double> metrics, 
        boolean[] improvement) {
        System.out.print("\n");
        int index = 0;
        for (Map.Entry<String, Double> entry : metrics.entrySet()) {
            String metric = entry.getKey();       
            Double value = entry.getValue(); 

            String valueAsString = String.format("%.10f", value);

            String color;
            if (improvement == null) {
                color = YELLOW;
            } else if (improvement[index++]) {
                color = GREEN;
            } else {
                color = RED;
            }

            System.out.println(BLUE + metric + ": " + color + valueAsString + RESET);
        }
    }

    private static void doProgressCallback(
        String outerLabel, int currentMeasurement1, 
        int totalMeasurement1, String measurement2, 
        int currentMeasurement2, int totalMeasurement2,
        long elapsedTime, 
        LinkedHashMap<String, Double> losses) {
        
        // Calculate the time remaining until the next epoch or batch
        long timePerMeasurement2 = elapsedTime / (currentMeasurement2 + 1);
        long timeRemaining = timePerMeasurement2 * (totalMeasurement2 - currentMeasurement2);

        // Replace the last line in the terminal
        String report = "\r";

        // Add epochs and batches
        report += BOLD + ORANGE + outerLabel + ": " + RESET + WHITE + currentMeasurement1 + "/" + totalMeasurement1 + 
            SEPARATOR + BOLD + ORANGE + measurement2 + ": " + RESET + WHITE + currentMeasurement2 + "/" + totalMeasurement2;


        // Report losses if applicable
        if (losses != null) {
            for (Map.Entry<String, Double> entry : losses.entrySet()) {
                String lossName = entry.getKey();       
                Double value = entry.getValue(); 
    
                String loss = String.format("%.6f", value);

                report += SEPARATOR + BOLD + ORANGE + lossName + ": " + RESET + WHITE + loss + RESET;
            }
        }
        // Report ETA
        report += SEPARATOR + BOLD + ORANGE + "ETA: " + RESET + WHITE + secondsToClock(
            (int)(timeRemaining * 0.000000001)) + RESET;
        
        System.out.print(report);
    }

        

    private static String secondsToClock(int totalSeconds) {
        int hours = 0; int minutes = 0;
        // hours
        if (totalSeconds > 3600) {
            int hoursDiv = totalSeconds / 3600;
            totalSeconds -= 3600 * hoursDiv;
            hours += hoursDiv;
        } else if (totalSeconds == 3600) {
            hours++;
            totalSeconds = 0;
        }
        // minutes
        if (totalSeconds > 60) {
            int minutesDiv = totalSeconds / 60;
            totalSeconds -= 60 * minutesDiv;
            minutes += minutesDiv;
        } else if (totalSeconds == 60) {
            minutes++;
            totalSeconds = 0;
        }
        if (hours != 0) {
            return hours + ":" + ((minutes < 10) ? "0" + minutes : "" + minutes);
        }
        return ((minutes < 10) ? "0" + minutes : "" + minutes) + ":" + 
            ((totalSeconds < 10) ? "0" + totalSeconds : "" + totalSeconds);
    }
}
