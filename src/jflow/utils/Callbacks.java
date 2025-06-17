package jflow.utils;

import java.util.List;
import java.util.LinkedHashMap;
import java.util.Map;

import jflow.data.JMatrix;
import jflow.model.Sequential;

/**
 * A compilation of useful printouts for custom train steps.
 */
public class Callbacks {
    private final static String SEPARATOR = 
        AnsiCodes.TEAL + " | " + AnsiCodes.RESET;


    /**
     * Print a detailed, ANSI-styled debug line in the terminal 
     * for a group of JMatrixes. <p>
     * Included stats: <ul>
     * <li> shape </li>
     * <li> absmean </li>
     * <li> L1 norm </li>
     * <li> L2 norm </li>
     * </ul>
     * @param debugData An array of JMatrixes grouped semantically.
     * @param debugTitle The title of this JMatrix group. Enter "" for no title.
     */
    public static void printStats(String debugTitle, JMatrix... debugData) {
        if (!debugTitle.equals("")) {
            debugTitle = " " + debugTitle + " ";
        }
        System.out.println(
                AnsiCodes.BLUE + "╭────────────────────" + AnsiCodes.BOLD + 
                debugTitle + AnsiCodes.RESET + AnsiCodes.BLUE + "────────────────────╮");

            // Iterate over matrices
            for (JMatrix data : debugData) {
                System.out.print(AnsiCodes.BLUE + "│ ");
                // Print name
                String dataName = data.getName();
                System.out.print(AnsiCodes.TEAL + dataName);
                // Print statistics
                System.out.print(SEPARATOR + AnsiCodes.ORANGE + "shape (N, C, H, W): " + AnsiCodes.WHITE + data.shapeAsString()); 
                System.out.print(SEPARATOR + AnsiCodes.ORANGE + "absmean: " + AnsiCodes.WHITE + data.absMean());
                System.out.print(SEPARATOR + AnsiCodes.ORANGE + "L1: " + AnsiCodes.WHITE + data.l1Norm());
                System.out.print(SEPARATOR + AnsiCodes.ORANGE + "L2: " + AnsiCodes.WHITE + data.l2Norm());

                System.out.println();
            }
            String closer = AnsiCodes.BLUE + "╰────────────────────";
            for (int i = 0; i < debugTitle.length(); i++) {
                closer += "─";
            }
            closer += "────────────────────╯" + AnsiCodes.RESET;
            System.out.println(closer);
        
    }
    /**
     * Prints a formatted header using ANSI styling, indicating that training has begun.
     * @param name The name of the model or setup that is undergoing training.
     */
    public static void printTrainingHeader(Sequential model) {
        System.out.println(
            AnsiCodes.BLUE + "=================== " +
            AnsiCodes.BOLD + "Training: " + model.name() + AnsiCodes.RESET +
            AnsiCodes.BLUE + " ==================" + AnsiCodes.RESET
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
     * Prints a formatted report of training metrics using ANSI styling.
     * Intended to follow Callbacks.printProgressCallback() after each training epoch or batch.
     *
     * @param metrics A list of metrics, each containing a name, numeric value,
     *                a flag indicating if it's a percentage (e.g., accuracy),
     *                and a flag for whether it improved.
     */
    public static void printMetricCallback(List<Metric> metrics) {
        System.out.print("\n");
        for (Metric m : metrics) {
            String name = m.name();
            double value = m.value();

            String valueAsString;
            if (m.isPercentage()) {
                valueAsString = capDouble(100 * value, 5) + "%";
            } else {
                valueAsString = capDouble(value, 8);
            }

            String color;
            if (m.improved()) {
                color = AnsiCodes.GREEN;
            } else {
                color = AnsiCodes.RED;
            }
            System.out.println(AnsiCodes.BLUE + name + ": " + 
                color + valueAsString + AnsiCodes.RESET);
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
        report += AnsiCodes.BOLD + AnsiCodes.ORANGE + outerLabel + ": " + 
            AnsiCodes.RESET + AnsiCodes.WHITE + currentMeasurement1 + "/" + totalMeasurement1 + 
            SEPARATOR + AnsiCodes.BOLD + AnsiCodes.ORANGE + measurement2 + ": " + 
            AnsiCodes.RESET + AnsiCodes.WHITE + currentMeasurement2 + "/" + totalMeasurement2;


        // Report losses if applicable
        if (losses != null) {
            for (Map.Entry<String, Double> entry : losses.entrySet()) {
                String lossName = entry.getKey();       
                Double value = entry.getValue(); 
    
                String loss = String.format("%.6f", value);

                report += SEPARATOR + AnsiCodes.BOLD + AnsiCodes.ORANGE + lossName + 
                ": " + AnsiCodes.RESET + AnsiCodes.WHITE + loss + AnsiCodes.RESET;
            }
        }
        // Report ETA
        report += SEPARATOR + AnsiCodes.BOLD + AnsiCodes.ORANGE + "ETA: " + 
            AnsiCodes.RESET + AnsiCodes.WHITE + secondsToClock(
            (int)(timeRemaining * 0.000000001)) + AnsiCodes.RESET;
        
        System.out.print(report);
    }
        
    // Format seconds remaining into an ETA
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

    // Convert a double to a String with given length
    private static String capDouble(double number, int length) {
        if (number == Double.POSITIVE_INFINITY ||
            number == Double.NEGATIVE_INFINITY) {
            return String.valueOf(number);
        }
        String numAsString = String.valueOf(number);
        // Avoid StingIndexOutOfBoundsException
        while (numAsString.length() < length) {
            numAsString += "0";
        }
        return numAsString.substring(0, length);
    }
}
