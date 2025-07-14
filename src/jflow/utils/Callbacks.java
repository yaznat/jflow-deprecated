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
        int shellWidth = 48;
        int topBracketWidth = shellWidth;
        String separator = AnsiCodes.YELLOW + " | ";
        if (!debugTitle.equals("")) {
            debugTitle = " " + debugTitle + " ";
        }
        topBracketWidth -= debugTitle.length();
        boolean oddWidth = topBracketWidth % 2 != 0;
        topBracketWidth /= 2;
        System.out.print(AnsiCodes.BLUE + "╭");
        for (int i = 0; i < topBracketWidth; i++) {
            System.out.print("─");
        }
        if (oddWidth) {
            System.out.print("─");
        }
        System.out.print(AnsiCodes.BOLD + debugTitle + AnsiCodes.RESET + AnsiCodes.BLUE);
        for (int i = 0; i < topBracketWidth; i++) {
            System.out.print("─");
        }
        System.out.println("╮");

        // Find maximum lengths for monospacing
        int numStats = 6;
        int[] maxLengths = new int[numStats];
        for (JMatrix data : debugData) {
            maxLengths[0] = Math.max(maxLengths[0], data.label().length());
            maxLengths[1] = Math.max(maxLengths[1], formatShapeLabel(data.shape()).length());
            maxLengths[2] = Math.max(maxLengths[2], formatShape(data.shape()).length());
            maxLengths[3] = Math.max(maxLengths[3], String.valueOf((float)data.absMean()).length());
            maxLengths[4] = Math.max(maxLengths[4], String.valueOf((float)data.l1Norm()).length());
            maxLengths[5] = Math.max(maxLengths[5], String.valueOf((float)data.l2Norm()).length());
        }
        // Iterate over matrices
        for (JMatrix data : debugData) {
            System.out.print(AnsiCodes.BLUE + "│ ");
            String[] stats = new String[numStats];
            stats[0] = data.label();
            stats[1] = formatShapeLabel(data.shape());
            stats[2] = formatShape(data.shape());
            stats[3] = String.valueOf((float)data.absMean());
            stats[4] = String.valueOf((float)data.l1Norm());
            stats[5] = String.valueOf((float)data.l2Norm());
               
            for (int i = 0; i < numStats; i++) {
                while (stats[i].length() < maxLengths[i]) {
                    stats[i] += " ";
                }
            }
            // Print statistics
            System.out.print(AnsiCodes.TEAL + stats[0]);
            System.out.print(separator + AnsiCodes.ORANGE + "shape " + stats[1] + stats[2]); 
            System.out.print(separator + AnsiCodes.ORANGE + "absMean: " + AnsiCodes.WHITE + stats[3]);
            System.out.print(separator + AnsiCodes.ORANGE + "l1: " + AnsiCodes.WHITE + stats[4]);
            System.out.print(separator + AnsiCodes.ORANGE + "l2: " + AnsiCodes.WHITE + stats[5]);

            System.out.println();
        }
        System.out.print(AnsiCodes.BLUE + "╰");
        for (int i = 0; i < shellWidth; i++) {
            System.out.print("─");
        }
        System.out.println("╯" + AnsiCodes.RESET);
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

    private static String formatShapeLabel(int[] shape) {
        int N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    
        if (H > 1 && W > 1)
            return AnsiCodes.TEAL + "(N, C, H, W)" + AnsiCodes.ORANGE + ":";
        if (H > 1)
            return AnsiCodes.TEAL + "(N, C, F)" + AnsiCodes.ORANGE + ":";
        if (W > 1)
            return AnsiCodes.TEAL + "(N, C, F)" + AnsiCodes.ORANGE + ":"; 
        if (C > 1)
            return AnsiCodes.TEAL + "(N, F)" + AnsiCodes.ORANGE + ":";
        return AnsiCodes.TEAL + "(N,)" + AnsiCodes.ORANGE + ":";
    }

    private static String formatShape(int[] shape) {
        int N = shape[0], C = shape[1], H = shape[2], W = shape[3];
    
        if (H > 1 && W > 1)
            return AnsiCodes.WHITE + " (" + N + "," + C + "," + H + "," + W + ")";
        if (H > 1)
            return AnsiCodes.WHITE + " (" + N + "," + C + "," + H + ")";
        if (W > 1)
            return AnsiCodes.WHITE + " (" + N + "," + C + "," + W + ")";
        if (C > 1)
            return AnsiCodes.WHITE + " (" + N + "," + C + ")";
        return AnsiCodes.WHITE + " (" + N + "," + ")";
    }
}
