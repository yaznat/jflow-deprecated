package jflow.utils;

import jflow.data.Image;
import jflow.data.JMatrix;

public class JPlot {

    /**
     * Displays an image in a new JFrame.
     * @param image image data wrapped in an Image.
     */
    public static void displayImage(Image image) {
        displayImage(image, 1);
    }
    /**
     * Displays an image in a new JFrame with a scale factor.
     * @param image             image data wrapped in an Image.
     * @param scaleFactor       The scale factor of the display.
     */
    public static void displayImage(Image image, int scaleFactor) {
        JMatrix pixels = image.getPixels();
        // Display the image

        new ImageDisplay(pixels, scaleFactor, String.valueOf(image.getLabel()));
    }

    /**
     * Displays an image in a new JFrame with a scale factor.
     * @param image             image data wrapped in a JMatrix with shape (1, channels, height, width).
     * @param scaleFactor       The scale factor of the display.
     */
    public static void displayImage(JMatrix image, int scaleFactor, String title) {
        // Display the image
        new ImageDisplay(image, scaleFactor, title);
    }
}
