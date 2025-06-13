package jflow.data;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.function.Function;

import javax.imageio.ImageIO;

public class Image {
    private JMatrix originalImage;
    private int yData, channels;
    private JMatrix xData;
    private boolean grayscale, lowMemoryMode, loadedFromCSV = false;
    private String path;
    private ArrayList<Function<JMatrix, JMatrix>> transforms = 
    new ArrayList<Function<JMatrix, JMatrix>>();


    /*
     * grayscaleCheck is currently buggy, so the user
     * must denote whether a directory is grayscale
     * or RGB.
     */
    protected Image(String path, int label, boolean grayscale, boolean lowMemoryMode) {
        this.path = path;
        this.grayscale = grayscale;
        this.channels = (grayscale) ? 1 : 3;
        this.lowMemoryMode = lowMemoryMode;
        yData = label;
    }
    // Flattened image from csv
    protected Image(float[] image, int label) {
        this.channels = 1;
        this.loadedFromCSV = true;
        int size = (int)Math.pow(image.length, 0.5);
        originalImage = JMatrix.zeros(1, 1, size, size);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                originalImage.set(0, 0, i, j, image[i * size + j]);
            }
        }
        yData = label;
    }

    protected Image(JMatrix image, int label) {
        this.channels = image.channels();
        originalImage = image;
        yData = label;
    }

    private void load() {
        try {
            BufferedImage img = ImageIO.read(new File(path));
            // 1 channel for grayscale, 3 for RGB
            if (grayscale) {
                originalImage = loadGrayscaleImage(img);
            } else {
                originalImage = loadRGBImage(img);
            }
            xData = originalImage;

        } catch (IOException e) {
            System.err.println("Error loading image: " + e.getMessage());
        }
    }

    private void applyTransforms() {
        for (Function<JMatrix, JMatrix> transform : transforms) {
            xData = transform.apply(xData);
        }
    }

    protected void addTransform(Function<JMatrix, JMatrix> transform) {
        transforms.add(transform);
    }

    public JMatrix getData() {
        loadingSequence();
        JMatrix copyReference = originalImage;
        if (lowMemoryMode) {
            unload();
        }
        return copyReference;
    }

    public float getPixel(int flatIndex) {
        int height = xData.height();
        int width = xData.width();
        int channelSize = height * width;

        int channelIndex = flatIndex / channelSize;
        int reuse = flatIndex % channelSize;
        int heightIndex = reuse / width;
        int widthIndex = reuse % width;

        return xData.get(0, channelIndex, heightIndex, widthIndex);
    }

    public int getLabel() {
        return yData;
    }

    public JMatrix getPixels() {
        if (originalImage == null) {
            load();
        }
        JMatrix copyReference = originalImage;
        if (lowMemoryMode) {
            unload();
        }

        return copyReference;
    }

    public int getWidth() {
        loadingSequence();
        JMatrix copyReference = originalImage;
        if (lowMemoryMode) {
            unload();
        }
        return copyReference.width();
    }

    public int numChannels() {
        return channels;
    }

    public int getHeight() {
        loadingSequence();
        JMatrix copyReference = originalImage;
        if (lowMemoryMode) {
            unload();
        }
        return copyReference.height();
    }

    private void loadingSequence() {
        if (xData == null) {
            // CSV images can't be reloaded
            if (loadedFromCSV) {
                xData = originalImage;
            } else {
                load();
            }
            applyTransforms();
        }
    }

    private void unload() {
        originalImage = xData = null;
    }

    // Return true if an image is grayscale, CURRENTLY BUGGY
    // private boolean grayscaleCheck(BufferedImage img) {
    //     int width = img.getWidth();
    //     int height = img.getHeight();
    //     Set<Integer> uniqueColors = new HashSet<>();

    //     for (int y = 0; y < height; y++) { 
    //         for (int x = 0; x < width; x++) { 
    //             int argb = img.getRGB(x, y);
    //             int red   = (argb >> 16) & 0xFF;
    //             int green = (argb >> 8)  & 0xFF;
    //             int blue  = (argb)       & 0xFF;

    //             // Store the unique grayscale intensity
    //             uniqueColors.add(red);

    //             // If we find non-gray pixels, exit early
    //             if (red != green || green != blue) {
    //                 return false;
    //             }
    //         }
    //     }
    //     return true;
    // }


    // Load an image as grayscale: (1, height, width)
    private JMatrix loadGrayscaleImage(BufferedImage img) {
        int width = img.getWidth();
        int height = img.getHeight();
        JMatrix grayscaleArray = JMatrix.zeros(1, 1, height, width); 

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int argb = img.getRGB(x, y);
                int gray = argb & 0xFF; 
                grayscaleArray.set(0, 0, y, x, gray);
            }
        }
        return grayscaleArray;
    }

    // Load an image as RGB: (3, height, width)
    private JMatrix loadRGBImage(BufferedImage img) {
        int width = img.getWidth();
        int height = img.getHeight();
        JMatrix rgbArray = JMatrix.zeros(1, 3, height, width); 

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int argb = img.getRGB(x, y);
                rgbArray.set(0, 0, y, x, (argb >> 16) & 0xFF); // Red
                rgbArray.set(0, 1, y, x, (argb >> 8)  & 0xFF); // Green
                rgbArray.set(0, 2, y, x, (argb)       & 0xFF); // Blue
            }
        }
        return rgbArray;
    }
}
    

