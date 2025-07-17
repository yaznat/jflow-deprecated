package jflow.layers;

import java.util.stream.IntStream;

import jflow.data.JMatrix;
import jflow.layers.templates.ShapeAlteringLayer;

public class Upsampling2D extends ShapeAlteringLayer{
    private int scaleFactor;

    /**
     * The Upsampling2D layer.
     * 
     * <p><b>Do not instantiate directly.</b> Use the static builder method:
     * {@code import static jflow.model.builder.*;}
     * and call {@code Upsampling2D(...)} instead of {@code new Upsampling2D(...)}.
     */
    public Upsampling2D(int scaleFactor) {
        super("up_sampling_2d");
        this.scaleFactor = scaleFactor;
    }

    @Override
    // Expand input by scale factor
    public JMatrix forward(JMatrix input, boolean training) {
        int numImages = input.length();
        int channels = input.channels();
        int height = input.height();
        int width = input.width();

        float[] inputMatrix = input.unwrap();
        
        int newHeight = height * scaleFactor;
        int newWidth = width * scaleFactor;

        JMatrix output = JMatrix.zeros(numImages, channels, newHeight, newWidth);

        IntStream.range(0, numImages * channels).parallel().forEach(index -> {
            int i = index / channels;  
            int c = index % channels;  

            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    // Map input value to square region of output values
                    float value = inputMatrix[i * channels * height * width + c * height * width + h * width + w];

                    for (int a = 0; a < scaleFactor; a++) {
                        for (int b = 0; b < scaleFactor; b++) {
                            int newH = h * scaleFactor + a;
                            int newW = w * scaleFactor + b;
                            output.set(i * channels * newHeight * newWidth + c * 
                                newHeight * newWidth + newH * newWidth + newW, value);
                        }
                    }
                }
            }
        });

        return trackOutput(output, training);
    }

    @Override
    // Shrink input gradient by scale factor, summing regions
    public JMatrix backward(JMatrix input) {
        int numImages = input.length();
        int channels = input.channels();
        int height = input.height();
        int width = input.width();
        float[] inputMatrix = input.unwrap(); 
        
        int newHeight = height / scaleFactor;
        int newWidth = width / scaleFactor;

        JMatrix gradient = JMatrix.zeros(numImages, channels, newHeight, newWidth);
    
        IntStream.range(0, numImages * channels).parallel().forEach(index -> {
            int i = index / channels;  
            int c = index % channels;  
    
            for (int newH = 0; newH < newHeight; newH++) {
                for (int newW = 0; newW < newWidth; newW++) {
                    int outputIndex = i * channels * newHeight * newWidth + c * newHeight * newWidth + newH * newWidth + newW;
    
                    float sum = 0; 
                    for (int a = 0; a < scaleFactor; a++) {
                        for (int b = 0; b < scaleFactor; b++) {
                            int h = newH * scaleFactor + a;
                            int w = newW * scaleFactor + b;
                            int inputIndex = i * channels * height * width + c * height * width + h * width + w;
                            sum += inputMatrix[inputIndex];  // Sum gradients from upscaled region
                        }
                    }
                    gradient.set(outputIndex, sum);
                }
            }
        });
    
        return trackGradient(gradient);

    }


    @Override
    public int[] outputShape() {
        int[] shape = getPreviousLayer().outputShape().clone();
        shape[2] *= scaleFactor;
        shape[3] *= scaleFactor;
        return shape;
    }
}
