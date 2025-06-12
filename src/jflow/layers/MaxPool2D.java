package jflow.layers;

import java.util.stream.IntStream;

import jflow.data.JMatrix;
import jflow.layers.templates.ShapeAlteringLayer;

public class MaxPool2D extends ShapeAlteringLayer {
    private int poolSize;
    private int stride;
    private int numImages;
    private int channels;
    private int imageHeight;
    private int imageWidth;
    private int outputHeight;
    private int outputWidth;

    private JMatrix lastInput;

    public MaxPool2D(int poolSize, int stride) {
        super("max_pool_2d");
        this.poolSize = poolSize;
        this.stride = stride;
    }

    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        this.imageHeight = input.height();
        this.imageWidth = input.width();
        this.numImages = input.length();
        this.channels = input.channels();

        lastInput = input;

        // Calculate output dimensions
        outputHeight = (imageHeight - poolSize) / stride + 1;
        outputWidth = (imageWidth - poolSize) / stride + 1;
 
        JMatrix output = JMatrix.zeros(numImages, channels, outputHeight, outputWidth);


        // Perform max pooling
        IntStream.range(0, numImages).parallel().forEach(i -> {
            for (int c = 0; c < channels; c++) {
                int inputOffset = i * channels * imageHeight * imageWidth + c * imageHeight * imageWidth;
                int outputOffset = i * channels * outputHeight * outputWidth + c * outputHeight * outputWidth;

                maxPool2D(input, inputOffset, output, outputOffset);
            }
        });

        return trackOutput(output, training); 
    }

    @Override
    public JMatrix backward(JMatrix dOutput) {
        JMatrix gradient = lastInput.zerosLike();

        // Calculate maxpool gradients
        IntStream.range(0, numImages).parallel().forEach(i -> {
            for (int c = 0; c < channels; c++) {
                int inputOffset = i * channels * imageHeight * imageWidth + c * imageHeight * imageWidth;
                int outputOffset = i * channels * outputHeight * outputWidth + c * outputHeight * outputWidth;
        
                backpropMaxPool2D(dOutput, outputOffset, gradient, inputOffset, lastInput); 
            }
        });

        return trackGradient(gradient);
    }

    // Perform max pooling on one image
    private void maxPool2D(JMatrix input, int inputOffset, JMatrix output, int outputOffset) {
        for (int sX = 0; sX < outputHeight; sX++) {
            for (int sY = 0; sY < outputWidth; sY++) {
                float max = Float.NEGATIVE_INFINITY;

                for (int poolX = 0; poolX < poolSize; poolX++) {
                    for (int poolY = 0; poolY < poolSize; poolY++) {
                        int x = sX * stride + poolX;
                        int y = sY * stride + poolY;
                        int idx = inputOffset + x * imageWidth + y;

                        max = Math.max(max, input.get(idx));
                    }
                }
                output.set(outputOffset + sX * outputWidth + sY, max);
            }
        }
    }
    // Calculate the max pool gradient for one image
    private void backpropMaxPool2D(JMatrix dOutput, int dOutputOffset, JMatrix gradient, int gradientOffset, JMatrix lastInput) {
        for (int sX = 0; sX < outputHeight; sX++) {
            for (int sY = 0; sY < outputWidth; sY++) {
                float max = Float.NEGATIVE_INFINITY;
                int maxX = 0, maxY = 0;
    
                // Find max position
                for (int poolX = 0; poolX < poolSize; poolX++) {
                    for (int poolY = 0; poolY < poolSize; poolY++) {
                        int x = sX * stride + poolX;
                        int y = sY * stride + poolY;
                        int inputIdx = gradientOffset + x * imageWidth + y;
    
                        if (lastInput.get(inputIdx) > max) { 
                            max = lastInput.get(inputIdx);
                            maxX = x;
                            maxY = y;
                        }
                    }
                }
    
                // Pass back the gradient only to the max position
                int maxIdx = gradientOffset + maxX * imageWidth + maxY;
                int dOutputIdx = dOutputOffset + sX * outputWidth + sY;
                gradient.set(maxIdx, gradient.get(maxIdx) + dOutput.get(dOutputIdx));
            }
        }
    }

    @Override
    public int[] outputShape() {
        int[] outputShape = null;
        int[] prev = getPreviousLayer().outputShape();

        if (getOutput() != null) {
            outputShape = getOutput().shape();
        } else {
            int oldHeight = prev[2];
            int oldWidth = prev[3];

            int outputHeight = (oldHeight - poolSize) / stride + 1;
            int outputWidth = (oldWidth - poolSize) / stride + 1;

            return new int[]{-1, prev[1], outputHeight, outputWidth};
            
        }
        return outputShape;
    }
}
