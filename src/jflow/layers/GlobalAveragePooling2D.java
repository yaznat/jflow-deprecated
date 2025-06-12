package jflow.layers;

import java.util.stream.IntStream;

import jflow.data.JMatrix;
import jflow.layers.templates.ShapeAlteringLayer;

public class GlobalAveragePooling2D extends ShapeAlteringLayer{
    private int batchSize;
    int channels;
    int height;
    int width;


    public GlobalAveragePooling2D() {
        super("gap_2d");
    }


    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        batchSize = input.length();
        channels = input.channels();
        height = input.height();
        width = input.width();
        int imageDim = height * width;
        JMatrix averaged = JMatrix.zeros(batchSize, channels, 1, 1);
        
        // For each batch item and channel
        IntStream.range(0, batchSize).parallel().forEach(n -> {
            for (int c = 0; c < channels; c++) {
                float sum = 0f;
                // Average over height and width dimensions
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        sum += input.get(n, c, h, w);
                    }
                }
                averaged.set(n, c, 0, 0, sum / imageDim);
            }
        });

        return trackOutput(averaged, training);
    }

    @Override
    public JMatrix backward(JMatrix input) {
        int imageDim = height * width;
        JMatrix expanded = JMatrix.zeros(batchSize, channels, height, width);
        
        // For each batch item and channel
        IntStream.range(0, batchSize).parallel().forEach(n -> {
            for (int c = 0; c < channels; c++) {
                // Get the gradient for this batch item and channel
                float gradient = input.get(n, c, 0, 0) / imageDim;
                
                // Distribute it evenly across all spatial dimensions
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        expanded.set(n, c, h, w, gradient);
                    }
                }
            }
        });

        return trackGradient(expanded);
    }


    @Override
    public int[] outputShape() {
        // Channels of the previous layer
        return new int[]{-1, getPreviousLayer().outputShape()[1]};
    }
}
