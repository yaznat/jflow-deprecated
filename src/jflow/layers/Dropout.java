package jflow.layers;

import java.util.Random;
import java.util.stream.IntStream;

import jflow.data.JMatrix;
import jflow.layers.templates.ShapePreservingLayer;

public class Dropout extends ShapePreservingLayer{
    private final double dropoutRate;
    private JMatrix dropoutMask;

    /**
     * The Dropout layer.
     * 
     * <p><b>Do not instantiate directly.</b> Use the static builder method:
     * {@code import static jflow.model.builder.*;}
     * and call {@code Dropout(...)} instead of {@code new Dropout(...)}.
     */
    public Dropout(double dropoutRate) {
        super("dropout");
        if (dropoutRate < 0.0 || dropoutRate >= 1.0) {
            throw new IllegalArgumentException("dropoutRate must be in [0.0, 1.0).");
        }        
        this.dropoutRate = dropoutRate;
    }

    private void generateDropoutMask(int length, int features) {
        dropoutMask = JMatrix.zeros(length, features, 1, 1);
        
        int size = length * features;
        int chunkSize = 1024;

        IntStream.range(0, (size + chunkSize - 1) / chunkSize).parallel().forEach(chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, size);

            // Hash-based unique seed per chunk, using JMatrix.seed for reproducability
            long localSeed = JMatrix.currentSeed() ^ Long.rotateLeft(chunk * 0x9E3779B97F4A7C15L, 17);
            Random rng = new Random(localSeed);

            for (int i = start; i < end; i++) {
                dropoutMask.set(i, rng.nextDouble() < dropoutRate ? 0 : 1);
            }
        });
    }

    private JMatrix applyMask(JMatrix input, boolean training) {
        if (training) {
            /*
             * Multiply with mask and scale nonzero 
             * results to keep sum about the same.
             */
            return input.multiply(dropoutMask).multiply(1.0 / (1.0 - dropoutRate));
        } 
        // Avoid dropout during inference
        return input.copy();
    }

    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        boolean convolutional = getPreviousShapeInfluencer() instanceof Conv2D;

        if (convolutional) {
            // (N, C, 1, 1) -- Spatial dropout
            generateDropoutMask(input.shape(0), input.shape(1));
        } else {
            // (N, F) -- elementwise broadcast
            generateDropoutMask(input.shape(0), input.shape(1) * input.shape(2) * input.shape(3));
        }

        return trackOutput(applyMask(input, training), training);
    }
    @Override
    public JMatrix backward(JMatrix input) {
        // Similar to forward
        return trackGradient(applyMask(input, true));
    }
}