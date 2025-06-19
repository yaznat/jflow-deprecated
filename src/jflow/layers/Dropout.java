package jflow.layers;

import java.util.concurrent.ThreadLocalRandom;
import java.util.stream.IntStream;

import jflow.data.JMatrix;
import jflow.layers.templates.ShapePreservingLayer;

public class Dropout extends ShapePreservingLayer{
    private double dropoutRate;
    private JMatrix dropoutMask;

    public Dropout(double dropoutRate) {
        super("dropout");
        this.dropoutRate = dropoutRate;
    }
    // Reset the dropout mask
    private void newDropoutMask(int inputSize, int outputSize) {
        dropoutMask = JMatrix.zeros(inputSize, outputSize, 1, 1);
        IntStream.range(0, inputSize).parallel().forEach(i -> {
            for (int j = 0; j < outputSize; j++) {
                dropoutMask.set(i * outputSize + j, (
                    ThreadLocalRandom.current().nextDouble() < dropoutRate) ? 0 : 1);
            }
        });
    }

    public JMatrix applyMask(JMatrix input, boolean training) {
        if (training) {
            /*
             * Multiply with mask and scale nonzero 
             * results to keep sum about the same.
             */
            return input.multiply(dropoutMask).multiply(1.0 / (1.0 - dropoutRate));
        } 
        return input.copy();
    }

    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        // Determine the type of mask to use
        boolean useFlat = getPreviousShapeInfluencer() instanceof Dense;

        if (useFlat) {
            newDropoutMask(input.length(), input.channels() * input.height() * input.width());
        } else {
            newDropoutMask(input.length(), input.channels());
        }

        return trackOutput(applyMask(input, training), training);
    }
    @Override
    public JMatrix backward(JMatrix input) {
        // Similar to forward
        return trackGradient(applyMask(input, true));
    }

}