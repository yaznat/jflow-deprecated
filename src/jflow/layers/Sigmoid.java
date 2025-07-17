package jflow.layers;

import java.util.stream.IntStream;

import jflow.data.JMatrix;
import jflow.layers.templates.ShapePreservingLayer;

public class Sigmoid extends ShapePreservingLayer{

    /**
     * The Sigmoid activation.
     * 
     * <p><b>Do not instantiate directly.</b> Use the static builder method:
     * {@code import static jflow.model.builder.*;}
     * and call {@code Sigmoid()} instead of {@code new Sigmoid()}.
     */
    public Sigmoid() {
        super("sigmoid");
    }

    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        JMatrix Z = input.zerosLike();

        int size = input.size();
        // Apply sigmoid: 1 / (1 + e ^(-x)) 
        IntStream.range(0, size).parallel().forEach(i -> {
            Z.set(i, 1.0 / (1.0 + Math.exp(-input.get(i))));
        });

        return trackOutput(Z, training);
    }

    @Override
    public JMatrix backward(JMatrix gradient) {
        JMatrix output = getOutput();
        JMatrix dSigmoid = output.zerosLike();
        int size = output.size();

        if (getNextLayer() == null) {
            /*
             * This is the output layer.
             * Use binary cross entropy: predicted - actual
             */ 
            return trackGradient(output.subtract(gradient));
        }

        IntStream.range(0, size).parallel().forEach(i -> {
            double sigValue = output.get(i);
            // Compute sigmoid derivative: σ(x) * (1 - σ(x))
            dSigmoid.set(i, sigValue * (1.0 - sigValue));
        });
        return trackGradient(dSigmoid.multiply(gradient));
    }
}
