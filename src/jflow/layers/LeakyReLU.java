package jflow.layers;


import java.util.stream.IntStream;

import jflow.data.JMatrix;
import jflow.layers.templates.ShapePreservingLayer;

public class LeakyReLU extends ShapePreservingLayer{
    private final float alpha;

    /**
     * The LeakyReLU activation.
     * 
     * <p><b>Do not instantiate directly.</b> Use the static builder method:
     * {@code import static jflow.model.builder.*;}
     * and call {@code LeakyReLU(...)} instead of {@code new LeakyReLU(...)}.
     */
    public LeakyReLU(double alpha) {
        super("leaky_re_lu");
        this.alpha = (float)alpha;
    }

    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        int size = input.size();
        JMatrix Z = input.zerosLike();

        IntStream.range(0, size).parallel().forEach(i -> {
            Z.set(i, (input.get(i) > 0) ? input.get(i) : alpha * input.get(i));
        });

        return trackOutput(Z, training);
    }

    @Override
    public JMatrix backward(JMatrix gradient) {
        int size = gradient.size();
        JMatrix output = getOutput();
        JMatrix dZ = output.zerosLike();

        IntStream.range(0, size).parallel().forEach(i -> {
            dZ.set(i, (output.get(i) > 0) ? gradient.get(i) : alpha * gradient.get(i));
        });

        return trackGradient(dZ);
    }
}
