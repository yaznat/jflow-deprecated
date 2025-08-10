package jflow.layers;


import jflow.data.JMatrix;
import jflow.layers.templates.ShapeAlteringLayer;

public class Flatten extends ShapeAlteringLayer{
    private int[] lastInputShape;

    /**
     * The Flatten layer.
     * 
     * <p><b>Do not instantiate directly.</b> Use the static builder method:
     * {@code import static jflow.model.builder.*;}
     * and call {@code Flatten()} instead of {@code new Flatten()}.
     */
    public Flatten() {
        super("flatten");
    }

    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        lastInputShape = input.shape();

        int length = input.shape(0);
        int features = input.shape(1) * input.shape(2) * input.shape(3);

        JMatrix output = input.reshape(length, features, 1, 1);

        return trackOutput(output, training);
    }

    @Override
    public JMatrix backward(JMatrix input) {
        JMatrix gradient = input.reshape(lastInputShape);

        return trackGradient(gradient);
    }
}
