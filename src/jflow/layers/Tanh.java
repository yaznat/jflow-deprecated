package jflow.layers;

import java.util.stream.IntStream;
import jflow.data.JMatrix;
import jflow.layers.templates.ShapePreservingLayer;

public class Tanh extends ShapePreservingLayer{

    /**
     * The Tanh activation.
     * 
     * <p><b>Do not instantiate directly.</b> Use the static builder method:
     * {@code import static jflow.model.builder.*;}
     * and call {@code Tanh()} instead of {@code new Tanh()}.
     */
    public Tanh() {
        super("tanh");
    }

    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        JMatrix output = input.zerosLike();

        int size = input.size();
        IntStream.range(0, size).parallel().forEach(i -> {
            output.set(i, Math.tanh(input.get(i)));
        });
        return trackOutput(output, training);
    }

    @Override
    public JMatrix backward(JMatrix gradient) {
        JMatrix output = getOutput();
        JMatrix dZ = output.zerosLike();
        int size = output.size();

        IntStream.range(0, size).parallel().forEach(i -> {
            double tanhVal = Math.tanh(output.get(i));  
            double dTanh = 1 - tanhVal * tanhVal;  
            dZ.set(i, gradient.get(i) * dTanh);
        });
        return trackGradient(dZ);
    }    
}
