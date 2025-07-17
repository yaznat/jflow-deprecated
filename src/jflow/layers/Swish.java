package jflow.layers;

import java.util.stream.IntStream;
import jflow.data.JMatrix;
import jflow.layers.templates.ShapePreservingLayer;

public class Swish extends ShapePreservingLayer {

    /**
     * The Swish activation.
     * 
     * <p><b>Do not instantiate directly.</b> Use the static builder method:
     * {@code import static jflow.model.builder.*;}
     * and call {@code Swish()} instead of {@code new Swish()}.
     */
    public Swish() {
        super("swish");
    }
    
    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        cacheInput(input, training);
        int size = input.size();
        JMatrix output = input.zerosLike();
        
        IntStream.range(0, size).parallel().forEach(i -> {
            double x = input.get(i);
            double sigmoid = 1.0 / (1.0 + Math.exp(-x));
            output.set(i, x * sigmoid);
        });
        
        return trackOutput(output, training);
    }
    
    @Override
    public JMatrix backward(JMatrix gradient) {
        int size = gradient.size();
        JMatrix input = getLastInput(); // Use original input x
        JMatrix dZ = input.zerosLike();
        
        IntStream.range(0, size).parallel().forEach(i -> {
            double x = input.get(i);
            double sigmoid = 1.0 / (1.0 + Math.exp(-x));
            
            // Derivative of Swish: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            double derivative = sigmoid + x * sigmoid * (1 - sigmoid);
            
            dZ.set(i, gradient.get(i) * derivative);
        });
        
        return trackGradient(dZ);
    }
}