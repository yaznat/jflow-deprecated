package jflow.layers;


import jflow.data.JMatrix;
import jflow.layers.templates.ShapeAlteringLayer;

public class Flatten extends ShapeAlteringLayer{

    public Flatten() {
        super("flatten");
    }
    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        JMatrix output = input.reshape(input.length(), input.channels() * 
            input.height() * input.width(), 1, 1);
        return trackOutput(output, training);
    }

    @Override
    public JMatrix backward(JMatrix input) {
        JMatrix gradient = input.reshape(getPreviousLayer().outputShape());
        return trackGradient(gradient);
    }
    
    @Override
    public int[] outputShape() {
        int[] prevOutputShape = getPreviousLayer().outputShape();
        int flattenedSize = prevOutputShape[1] * prevOutputShape[2] * prevOutputShape[3];

        return new int[]{1, flattenedSize};
    }
    
}
