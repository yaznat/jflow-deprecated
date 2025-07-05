package jflow.layers;

import jflow.data.JMatrix;
import jflow.layers.templates.ShapeAlteringLayer;

public class Reshape extends ShapeAlteringLayer{
    private int newChannels;
    private int newHeight;
    private int newWidth;
    private int oldChannels;
    private int oldHeight;
    private int oldWidth;
    private int oldLength;


    public Reshape(int channels, int height, int width) {
        super("reshape");
        this.newChannels = channels;
        this.newHeight = height;
        this.newWidth = width;
    }

    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        this.oldLength = input.length();
        this.oldChannels = input.channels();
        this.oldHeight = input.height();
        this.oldWidth = input.width();

        // Account for dense layers being transposed
        if (getPreviousShapeInfluencer() instanceof Dense) {
            input = input.T();
        }

        return trackOutput(input.reshape(input.length(), newChannels, newHeight, newWidth), training);
    }

    

    @Override
    public JMatrix backward(JMatrix input) {
        return trackGradient(input.reshape(oldLength, oldChannels, oldHeight, oldWidth));
    }

    @Override
    public int[] outputShape() {
        return new int[] {-1, newChannels, newHeight, newWidth};
    }
}
