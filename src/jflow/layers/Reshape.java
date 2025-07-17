package jflow.layers;

import jflow.data.JMatrix;
import jflow.layers.templates.ShapeAlteringLayer;

public class Reshape extends ShapeAlteringLayer{
    private int newLength;
    private int newChannels;
    private int newHeight;
    private int newWidth;
    private int oldChannels;
    private int oldHeight;
    private int oldWidth;
    private int oldLength;


    /**
     * The Reshape layer.
     * 
     * <p><b>Do not instantiate directly.</b> Use the static builder method:
     * {@code import static jflow.model.builder.*;}
     * and call {@code Reshape(...)} instead of {@code new Reshape(...)}.
     */
    public Reshape(int length, int channels, int height, int width) {
        super("reshape");
        this.newLength = length;
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

        int newLength = (this.newLength == -1) ? input.length() : this.newLength;

        return trackOutput(input.reshape(newLength, newChannels, newHeight, newWidth), training);
    }

    

    @Override
    public JMatrix backward(JMatrix input) {
        return trackGradient(input.reshape(oldLength, oldChannels, oldHeight, oldWidth));
    }

    @Override
    public int[] outputShape() {
        return new int[] {1, newChannels, newHeight, newWidth};
    }
}
