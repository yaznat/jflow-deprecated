package jflow.layers;

import java.util.Arrays;

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

    private String mode;

    private static final String[] modeDict = new String[] {
        "merge_batch_seq",
        "split_batch_seq",
    };


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

    /**
     * The Reshape layer.
     * 
     * <p><b>Do not instantiate directly.</b> Use the static builder method:
     * {@code import static jflow.model.builder.*;}
     * and call {@code Reshape(...)} instead of {@code new Reshape(...)}.
     */
    public Reshape(String mode) {
        super("reshape");
        if(!Arrays.stream(modeDict).anyMatch(item -> item.equals(mode))) {
            throw new IllegalArgumentException(
                "Unknown reshape mode: " + mode + 
                "\nView the jflow.model.Builder.Reshape(...) JavaDoc for a list of supported modes."
            ); 
        }
        this.mode = mode;
    }

    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        this.oldLength = input.length();
        this.oldChannels = input.channels();
        this.oldHeight = input.height();
        this.oldWidth = input.width();

        switch (mode) {
            case "merge_batch_seq":
                return trackOutput(input.reshape(oldLength * oldChannels, oldHeight, 1, 1), training);
            case "split_batch_seq":
                // Assume seqLen from embedding layer
                int seqLen = getLayerList().getFirst().outputShape()[1];
                return trackOutput(input.reshape(oldLength / seqLen, seqLen, oldChannels, oldHeight), training); 
            default:
                int newLength = (this.newLength == -1) ? input.length() : this.newLength;

                return trackOutput(input.reshape(newLength, newChannels, newHeight, newWidth), training);
        }

        
    }

    @Override
    public JMatrix backward(JMatrix input) {
        return trackGradient(input.reshape(oldLength, oldChannels, oldHeight, oldWidth));
    }
}
