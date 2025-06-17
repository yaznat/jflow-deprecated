package jflow.layers.templates;

import jflow.data.JMatrix;
import jflow.model.Layer;

public abstract class ShapePreservingLayer extends Layer{

    public ShapePreservingLayer(String type) {
        super(type, false);
    }

    public abstract JMatrix forward(JMatrix input, boolean training);

    public abstract JMatrix backward(JMatrix input);

    @Override
    public int[] outputShape() {
        return getPreviousLayer().outputShape();
    }    

    @Override
    protected JMatrix[] forwardDebugData() {
        if (getOutput() == null) {
            return null;
        }
        return new JMatrix[]{getOutput().setName("activation")};
    }

    @Override
    protected JMatrix[] backwardDebugData() {
        if (getGradient() == null) {
            return null;
        }
        return new JMatrix[]{getGradient().setName("dActivation")};
    }
}
