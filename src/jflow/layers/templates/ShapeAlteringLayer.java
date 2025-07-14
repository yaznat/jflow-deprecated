package jflow.layers.templates;

import jflow.data.JMatrix;
import jflow.model.Layer;

public abstract class ShapeAlteringLayer extends Layer{
    public ShapeAlteringLayer(String type) {
        super(type, true);
    }

    public abstract JMatrix forward(JMatrix input, boolean training);

    public abstract JMatrix backward(JMatrix input);

    @Override
    protected JMatrix[] forwardDebugData() {
        if (getOutput() == null) {
            return null;
        }
        return new JMatrix[]{getOutput().label("activation")};
    }

    @Override
    protected JMatrix[] backwardDebugData() {
        if (getGradient() == null) {
            return null;
        }
        return new JMatrix[]{getGradient().label("dActivation")};
    }
}
