package jflow.layers.templates;

import jflow.data.JMatrix;

public abstract class TrainableLayer extends ShapeAlteringLayer{
    public TrainableLayer(String type) {
        super(type);
    }

    public abstract JMatrix[] getParameterGradients();

    public abstract void updateParameters(JMatrix[] parameterUpdates);

    public abstract JMatrix[] getWeights();

    @Override
    protected JMatrix[] forwardDebugData() {
        if (getOutput() == null) {
            return null;
        }
        return new JMatrix[]{getOutput().setName("output")};
    }

    @Override 
    protected JMatrix[] backwardDebugData() {
        JMatrix[] parameterGradients = getParameterGradients();
        int numParameterGradients = parameterGradients.length;
        boolean hasDX = true;
        if (getGradient() == null) {
            hasDX = false;
        }

        JMatrix[] debugData = new JMatrix[numParameterGradients + ((hasDX) ? 1 : 0)];

        for (int i = 0; i < numParameterGradients; i++) {
            debugData[i] = parameterGradients[i];
        }
        // Ensure dX is properly named
        if (hasDX) {
            debugData[numParameterGradients] = getGradient().setName("dOutput");
        }
        return debugData;
    }
}
