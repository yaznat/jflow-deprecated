package jflow.layers.templates;

import jflow.data.JMatrix;

public abstract class TrainableLayer extends ShapeAlteringLayer {
    private boolean built = false;

    public TrainableLayer(String type) {
        super(type);
    }

    protected abstract void build(int[] inputShape);
    protected abstract JMatrix trainableForwardPass(JMatrix input, boolean training);
    protected abstract JMatrix trainableBackwardPass(JMatrix gradient);
    public abstract JMatrix[] getParameterGradients();
    public abstract void updateParameters(JMatrix[] parameterUpdates);
    public abstract JMatrix[] getParameters();


    /**
     * @return True if this layer is built and parameters are initialized, false otherwise.
     */
    public boolean isBuilt() {
        return built;
    }

    @Override
    public JMatrix forward(JMatrix input, boolean training) {
        // Avoid handling build state in child classes
        if (!built) {
            int[] inputShape = input.shape();
            int[] userSpecifiedShape = getInputShape();
            if (userSpecifiedShape != null) inputShape = userSpecifiedShape;
            build(inputShape);
            built = true;
        }
        return trainableForwardPass(input, training);
    }

    @Override
    public JMatrix backward(JMatrix gradient) {
        return trainableBackwardPass(gradient);
    }

    @Override
    protected JMatrix[] forwardDebugData() {
        JMatrix[] parameters = getParameters();
        int numParameters = parameters.length;
        boolean hasOutput = getOutput() != null;
        
        int outputOffset = ((hasOutput) ? 1 : 0);
        JMatrix[] debugData = new JMatrix[numParameters + outputOffset];
        // Ensure output is properly named
        if (hasOutput) {
            debugData[0] = getOutput().label("output");
        }

        for (int i = 0; i < numParameters; i++) {
            debugData[i + outputOffset] = parameters[i];
        }

        return debugData;
    }

    @Override 
    protected JMatrix[] backwardDebugData() {
        JMatrix[] parameterGradients = getParameterGradients();
        int numParameterGradients = parameterGradients.length;
        boolean hasDX = getGradient() != null;
        
        int dXoffset = ((hasDX) ? 1 : 0);
        JMatrix[] debugData = new JMatrix[numParameterGradients + dXoffset];
        // Ensure dInput is properly named
        if (hasDX) {
            debugData[0] = getGradient().label("dInput");
        }

        for (int i = 0; i < numParameterGradients; i++) {
            debugData[i + dXoffset] = parameterGradients[i];
        }

        return debugData;
    }
}
