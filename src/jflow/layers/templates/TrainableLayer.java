package jflow.layers.templates;

import jflow.data.JMatrix;

public abstract class TrainableLayer extends ShapeAlteringLayer {

    public TrainableLayer(String type) {
        super(type);
    }

    public abstract JMatrix[] getParameterGradients();

    public abstract void updateParameters(JMatrix[] parameterUpdates);
    
    public abstract JMatrix[] getParameters();

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
        // Ensure dX is properly named
        if (hasDX) {
            debugData[0] = getGradient().label("dInput");
        }

        for (int i = 0; i < numParameterGradients; i++) {
            debugData[i + dXoffset] = parameterGradients[i];
        }

        return debugData;
    }
}
