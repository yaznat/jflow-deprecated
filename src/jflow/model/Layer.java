package jflow.model;


import jflow.data.JMatrix;
import jflow.utils.Callbacks;

public abstract class Layer {
    // Layer graph
    private Layer previousLayer;
    private Layer nextLayer;
    private Layer enclosingLayer;

    // Data
    private JMatrix output;
    private JMatrix gradient;
    private JMatrix lastInput;

    // Shape and configuration
    private int[] inputShape;
    private final String type;
    private int numTrainableParameters;
    private int IDnum;
    private final boolean isShapeInfluencer;
    private boolean gradientStorageDisabled = false;
        

    public abstract JMatrix forward(JMatrix input, boolean training);

    public abstract JMatrix backward(JMatrix input);

    public abstract int[] outputShape();

    protected Layer(String type, boolean isShapeInfluencer) {
        this.type = type;
        this.isShapeInfluencer = isShapeInfluencer;
    }

    protected Layer getNextLayer() {
        return nextLayer;
    }
    protected Layer getPreviousLayer() {
        return previousLayer;
    }

    protected void setNextLayer(Layer nextLayer) {
        this.nextLayer = nextLayer;
    }

    public void setPreviousLayer(Layer previousLayer) {
        this.previousLayer = previousLayer;
    }

    protected void setEnclosingLayer(Layer enclosingLayer) {
        this.enclosingLayer = enclosingLayer;
    }

    protected Layer getEnclosingLayer() {
        return enclosingLayer;
    }
    protected boolean isInternal() {
        return enclosingLayer != null;
    }

    // Some layers will override this
    protected void build(int IDnum) {
        this.IDnum = IDnum;
    }


    public JMatrix getOutput() {
        return output;
    }
    public JMatrix getGradient() {
        return gradient;
    }

    protected JMatrix trackOutput(JMatrix output, boolean training) {
        // Last layer must store output
        if (training || getNextLayer() == null) {
            this.output = output;
        } else {
            // Ensure memory is freed
            this.output = null;
        }
        return output;
    }
    
    protected JMatrix trackGradient(JMatrix gradient) {
        if (gradientStorageDisabled) {
            // Ensure memory is freed
            this.gradient = null;
        } else {
            this.gradient = gradient;
        }
        clearInputCache(); // Not needed after backward() is complete

        return gradient;
    }

    protected void cacheInput(JMatrix input, boolean training) {
        if (training) {
            lastInput = input;
        }
    }

    private void clearInputCache() {
        lastInput = null;
    }


    protected JMatrix getLastInput() {
        return lastInput;
    }

    protected void setInputShape(int[] inputShape) {
        this.inputShape = inputShape;
    }

    protected void disableGradientStorage() {
        gradientStorageDisabled = true;
    }
    

    protected int[] getInputShape() {
        if (inputShape == null) {
            return getPreviousLayer().outputShape();
        }
        return inputShape;
    }

    // True if this layer changes output shape (e.g., Dense, Conv2D, Flatten)
    protected boolean isShapeInfluencer() {
        return isShapeInfluencer;
    }

    protected Layer getPreviousShapeInfluencer() {
        Layer prevLayer = getPreviousLayer();
        while (!prevLayer.isShapeInfluencer()) {
            prevLayer = prevLayer.getPreviousLayer();
        }
        return prevLayer;
    }

    protected boolean trainable() {
        return numTrainableParameters() != 0;
    }

    protected void setIDNum(int IDnum) {
        this.IDnum = IDnum;
    }
    protected int numTrainableParameters() {
        return numTrainableParameters;
    }

    // For proper layer build after initialization
    protected void setNumTrainableParameters(int numTrainableParameters) {
        this.numTrainableParameters = numTrainableParameters;
    }

    protected String getType() {
        return type;
    }
    protected String getName() {
        return type + "_" + IDnum;
    }

    protected int getLayerIndex() {
        if (getPreviousLayer() == null) {
            return 0;
        } else if (getPreviousLayer() instanceof FunctionalLayer) {
            return 0 + getPreviousLayer().getLayerIndex();
        }   else {
            return 1 + getPreviousLayer().getLayerIndex();
        }
    }

    // Count the number of layers in the linked list of a certain type.
    protected int getLayerTypeCount(String layerType) {
        int count = 1;
        Layer prevLayer = getPreviousLayer();
        while (prevLayer != null) {
            if (prevLayer.getType().equals(layerType)) {
                count++;
            }
            prevLayer = prevLayer.getPreviousLayer();
        }
        return count;
    }

    protected JMatrix[] forwardDebugData() {
        if (getOutput() == null) {
            return null;
        }
        return new JMatrix[]{getOutput().label("activation")};
    }

    protected JMatrix[] backwardDebugData() {
        if (getGradient() == null) {
            return null;
        }
        return new JMatrix[]{getGradient().label("dActivation")};
    }

    public void printForwardDebug() {
        JMatrix[] debugData = forwardDebugData();
        if (!(debugData == null)) {
            String title = getName();
            Callbacks.printStats(title, debugData);
        }
    }

    public void printBackwardDebug() {
        JMatrix[] debugData = backwardDebugData();
        if (!(debugData == null)) {
            String title = getName();
            Callbacks.printStats(title, debugData);
        }
    }
}
