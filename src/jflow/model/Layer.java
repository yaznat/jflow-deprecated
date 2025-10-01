package jflow.model;


import jflow.data.JMatrix;
import jflow.utils.Callbacks;

public abstract class Layer {
    // Layer graph
    private LayerList layers;
    private Layer lastLayer; // Cached for performance

    // Data
    private JMatrix output;
    private JMatrix gradient;
    private JMatrix lastInput;

    // Name and configuration
    private final String type;
    private String name;
    private int numTrainableParameters;
    private final boolean isShapeInfluencer;
    private boolean gradientStorageDisabled = false;
    private boolean debugEnabled;
    private int[] inputShape;
    private int[] outputShape;

    public abstract JMatrix forward(JMatrix input, boolean training);

    public abstract JMatrix backward(JMatrix input);

    protected Layer(String type, boolean isShapeInfluencer) {
        this.type = type;
        this.isShapeInfluencer = isShapeInfluencer;
    }

    protected void link(LayerList layers) {
        this.layers = layers;
        lastLayer = layers.getLast();
    }

    protected LayerList getLayerList() {
        return layers;
    }

    protected void setName(String name) {
        this.name = name;
    }

    protected String getType() {
        return type;
    }
    public String getName() {
        return name;
    }

    protected void setInputShape(int[] inputShape) {
        this.inputShape = inputShape;
    }

    protected int[] getInputShape() {
        return inputShape;
    }

    protected void enableDebugForThisLayer() {
        this.debugEnabled = true;
    }
    protected void disableDebugForThisLayer() {
        this.debugEnabled = false;
    }
    protected boolean debugEnabled() {
        return debugEnabled;
    }

    public JMatrix getOutput() {
        return output;
    }

    public int[] outputShape() {
        return outputShape;
    }

    public JMatrix getGradient() {
        return gradient;
    }
    
    protected void cacheInput(JMatrix input, boolean training) {
        if (training) {
            lastInput = input;
        }
    }

    protected JMatrix getLastInput() {
        return lastInput;
    }

    protected JMatrix trackOutput(JMatrix output, boolean training) {
        this.outputShape = output.shape();
        if (training || debugEnabled || lastLayer.equals(this)) { // Last layer should store output
            this.output = output;
        } else {
            // Ensure memory is freed
            this.output = null;
        }
        return output;
    }

    private void clearInputCache() {
        lastInput = null;
    }
    
    protected JMatrix trackGradient(JMatrix gradient) {
        if (gradientStorageDisabled && !debugEnabled) {
            // Ensure memory is freed
            this.gradient = null;
        } else {
            this.gradient = gradient;
        }
        clearInputCache(); // Not needed after backward() is complete

        return gradient;
    }

    protected void disableGradientStorage() {
        gradientStorageDisabled = true;
    }
    

    // True if this layer alters shape from input to output
    protected boolean isShapeInfluencer() {
        return isShapeInfluencer;
    }

    protected boolean trainable() {
        return numTrainableParameters() != 0;
    }

    protected int numTrainableParameters() {
        return numTrainableParameters;
    }

    // For proper layer build after initialization
    protected void setNumTrainableParameters(int numTrainableParameters) {
        this.numTrainableParameters = numTrainableParameters;
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
