package jflow.model;


import jflow.data.JMatrix;

public abstract class Layer {
    private Layer previousLayer;
    private Layer nextLayer;
    private Layer enclosingLayer;

    private JMatrix output;
    private JMatrix gradient;

    private int[] inputShape;

    private String type;

    private int numTrainableParameters;
    private int IDnum;

    private boolean isShapeInfluencer;
    private boolean gradientStorageDisabled = false;
        
    protected Layer(String type, boolean isShapeInfluencer) {
        this.type = type;
        this.isShapeInfluencer = isShapeInfluencer;
    }

    protected void build(int IDnum) {
        this.IDnum = IDnum;
        // if (getPreviousLayer() == null) {
        //     System.out.println(getName() + " connected to input");
        // } else {
        //     System.out.println(getName() + " connected to " + getPreviousLayer().getName());
        // }
        
    }

    public abstract JMatrix forward(JMatrix input, boolean training);

    public abstract JMatrix backward(JMatrix input);

    public abstract int[] outputShape();

    protected abstract JMatrix[] debugData();

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
        if (!gradientStorageDisabled) {
            this.gradient = gradient;
        }
        return gradient;
    }

    protected void setInputShape(int[] inputShape) {
        this.inputShape = inputShape;
    }

    protected void disableGradientStorage() {
        gradientStorageDisabled = true;
    }

    protected boolean gradientStorageDisabled() {
        return gradientStorageDisabled;
    }
    
    protected int[] internalGetInputShape() {
        return inputShape;
    }

    protected int[] getInputShape() {
        if (internalGetInputShape() == null) {
            return getPreviousLayer().outputShape();
        }
        return internalGetInputShape();
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
    protected Layer getNextLayer() {
        return nextLayer;
    }
    protected Layer getPreviousLayer() {
        return previousLayer;
    }

    protected void setNextLayer(Layer nextLayer) {
        this.nextLayer = nextLayer;
    }

    protected void setPreviousLayer(Layer previousLayer) {
        this.previousLayer = previousLayer;
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


    protected void setEnclosingLayer(Layer enclosingLayer) {
        this.enclosingLayer = enclosingLayer;
    }

    protected Layer getEnclosingLayer() {
        return enclosingLayer;
    }
    protected boolean isInternal() {
        return enclosingLayer != null;
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

    public void printDebug() {
        JMatrix[] debugData = debugData();
        if (!(debugData == null)) {
            String title = getName() + " gradients";
            System.out.println("\033[94m╭──────────────────── \033[0m\033[1;94m" + 
            title + "\033[0m\033[94m ────────────────────╮\033[0m");

            // Iterate over entries
            for (JMatrix data : debugData) {
                String dataName = data.getName();
                System.out.print("\033[94m│\033[0m " + "\033[38;2;0;153;153m" + dataName + "\033[0m");
                // Print statistics for the entry
                System.out.print(" \033[38;2;222;197;15m|\033[0m \033[38;2;255;165;0mshape:\033[0m \033[37m" + data.shapeAsString() + "\033[0m"); // shape
                System.out.print(" \033[38;2;222;197;15m|\033[0m \033[38;2;255;165;0mabsmax:\033[0m \033[37m" + data.absMax() + "\033[0m"); // absmax
                System.out.print(" \033[38;2;222;197;15m|\033[0m \033[38;2;255;165;0mabsmean:\033[0m \033[37m" + data.absMean() + "\033[0m"); // absmean
                System.out.print(" \033[38;2;222;197;15m|\033[0m \033[38;2;255;165;0mL1:\033[0m \033[37m" + data.l1Norm() + "\033[0m\n"); // L1 norm
            }
            String closer = "\033[94m╰─────────────────────";
            for (int i = 0; i < title.length(); i++) {
                closer += "─";
            }
            closer += "─────────────────────╯\033[0m";
            System.out.println(closer);
        }
        
    }
}
