package jflow.model;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import jflow.data.JMatrix;
import jflow.layers.templates.TrainableLayer;

public abstract class Optimizer {
    // LinkedHashMap preserves retrieval order, which is necessary
    private LinkedHashMap<TrainableLayer, JMatrix[]> layerMoments = new LinkedHashMap<>();
    private HashMap<String, TrainableLayer> layerID = new HashMap<>();
    private String name;
    private double threshold = -1;

    protected Optimizer(String name){
        this.name = name;
    }
    

    public abstract void apply(HashMap<String, JMatrix[]> layerGradients);

    /**
     * Set the global clip norm of this optimizer. 
     * If the global frobenius norm exceeds the the threshold,
     * all weights are scaled by the difference.
     * @param threshold the frobenius norm threshold.
     */
    public Optimizer clipNorm(double threshold) {
        this.threshold = Math.abs(threshold);
        return this;
    }

    protected double getClipNorm() {
        return threshold;
    }

    protected boolean useClipping() {
        return threshold != -1;
    }

    protected abstract void initializeLayer(TrainableLayer layer);

    protected String getName() {
        return name;
    }

    protected HashMap<TrainableLayer, JMatrix[]> getMoments() {
        return layerMoments;
    }
    protected HashMap<String, TrainableLayer> getLayerID() {
        return layerID;
    }

    protected JMatrix[] getWeights() {
        int totalNumWeights = 0;
        // Count number of weights
        for (Map.Entry<TrainableLayer, JMatrix[]> entry : layerMoments.entrySet()) {
            totalNumWeights += entry.getValue().length;
        }
        // Assemble values into an array
        JMatrix[] weights = new JMatrix[totalNumWeights];
        int index = 0;
        for (Map.Entry<TrainableLayer, JMatrix[]> entry : layerMoments.entrySet()) {
            for (JMatrix weight : entry.getValue()) {
                weight.label(String.valueOf(index)); // Ensure that each weight has a unique label
                weights[index++] = weight;
            }
        }
        return weights;
    }
}
