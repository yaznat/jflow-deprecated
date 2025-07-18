package jflow.model;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;

import jflow.data.JMatrix;
import jflow.layers.templates.TrainableLayer;

public abstract class Optimizer<T extends Optimizer<T>> {
    // LinkedHashMap preserves retrieval order - necessary since moments are identified by order, not name.
    private LinkedHashMap<TrainableLayer, JMatrix[]> layerMoments = new LinkedHashMap<>();
    private HashMap<String, TrainableLayer> layerID = new HashMap<>();
    private String name;
    private double clipThreshold = -1;

    protected abstract void applyUpdates(HashMap<String, JMatrix[]> layerGradients);
    protected abstract void initializeLayer(TrainableLayer layer);

    protected Optimizer(String name){
        this.name = name;
    }

    /**
     * Set the global clip norm of this optimizer. 
     * If the global l2 norm of weight updates exceeds 
     * the the threshold, all are multiplied by {@code threshold / global_norm}.
     * @param threshold the l2 norm threshold.
     */
    @SuppressWarnings("unchecked")
    public T clipNorm(double threshold) {
        this.clipThreshold = Math.abs(threshold); // Prevent negative values
        return (T) this;
    }

    private boolean useClipping() {
        return clipThreshold != -1;
    }

    protected void clipIfNecessary(HashMap<String, JMatrix[]> layerGradients) {
        if (!useClipping()) {
            return;
        }
        // Calculate global l2 norm
        double globalGradNormSquared = 0.0;
        for (JMatrix[] grads : layerGradients.values()) {
            for (JMatrix grad : grads) {
                double l2Norm = grad.l2Norm();
                globalGradNormSquared += l2Norm * l2Norm;
            }
        }
        double globalGradNorm = Math.sqrt(globalGradNormSquared);
        // Scale weight updates if necessary
        if (globalGradNorm > clipThreshold) {
            double clipScale = clipThreshold / (globalGradNorm + 1e-6);  // epsilon for stability
            for (JMatrix[] grads : layerGradients.values()) {
                for (JMatrix grad : grads) {
                    grad.multiplyInPlace(clipScale);
                }
            }
        }
    }

    public void apply(HashMap<String, JMatrix[]> layerGradients) {
        clipIfNecessary(layerGradients);
        applyUpdates(layerGradients);
    }

    protected String getName() {
        return name;
    }

    protected HashMap<TrainableLayer, JMatrix[]> getMoments() {
        return layerMoments;
    }
    protected HashMap<String, TrainableLayer> getLayerID() {
        return layerID;
    }

    protected JMatrix[] getSerializable() {
        int totalNumMoments = 0;
        // Count number of moments
        for (Map.Entry<TrainableLayer, JMatrix[]> entry : layerMoments.entrySet()) {
            totalNumMoments += entry.getValue().length;
        }
        // Assemble values into an array
        JMatrix[] moments = new JMatrix[totalNumMoments];
        int index = 0;
        for (Map.Entry<TrainableLayer, JMatrix[]> entry : layerMoments.entrySet()) {
            for (JMatrix moment : entry.getValue()) {
                moment.label(String.valueOf(index)); // Ensure that each weight has a unique label
                moments[index++] = moment;
            }
        }
        return moments;
    }
}
