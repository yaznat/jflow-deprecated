package jflow.model;

import java.util.HashMap;
import java.util.Map;

import jflow.data.JMatrix;
import jflow.layers.templates.TrainableLayer;

public class Adam extends Optimizer {
    private double beta1; // Momentum coefficient of the first moment
    private double beta2; // Momentum coefficient of the second moment
    private double learningRate;
    private double epsilon = 1e-8; // Small constant for numerical stability
    private long timesteps = 0;

    protected Adam(double learningRate, double beta1, double beta2) {
        super("adam");
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.learningRate = learningRate;
    }

    protected Adam(double learningRate) {
        super("adam");
        this.beta1 = 0.9;
        this.beta2 = 0.999;
        this.learningRate = learningRate;
    }

    protected long getTimeSteps() {
        return timesteps;
    }
    protected void setTimeSteps(long timeSteps) {
        this.timesteps = timeSteps;
    }

    @Override
    public void applyUpdates(HashMap<String, JMatrix[]> layerGradients) {
        timesteps++;

        for (Map.Entry<String, JMatrix[]> entry : layerGradients.entrySet()) {
            TrainableLayer layer = getLayerID().get(entry.getKey());

            JMatrix[] gradients = entry.getValue();

            JMatrix[] moments = getMoments().get(layer);

            JMatrix[] updates = new JMatrix[gradients.length];

            
            for (int i = 0; i < gradients.length; i++) {
                JMatrix weightGradients = gradients[i];
                JMatrix mWeights = moments[2 * i];
                JMatrix vWeights = moments[2 * i + 1];

                // Update first moments (momentum)
                mWeights.multiplyInPlace(beta1).addInPlace(weightGradients.multiply(1 - beta1));

                // Update second moments (velocity)
                vWeights.multiplyInPlace(beta2).addInPlace(weightGradients.multiply(weightGradients).multiply(1 - beta2));

                // Calculate bias-corrected moments
                JMatrix mWeightsCorrected = mWeights.divide(1 - Math.pow(beta1, timesteps));
                JMatrix vWeightsCorrected = vWeights.divide(1 - Math.pow(beta2, timesteps));

                // Calculate parameter updates
                JMatrix weightUpdate = mWeightsCorrected.divideInPlace(
                    vWeightsCorrected.sqrt().addInPlace(epsilon));
                
                updates[i] = weightUpdate.multiplyInPlace(learningRate);
            }

            // Apply updates
            layer.updateParameters(updates);
            // Reset gradients
            for (JMatrix m : layer.getParameterGradients()) {
                m.fill(0);
            }
        }
    }

    @Override
    protected void initializeLayer(TrainableLayer layer) {
        JMatrix[] gradients = layer.getParameterGradients();
        int numWeights = gradients.length;

        JMatrix[] moments = new JMatrix[numWeights * 2];

        for (int i = 0; i < numWeights; i++) {
            // Initialized to zero
            JMatrix mWeights = gradients[i].zerosLike();
            moments[2 * i] = mWeights;
        
            JMatrix vWeights = gradients[i].zerosLike();
            moments[2 * i + 1] = vWeights;
        }

        getMoments().put(layer, moments);
        getLayerID().put(layer.getName(), layer);
    }
}
