package jflow.model;

import java.util.HashMap;
import java.util.Map;
import jflow.data.JMatrix;
import jflow.layers.templates.TrainableLayer;

public class AdaGrad extends Optimizer<AdaGrad> {
    private double learningRate;
    private double epsilon = 1e-8;  // Small constant for numerical stability

    public AdaGrad(double learningRate) {
        super("adagrad");
        this.learningRate = learningRate;
    }

    public AdaGrad(double learningRate, double epsilon) {
        super("adagrad");
        this.learningRate = learningRate;
        this.epsilon = epsilon;
    }

    @Override
    public void applyUpdates(HashMap<String, JMatrix[]> layerGradients) {
        for (Map.Entry<String, JMatrix[]> entry : layerGradients.entrySet()) {
            TrainableLayer layer = getLayerID().get(entry.getKey());
            JMatrix[] gradients = entry.getValue();
            JMatrix[] accumSquaredGrads = getMoments().get(layer);
            JMatrix[] updates = new JMatrix[gradients.length];

            for (int i = 0; i < gradients.length; i++) {
                JMatrix weightGradients = gradients[i];
                JMatrix accumSquared = accumSquaredGrads[i];

                // Accumulate squared gradients
                accumSquared.addInPlace(weightGradients.multiply(weightGradients));

                JMatrix weightUpdate = weightGradients.divide(accumSquared.sqrt().addInPlace(epsilon));
                
                // Calculate parameter update
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
    public void initializeLayer(TrainableLayer layer) {
        JMatrix[] parameters = layer.getParameterGradients();
        JMatrix[] accumSquaredGrads = new JMatrix[parameters.length];
        
        for (int i = 0; i < parameters.length; i++) {
            // Initialize accumulated squared gradients matrix with zeros
            accumSquaredGrads[i] = parameters[i].zerosLike();
        }
        
        getMoments().put(layer, accumSquaredGrads);
        getLayerID().put(layer.getName(), layer);
    }
}